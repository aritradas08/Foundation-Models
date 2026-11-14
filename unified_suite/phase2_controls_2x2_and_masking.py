#!/usr/bin/env python3
"""
Workstream 3: Physics & Reconstruction Controls Isolating Displaced Tracks vs. B-Field & Multiplicity:
  1. Truth-to-Reco Efficiency vs. Production Radius r_perp = sqrt(vx^2 + vy^2) across the 2x2 factorial
     arms {v1_prompt_2T, v1_prompt_3T, v2_displaced_2T, v2_displaced_3T} (runs 0..9).
  2. Complete 2x2 Factorial Benchmark ({Prompt vs. Displaced} x {2.0 T Seeding vs. 3.0 T Seeding})
     across representative models from every paradigm in both Binary Transfer Probes and Shape-Only NPLM.
     Crucially, v2_displaced_2T uses the EXACT same 2.0 T ACTS seed-finder config as all SM backgrounds!
  3. Impact-Parameter Masking Sensitivity Ablation ((d0, z0) = 0 applied BEFORE track scaling & VQ
     tokenization to SM reference, SM null toys, and BSM signals, with full null recalibration)
     + Multiplicity-Matched Signal Injection Control (reweighting signal draws so p_sig(N_tracks) == p_SM(N_tracks)).
"""
import argparse
import glob
import json
import os
import pickle
import sys
import time
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import uproot

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(REPO_ROOT / "Tokenization-scripts" / "scripts"))

from common import (
    MLPEncoder,
    SetTransformerEncoder,
    compute_summaries28_np,
    load_events_fast,
    raw6_to_feat7_np,
    raw6_to_vqvae6_np,
)
from evaluate_probes import make_stratified_split
from nplm_runner import compute_z_scores, evaluate_null_calibration, rbf_features_gpu, train_kernel_lbfgs
from phase2_nplm_calibrated import clopper_pearson_ci, gpfs_lock, order_stat_threshold, prepare_kernel_basis, run_null_toys_gpu
from phase2_prepare_and_extract import compute_handcrafted6_np, extract_set_transformer
from phase2_probes_and_scratch import train_probe_v2
from train_and_extract import set_seed

DATA_DIR = Path("/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/data")
CANON_EVAL_PARQUET = DATA_DIR / "eval_downstream_1050k.parquet"
HP_2X2_DIR = DATA_DIR / "hp_2x2_control"
V1_SRC_RUNS = Path("/global/cfs/cdirs/m4958/data/ColliderML/simulation/hard_scatter/higgs_portal/v1/runs")
V2_SRC_RUNS = Path("/global/cfs/cdirs/m4958/data/ColliderML/simulation/hard_scatter/higgs_portal/v2/runs")

RESULTS_BASE = Path("/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/results")
CKPT_BASE = RESULTS_BASE / "checkpoints"
EMBED_V2_DIR = RESULTS_BASE / "embeddings_v2"
CTRL_V2_DIR = RESULTS_BASE / "controls_v2"

REP_MODELS = [
    "supcon_d64", "supervised_e2e_d64",
    "ssl_kinematic_d64", "ssl_kinematic_d32",
    "ssl_tokens_d64", "ssl_tokens_d32", "ssl_tokens_d8",
    "ae_transformer_d32", "ae_mlp_d32",
    "vqvae_encoder_d128", "vqvae_code_d8", "vqvae_quant_d8",
    "raw_summary_d28", "pca_summary_d8", "phys_handcrafted_d6", "mult_only_d1",
    "random_init_d64",
]


def load_arm_parquet_dir(pq_dir: Path, out_merged_pq: Path, arm_label: int) -> dict[str, np.ndarray]:
    with gpfs_lock(out_merged_pq):
        if not out_merged_pq.exists():
            files = sorted(glob.glob(str(pq_dir / "*.parquet")))
            assert len(files) > 0, f"No parquet files found in {pq_dir}"
            tables = []
            for f in files:
                t = pq.read_table(f)
                lengths = pc.list_value_length(t["d0"]).to_numpy()
                keep = lengths > 0
                if not np.all(keep):
                    t = t.filter(pa.array(keep))
                if t.num_rows == 0:
                    continue
                if t["d0"].num_chunks > 1:
                    t = t.combine_chunks()
                offsets = t["d0"].chunk(0).offsets
                qop_flat = t["qop"].chunk(0).values.to_numpy().astype(np.float32)
                theta_flat = t["theta"].chunk(0).values.to_numpy().astype(np.float32)
                p_flat = (1.0 / np.maximum(np.abs(qop_flat), 1e-9)).astype(np.float32)
                pt_flat = (p_flat * np.sin(theta_flat)).astype(np.float32)
                n_rows = t.num_rows
                t = t.set_column(0, "event_id", pa.array(t["event_id"].to_numpy().astype(np.int64), type=pa.int64()))
                t = t.append_column("p", pa.ListArray.from_arrays(offsets, pa.array(p_flat, type=pa.float32())))
                t = t.append_column("pt", pa.ListArray.from_arrays(offsets, pa.array(pt_flat, type=pa.float32())))
                t = t.append_column("label", pa.array(np.full(n_rows, arm_label, dtype=np.int64), type=pa.int64()))
                tables.append(t)
            full = pa.concat_tables(tables)
            pq.write_table(full, out_merged_pq, compression="snappy")
            print(f"[2x2] Built {out_merged_pq} with {full.num_rows} valid non-empty events")
    return load_events_fast(out_merged_pq)


def compute_truth_tracking_efficiency() -> dict:
    """
    Computes truth-to-reco tracking efficiency vs production radius r_perp = sqrt(vx^2 + vy^2)
    on runs 0..4 (5,000 generated events per arm) across all 4 arms of the 2x2 factorial control,
    both for primary hard-scatter/BSM decay products (generation == 0) and all charged particles,
    plus event-level zero-track rates across all 10,000 generated events (runs 0..9).
    """
    arms = {
        "v1_prompt_2T": (V1_SRC_RUNS, V1_SRC_RUNS),
        "v1_prompt_3T": (V1_SRC_RUNS, HP_2X2_DIR / "v1_prompt_3T" / "runs"),
        "v2_displaced_2T": (V2_SRC_RUNS, HP_2X2_DIR / "v2_displaced_2T" / "runs"),
        "v2_displaced_3T": (V2_SRC_RUNS, V2_SRC_RUNS),
    }
    r_bins = [(0.0, 1.0), (1.0, 30.0), (30.0, 70.0), (70.0, 150.0), (150.0, 500.0), (500.0, 1e5)]
    out = {}

    for arm_name, (part_runs_dir, reco_runs_dir) in arms.items():
        # 1. Event-level track multiplicity & zero-track count across all 10 runs (10,000 generated events)
        n_gen_events = 0
        n_zero_track_events = 0
        ntrk_list = []
        ntrk_pt075_list = []
        frac_sub075_num, frac_sub075_den = 0, 0
        frac_d0_gt1_num = 0

        for r in range(10):
            ts_path = reco_runs_dir / str(r) / "tracksummary_ambi.root"
            arr = uproot.open(ts_path)["tracksummary"].arrays(["event_nr", "eLOC0_fit", "eTHETA_fit", "eQOP_fit"])
            n_tr = ak.to_numpy(ak.num(arr["eLOC0_fit"], axis=1))
            n_gen_events += 1000
            # Note: tracksummary has 1 entry per event processed by ACTS (1000 per run)
            n_empty_in_tree = int((n_tr == 0).sum()) + max(0, 1000 - len(n_tr))
            n_zero_track_events += n_empty_in_tree
            ntrk_list.append(n_tr[n_tr > 0])

            qop = np.abs(ak.to_numpy(ak.flatten(arr["eQOP_fit"], axis=1)))
            th = ak.to_numpy(ak.flatten(arr["eTHETA_fit"], axis=1))
            d0 = np.abs(ak.to_numpy(ak.flatten(arr["eLOC0_fit"], axis=1)))
            pt = np.sin(th) / np.maximum(qop, 1e-9)
            frac_sub075_num += int((pt < 0.75).sum())
            frac_sub075_den += len(pt)
            frac_d0_gt1_num += int((d0 > 1.0).sum())

            pt_ak = np.sin(arr["eTHETA_fit"]) / np.maximum(np.abs(arr["eQOP_fit"]), 1e-9)
            ntrk_pt075 = ak.to_numpy(ak.sum(pt_ak > 0.75, axis=1))
            ntrk_pt075_list.append(ntrk_pt075[n_tr > 0])

        ntrk_nonempty = np.concatenate(ntrk_list)
        ntrk_pt075_nonempty = np.concatenate(ntrk_pt075_list)

        # 2. Truth-matched efficiency vs r_perp on runs 0..4 (5,000 events)
        bin_counts_gen0 = {f"[{lo:g},{hi:g})": [0, 0] for lo, hi in r_bins}
        bin_counts_all = {f"[{lo:g},{hi:g})": [0, 0] for lo, hi in r_bins}
        fake_num, fake_den = 0, 0

        for r in range(5):
            p_arr = uproot.open(part_runs_dir / str(r) / "particles.root")["particles"].arrays(
                ["event_id", "vertex_primary", "vertex_secondary", "particle", "generation", "sub_particle", "vx", "vy", "pt", "eta", "q"]
            )
            t_arr = uproot.open(reco_runs_dir / str(r) / "tracksummary_ambi.root")["tracksummary"].arrays(
                ["event_nr", "majorityParticleId", "nMajorityHits"]
            )
            t_map = {}
            for ev, m, nh in zip(ak.to_numpy(t_arr["event_nr"]), t_arr["majorityParticleId"], t_arr["nMajorityHits"]):
                mlist = m.tolist()
                nh_np = ak.to_numpy(nh)
                fake_den += len(mlist)
                # Unmatched / fake tracks have vertex_primary == 0 ((0,0,0,0,0)) or nMajorityHits == 0
                valid_set = set()
                for key, hits in zip(mlist, nh_np.tolist()):
                    tup = tuple(key)
                    if tup[0] == 0 or hits == 0:
                        fake_num += 1
                    else:
                        valid_set.add(tup)
                t_map[int(ev)] = valid_set

            for i, ev in enumerate(ak.to_numpy(p_arr["event_id"])):
                q = ak.to_numpy(p_arr["q"][i])
                pt = ak.to_numpy(p_arr["pt"][i])
                eta = ak.to_numpy(p_arr["eta"][i])
                gen = ak.to_numpy(p_arr["generation"][i])
                vx = ak.to_numpy(p_arr["vx"][i])
                vy = ak.to_numpy(p_arr["vy"][i])
                r_perp = np.sqrt(vx ** 2 + vy ** 2)

                fid = (q != 0) & (pt > 1.0) & (np.abs(eta) < 3.0)
                if not fid.any():
                    continue
                vp = ak.to_numpy(p_arr["vertex_primary"][i])
                vs = ak.to_numpy(p_arr["vertex_secondary"][i])
                pa_id = ak.to_numpy(p_arr["particle"][i])
                sp = ak.to_numpy(p_arr["sub_particle"][i])
                mset = t_map.get(int(ev), set())

                for lo, hi in r_bins:
                    bkey = f"[{lo:g},{hi:g})"
                    s_all = fid & (r_perp >= lo) & (r_perp < hi)
                    if s_all.any():
                        keys = list(zip(vp[s_all].tolist(), vs[s_all].tolist(), pa_id[s_all].tolist(), gen[s_all].tolist(), sp[s_all].tolist()))
                        bin_counts_all[bkey][1] += len(keys)
                        bin_counts_all[bkey][0] += sum(1 for k in keys if k in mset)

                    s_g0 = s_all & (gen == 0)
                    if s_g0.any():
                        keys0 = list(zip(vp[s_g0].tolist(), vs[s_g0].tolist(), pa_id[s_g0].tolist(), gen[s_g0].tolist(), sp[s_g0].tolist()))
                        bin_counts_gen0[bkey][1] += len(keys0)
                        bin_counts_gen0[bkey][0] += sum(1 for k in keys0 if k in mset)

        out[arm_name] = {
            "n_generated_events": n_gen_events,
            "n_zero_track_events": n_zero_track_events,
            "zero_track_frac": n_zero_track_events / n_gen_events,
            "mean_ntrk_nonempty": float(ntrk_nonempty.mean()),
            "mean_ntrk_all_generated": float(ntrk_nonempty.sum() / n_gen_events),
            "mean_ntrk_pt_gt_075_nonempty": float(ntrk_pt075_nonempty.mean()),
            "frac_tracks_pt_lt_075": frac_sub075_num / max(frac_sub075_den, 1),
            "frac_tracks_abs_d0_gt_1mm": frac_d0_gt1_num / max(frac_sub075_den, 1),
            "fake_track_fraction": fake_num / max(fake_den, 1),
            "eff_vs_r_perp_primary_gen0_pt1GeV": {
                k: {"n_reco": v[0], "n_truth": v[1], "eff": (v[0] / v[1] if v[1] > 0 else np.nan)}
                for k, v in bin_counts_gen0.items()
            },
            "eff_vs_r_perp_all_charged_pt1GeV": {
                k: {"n_reco": v[0], "n_truth": v[1], "eff": (v[0] / v[1] if v[1] > 0 else np.nan)}
                for k, v in bin_counts_all.items()
            },
        }
        print(
            f"[truth_eff] {arm_name:18s} | zero_trk={out[arm_name]['zero_track_frac']*100:5.2f}% | "
            f"<Ntrk>={out[arm_name]['mean_ntrk_nonempty']:5.1f} (<pT>0.75>={out[arm_name]['mean_ntrk_pt_gt_075_nonempty']:5.1f}) | "
            f"frac(|d0|>1mm)={out[arm_name]['frac_tracks_abs_d0_gt_1mm']*100:5.2f}% | fake={out[arm_name]['fake_track_fraction']*100:4.2f}%",
            flush=True,
        )
    return out


class EmbeddingExtractorSuite:
    """Unified on-the-fly extractor for arbitrary (raw6, mask) arrays (normal or (d0,z0)=0 masked)."""
    def __init__(self, device: torch.device):
        self.device = device
        from vqvae_tracks import TrackVQVAE
        from pretrain_ssl_contrastive_tokens import TokenSetTransformerEncoder

        with open(CKPT_BASE / "physics_baselines" / "scalers.pkl", "rb") as f:
            self.base_sc = pickle.load(f)

        vq_dir = Path("/global/cfs/cdirs/m4958/usr/aritra08/token/vqvae_tracks/sm_only_v1")
        with open(vq_dir / "scaler.pkl", "rb") as f:
            self.vq_scaler = pickle.load(f)
        vq_ckpt = torch.load(vq_dir / "vqvae_best.pt", map_location=device)
        self.vqvae = TrackVQVAE(**vq_ckpt["cfg"]).to(device)
        self.vqvae.load_state_dict(vq_ckpt["state_dict"])
        self.vqvae.eval()
        self.TokenCls = TokenSetTransformerEncoder

    def extract_models(self, raw6: np.ndarray, mask: np.ndarray, model_names: list[str]) -> dict[str, np.ndarray]:
        feat7 = raw6_to_feat7_np(raw6, mask)
        N = len(raw6)
        out = {}

        # Physics baselines
        if any(m in model_names for m in ["raw_summary_d28", "pca_summary_d8"]):
            s28 = compute_summaries28_np(feat7, mask, scaler=self.base_sc["trk_scaler"])
            s28_s = self.base_sc["sum28_scaler"].transform(s28).astype(np.float32)
            if "raw_summary_d28" in model_names:
                out["raw_summary_d28"] = s28_s
            if "pca_summary_d8" in model_names:
                out["pca_summary_d8"] = self.base_sc["pca8_scaler"].transform(self.base_sc["pca8"].transform(s28_s)).astype(np.float32)

        if any(m in model_names for m in ["phys_handcrafted_d6", "mult_only_d1"]):
            hc6 = compute_handcrafted6_np(raw6, mask)
            if "phys_handcrafted_d6" in model_names:
                out["phys_handcrafted_d6"] = self.base_sc["hc6_scaler"].transform(hc6).astype(np.float32)
            if "mult_only_d1" in model_names:
                out["mult_only_d1"] = self.base_sc["mult1_scaler"].transform(hc6[:, 0:1]).astype(np.float32)

        # VQ-VAE & Token SSL
        vq_needed = [m for m in model_names if m.startswith("vqvae_") or m.startswith("ssl_tokens_")]
        if vq_needed:
            vq6, m_vq = raw6_to_vqvae6_np(raw6, mask)
            vq_mean = self.vq_scaler.mean_.astype(np.float32).reshape(1, 1, 6)
            vq_scale = self.vq_scaler.scale_.astype(np.float32).reshape(1, 1, 6)
            z_e128, z_pre8, z_q8 = [], [], []
            toks = np.full((N, vq6.shape[1]), -1, dtype=np.int16)
            with torch.no_grad():
                for st in range(0, N, 1024):
                    en = min(st + 1024, N)
                    mb = m_vq[st:en]
                    xb = (vq6[st:en] - vq_mean) / vq_scale
                    xb[~mb] = 0.0
                    xt = torch.from_numpy(xb).to(self.device)
                    mt = torch.from_numpy(mb).to(self.device)
                    h = self.vqvae.encoder(self.vqvae.enc_in(xt), src_key_padding_mask=~mt)
                    hn = self.vqvae.enc_out[0](h)
                    ze = self.vqvae.enc_out[1](hn)
                    zq, idx, _, _, _ = self.vqvae.vq(ze, mt)
                    mf = mt.unsqueeze(-1).float()
                    den = mf.sum(dim=1).clamp(min=1.0)
                    if "vqvae_encoder_d128" in model_names:
                        z_e128.append(((hn * mf).sum(dim=1) / den).cpu().numpy())
                    if "vqvae_code_d8" in model_names:
                        z_pre8.append(((ze * mf).sum(dim=1) / den).cpu().numpy())
                    if "vqvae_quant_d8" in model_names:
                        z_q8.append(((zq * mf).sum(dim=1) / den).cpu().numpy())
                    toks[st:en] = idx.cpu().numpy().astype(np.int16)
            if "vqvae_encoder_d128" in model_names:
                out["vqvae_encoder_d128"] = np.concatenate(z_e128, axis=0)
            if "vqvae_code_d8" in model_names:
                out["vqvae_code_d8"] = np.concatenate(z_pre8, axis=0)
            if "vqvae_quant_d8" in model_names:
                out["vqvae_quant_d8"] = np.concatenate(z_q8, axis=0)

            for m in [x for x in model_names if x.startswith("ssl_tokens_d")]:
                d = int(m.split("_d")[-1])
                cdir = CKPT_BASE / m
                if not (cdir / "encoder.pt").exists():
                    continue
                tenc = self.TokenCls(codebook_size=8192, latent_dim=d).to(self.device)
                tenc.load_state_dict(torch.load(cdir / "encoder.pt", map_location=self.device))
                tenc.eval()
                zl = []
                with torch.no_grad():
                    for st in range(0, N, 2048):
                        en = min(st + 2048, N)
                        tb = torch.from_numpy(toks[st:en].astype(np.int64)).clamp(min=0).to(self.device)
                        mb = torch.from_numpy(m_vq[st:en]).to(self.device)
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(self.device.type == "cuda")):
                            zl.append(tenc(tb, mb).float().cpu().numpy())
                out[m] = np.concatenate(zl, axis=0)

        # SupCon, AE_MLP, Random, Unified SetTransformers
        for m in model_names:
            if m in out:
                continue
            if m.startswith("supcon_d"):
                d = int(m.split("_d")[-1])
                cdir = Path(f"/global/cfs/cdirs/m4958/usr/aritra08/cont_learn/supervised/results/d{d}/checkpoints")
                with open(cdir / "scaler.pkl", "rb") as f:
                    sc = pickle.load(f)
                enc = SetTransformerEncoder(latent_dim=d, norm_first=True, activation="gelu", use_out_norm=True).to(self.device)
                enc.load_state_dict(torch.load(cdir / "encoder_supervised_simclr.pt", map_location=self.device))
                out[m] = extract_set_transformer(enc, feat7, mask, sc, self.device, max_t_cap=256)
            elif m.startswith("ae_mlp_d"):
                d = int(m.split("_d")[-1])
                mlp_runs = {8: "sm_only_v1_run2", 32: "sm_only_v2_run2", 64: "sm_only_v3_run2"}
                cdir = Path(f"/global/cfs/cdirs/m4958/usr/aritra08/autoencoders/mlp/results/{mlp_runs[d]}/checkpoints")
                with open(cdir / "scaler.pkl", "rb") as f:
                    tsc = pickle.load(f)
                with open(cdir / "target_scaler.pkl", "rb") as f:
                    tgsc = pickle.load(f)
                s28 = compute_summaries28_np(feat7, mask, scaler=tsc)
                s28_s = tgsc.transform(s28).astype(np.float32)
                enc = MLPEncoder(latent_dim=d).to(self.device)
                enc.load_state_dict(torch.load(cdir / "encoder_ae_mlp.pt", map_location=self.device))
                enc.eval()
                zc = []
                with torch.no_grad():
                    for st in range(0, N, 16384):
                        zc.append(enc(torch.from_numpy(s28_s[st:st + 16384]).to(self.device)).cpu().numpy())
                out[m] = np.concatenate(zc, axis=0)
            elif m.startswith("random_init_d"):
                d = int(m.split("_d")[-1])
                set_seed(1000 + d)
                enc = SetTransformerEncoder(latent_dim=d, norm_first=True, activation="gelu", use_out_norm=True).to(self.device)
                out[m] = extract_set_transformer(enc, feat7, mask, self.base_sc["trk_scaler"], self.device, max_t_cap=128)
            else:
                cdir = CKPT_BASE / m
                if (cdir / "encoder.pt").exists():
                    d = int(m.split("_d")[-1])
                    with open(cdir / "scaler.pkl", "rb") as f:
                        sc = pickle.load(f)
                    enc = SetTransformerEncoder(latent_dim=d, norm_first=True, activation="gelu", use_out_norm=True).to(self.device)
                    enc.load_state_dict(torch.load(cdir / "encoder.pt", map_location=self.device))
                    out[m] = extract_set_transformer(enc, feat7, mask, sc, self.device, max_t_cap=128)
        return out


def eval_custom_signal_nplm_and_probe(
    z_sm: np.ndarray,
    n_tracks_sm: np.ndarray,
    sig_dict: dict[str, tuple[np.ndarray, np.ndarray]],
    device: torch.device,
    n_null: int = 500,
    n_sig_toys: int = 100,
) -> list[dict]:
    """
    Given SM embeddings z_sm (N=750,000) and dict of signal arms {arm_name: (z_sig, n_tracks_sig)},
    evaluates:
      1. Binary supervised transfer probe ROC-AUC (balanced 20k SM vs arm, 70/10/20 split)
      2. Shape-only Kernel NPLM (N_R=500k, N_B=100k, M=316, 500 cal + 500 val nulls, 100 signal toys)
         at f_inj in {0.005, 0.01, 0.025}, BOTH unweighted and N_tracks-distribution-matched!
    """
    dummy_labels = np.zeros(len(z_sm), dtype=np.int64)
    basis = prepare_kernel_basis(z_sm, dummy_labels, 500_000, 316, ref_seed=42, device=device)

    # Re-fit identical scaler on R to transform custom signal pools
    rng_ref = np.random.default_rng(42)
    perm_bkg = rng_ref.permutation(len(z_sm))
    from sklearn.preprocessing import StandardScaler
    sc_r = StandardScaler().fit(z_sm[perm_bkg[:500_000]])

    t_cal = run_null_toys_gpu(basis, 500_000, 100_000, n_null, toy_seed=91_001)
    gof = evaluate_null_calibration(t_cal)
    thr_1e2 = order_stat_threshold(t_cal, 1e-2)

    # Precompute SM N_tracks histogram for multiplicity-matched sampling weights
    max_bin = 257
    sm_hist = np.bincount(np.clip(n_tracks_sm, 0, max_bin - 1), minlength=max_bin).astype(np.float64)
    sm_prob_ntrk = sm_hist / sm_hist.sum()

    rng_pr = np.random.default_rng(42)
    sm_sub_probe = rng_pr.choice(len(z_sm), size=20_000, replace=False)

    out_rows = []
    for arm_name, (z_sig, ntrk_sig) in sig_dict.items():
        # 1. Binary Probe AUC
        idx_y = np.concatenate([np.zeros(len(sm_sub_probe), dtype=np.int64), np.ones(len(z_sig), dtype=np.int64)])
        z_bin = np.concatenate([z_sm[sm_sub_probe], z_sig], axis=0)
        b_tr, b_va, b_te = make_stratified_split(idx_y, 0.70, 0.10, seed=77)
        pr_res = train_probe_v2(z_bin[b_tr], idx_y[b_tr], z_bin[b_va], idx_y[b_va], z_bin[b_te], idx_y[b_te], 2, device, "linear", batch_size=512, compute_boot=True)

        # Multiplicity-matching weights w(N_trk) = p_SM(N_trk) / p_sig(N_trk)
        sig_hist = np.bincount(np.clip(ntrk_sig, 0, max_bin - 1), minlength=max_bin).astype(np.float64)
        sig_prob_ntrk = np.maximum(sig_hist / sig_hist.sum(), 1e-8)
        w_mult = sm_prob_ntrk[np.clip(ntrk_sig, 0, max_bin - 1)] / sig_prob_ntrk[np.clip(ntrk_sig, 0, max_bin - 1)]
        w_mult = w_mult / w_mult.sum()
        ess_mult = float(1.0 / np.sum(w_mult ** 2))

        sig_scaled_t = torch.from_numpy(sc_r.transform(z_sig).astype(np.float32)).to(device=device, dtype=torch.float64)
        n_pool = basis["n_pool"]

        for mult_matched in [False, True]:
            for frac in [0.005, 0.01, 0.025]:
                n_s = int(round(frac * 100_000))
                n_b_draw = 100_000 - n_s
                rng_t = np.random.default_rng(95_000 + int(mult_matched) * 1000 + int(frac * 10000))
                ts = np.empty(n_sig_toys, dtype=np.float64)
                for ti in range(n_sig_toys):
                    ib = rng_t.choice(n_pool, size=n_b_draw, replace=False)
                    if not mult_matched:
                        isig = rng_t.choice(len(z_sig), size=n_s, replace=(n_s > len(z_sig)))
                    else:
                        isig = rng_t.choice(len(z_sig), size=n_s, replace=True, p=w_mult)
                    dt = torch.cat([
                        basis["bkg_pool_t"][torch.from_numpy(ib).to(device)],
                        sig_scaled_t[torch.from_numpy(isig).to(device)],
                    ], dim=0)
                    phi_d = rbf_features_gpu(dt, basis["centers_t"], basis["sigma"])
                    _, conv = train_kernel_lbfgs(basis["phi_ref"], phi_d, basis["k_mm"], 100_000, 500_000)
                    ts[ti] = conv["t_final"]

                t_med = float(np.median(ts))
                zd = compute_z_scores(t_med, t_cal, gof)
                k = int((ts > thr_1e2).sum())
                lo, hi = clopper_pearson_ci(k, len(ts))
                out_rows.append({
                    "arm": arm_name,
                    "mult_matched": mult_matched,
                    "ess_mult_pool": ess_mult if mult_matched else float(len(z_sig)),
                    "probe_auc": pr_res["macro_auc"],
                    "probe_auc_se": pr_res["macro_auc_se"],
                    "frac": frac,
                    "null_mean": gof["mean"],
                    "null_std": gof["std"],
                    "t_median": t_med,
                    "t_q16": float(np.percentile(ts, 16)),
                    "t_q84": float(np.percentile(ts, 84)),
                    "snr_median": zd["snr"],
                    "z_asymptotic": zd["z_asymptotic"],
                    "power_1e2": k / len(ts),
                    "power_1e2_ci_lo": lo,
                    "power_1e2_ci_hi": hi,
                })
        del sig_scaled_t

    del basis
    torch.cuda.empty_cache()
    return out_rows


def run_2x2_and_masking_controls(device: torch.device, models_filter: list[str] | None = None):
    CTRL_V2_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Truth-matched tracking efficiency across all 4 arms
    eff_path = CTRL_V2_DIR / "hp_2x2_truth_efficiency.json"
    with gpfs_lock(eff_path, timeout=300.0):
        if not eff_path.exists():
            print("\n=== Step 3.1: Measuring Truth-to-Reco Efficiency Across 2x2 Factorial Arms ===")
            eff_stats = compute_truth_tracking_efficiency()
            with open(eff_path, "w") as f:
                json.dump(eff_stats, f, indent=2)
        else:
            print(f"[truth_eff] Using cached {eff_path}")

    # 2. Load 4 factorial arms (runs 0..9, ~10,000 events per arm)
    arm_data = {}
    for idx_arm, aname in enumerate(["v1_prompt_2T", "v1_prompt_3T", "v2_displaced_2T", "v2_displaced_3T"]):
        arm_data[aname] = load_arm_parquet_dir(
            HP_2X2_DIR / aname / "parquet",
            HP_2X2_DIR / f"{aname}_runs0_9.parquet",
            arm_label=10 + idx_arm,
        )

    extractor = EmbeddingExtractorSuite(device)
    base_list = models_filter if models_filter else REP_MODELS
    avail_models = [m for m in base_list if (EMBED_V2_DIR / f"{m}.npz").exists()]

    p_2x2 = CTRL_V2_DIR / "hp_2x2_factorial_results.csv"
    done_2x2 = set(pd.read_csv(p_2x2)["model"]) if p_2x2.exists() else set()
    rem_2x2 = [m for m in avail_models if m not in done_2x2]
    print(f"\n=== Step 3.2: Extracting & Evaluating 2x2 Factorial Arms Across {len(rem_2x2)} Remaining Models (cached={len(avail_models)-len(rem_2x2)}) ===")
    arm_embeds = {aname: extractor.extract_models(ad["raw6"], ad["mask"], rem_2x2) for aname, ad in arm_data.items()} if rem_2x2 else {}

    rows_2x2 = []
    for mname in rem_2x2:
        t0 = time.time()
        d_canon = np.load(EMBED_V2_DIR / f"{mname}.npz")
        sm_mask = np.isin(d_canon["labels"], [0, 1, 2])
        z_sm = d_canon["z"][sm_mask].astype(np.float32)
        ntrk_sm = d_canon["n_tracks"][sm_mask]

        sig_dict = {aname: (arm_embeds[aname][mname], arm_data[aname]["lengths"]) for aname in arm_data}
        m_rows = eval_custom_signal_nplm_and_probe(z_sm, ntrk_sm, sig_dict, device)
        for r in m_rows:
            rows_2x2.append({"model": mname, **r})

        r_v1_2t = [x for x in m_rows if x["arm"] == "v1_prompt_2T" and not x["mult_matched"] and x["frac"] == 0.01][0]
        r_v2_2t = [x for x in m_rows if x["arm"] == "v2_displaced_2T" and not x["mult_matched"] and x["frac"] == 0.01][0]
        r_v2_2t_mm = [x for x in m_rows if x["arm"] == "v2_displaced_2T" and x["mult_matched"] and x["frac"] == 0.01][0]
        r_v2_3t = [x for x in m_rows if x["arm"] == "v2_displaced_3T" and not x["mult_matched"] and x["frac"] == 0.01][0]
        print(
            f"[2x2] {mname:22s} | ProbeAUC: v1_2T={r_v1_2t['probe_auc']:.4f} -> v2_2T={r_v2_2t['probe_auc']:.4f} (v2_3T={r_v2_3t['probe_auc']:.4f}) | "
            f"NPLM SNR(1%): v1_2T={r_v1_2t['snr_median']:5.2f} -> v2_2T={r_v2_2t['snr_median']:5.2f} [MultMatched={r_v2_2t_mm['snr_median']:5.2f}] (v2_3T={r_v2_3t['snr_median']:5.2f}) ({time.time()-t0:.1f}s)",
            flush=True,
        )

        p_2x2 = CTRL_V2_DIR / "hp_2x2_factorial_results.csv"
        with gpfs_lock(p_2x2):
            df_2x2 = pd.DataFrame(rows_2x2)
            if p_2x2.exists():
                old = pd.read_csv(p_2x2)
                old = old[~old["model"].isin(df_2x2["model"])]
                df_2x2 = pd.concat([old, df_2x2], ignore_index=True)
            df_2x2.to_csv(p_2x2, index=False)
    print(f"Saved {CTRL_V2_DIR / 'hp_2x2_factorial_results.csv'}")

    # 3. Impact-Parameter Masking Sensitivity Ablation: (d0, z0) = 0 BEFORE scaling & tokenization
    #    on BOTH SM (750k) and signals (HP_v2_100k, HP_v2_2T_10k, HV_100k), with full null recalibration!
    p_mask = CTRL_V2_DIR / "impact_param_zeroed_ablation.csv"
    done_mask = set(pd.read_csv(p_mask)["model"]) if p_mask.exists() else set()
    rem_mask = [m for m in avail_models if m not in done_mask]
    print(f"\n=== Step 3.3: Running (d0, z0) = 0 Pre-Scaling Sensitivity Ablation ({len(rem_mask)} models, cached={len(avail_models)-len(rem_mask)}) ===")
    if not rem_mask:
        print(f"All requested models already in {p_mask}")
        return
    eval_data = load_events_fast(CANON_EVAL_PARQUET)
    raw6_masked = eval_data["raw6"].copy()
    raw6_masked[:, :, 0] = 0.0  # symlog(0) = 0
    raw6_masked[:, :, 1] = 0.0  # symlog(0) = 0
    mask_all = eval_data["mask"]
    labels_all = eval_data["labels"]
    ntrk_all = eval_data["lengths"]
    del eval_data

    z_masked_all = extractor.extract_models(raw6_masked, mask_all, rem_mask)
    del raw6_masked

    raw6_v2_2t_masked = arm_data["v2_displaced_2T"]["raw6"].copy()
    raw6_v2_2t_masked[:, :, 0] = 0.0
    raw6_v2_2t_masked[:, :, 1] = 0.0
    z_v2_2t_masked = extractor.extract_models(raw6_v2_2t_masked, arm_data["v2_displaced_2T"]["mask"], rem_mask)

    sm_m = np.isin(labels_all, [0, 1, 2])
    hp_m = labels_all == 3
    hv_m = labels_all == 4

    rows_mask = []
    for mname in rem_mask:
        t0 = time.time()
        zm = z_masked_all[mname]
        sig_m_dict = {
            "higgs_portal_v2_d0z0_zeroed": (zm[hp_m], ntrk_all[hp_m]),
            "v2_displaced_2T_d0z0_zeroed": (z_v2_2t_masked[mname], arm_data["v2_displaced_2T"]["lengths"]),
            "hidden_valley_d0z0_zeroed": (zm[hv_m], ntrk_all[hv_m]),
        }
        m_rows = eval_custom_signal_nplm_and_probe(zm[sm_m], ntrk_all[sm_m], sig_m_dict, device)
        for r in m_rows:
            rows_mask.append({"model": mname, **r})

        rhp = [x for x in m_rows if x["arm"] == "v2_displaced_2T_d0z0_zeroed" and not x["mult_matched"] and x["frac"] == 0.01][0]
        rhp_mm = [x for x in m_rows if x["arm"] == "v2_displaced_2T_d0z0_zeroed" and x["mult_matched"] and x["frac"] == 0.01][0]
        print(
            f"[d0z0=0] {mname:22s} | v2_2T(d0z0=0) ProbeAUC={rhp['probe_auc']:.4f} | "
            f"NPLM SNR(1%)={rhp['snr_median']:5.2f} [MultMatched={rhp_mm['snr_median']:5.2f}] ({time.time()-t0:.1f}s)",
            flush=True,
        )

        p_mask = CTRL_V2_DIR / "impact_param_zeroed_ablation.csv"
        with gpfs_lock(p_mask):
            df_m = pd.DataFrame(rows_mask)
            if p_mask.exists():
                old = pd.read_csv(p_mask)
                old = old[~old["model"].isin(df_m["model"])]
                df_m = pd.concat([old, df_m], ignore_index=True)
            df_m.to_csv(p_mask, index=False)
    print(f"Saved {CTRL_V2_DIR / 'impact_param_zeroed_ablation.csv'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, nargs="*", default=None)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    run_2x2_and_masking_controls(device, args.models)


if __name__ == "__main__":
    main()
