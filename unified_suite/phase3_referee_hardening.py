#!/usr/bin/env python3
"""
Phase 3 Referee-Hardening Suite for ColliderML Representation Learning Paper:
  1. supcon_subspace:
       - Decomposes supcon_d64 linear probe weights (HP_v2, HV) into the SM PCA eigenbasis
         (PCs 1..2, PCs 1..3, PCs 4..16, PCs 4..64).
       - Evaluates paired binary probe AUC and paired shape-only Kernel NPLM across subspaces
         (raw vs whitened PCs, plus top-8 and top-16 whitened PCs) to explain quantitatively
         why supcon_d64 achieves 0.9414 supervised transfer AUC on HP_v2 yet SNR=3.57 in
         unsupervised 64-D RBF NPLM.
  2. bandwidth_zca_sweep:
       - Sweeps ZCA shrinkage gamma in {1e-1, 1e-2, 1e-4, 1e-6} on supcon_d64, ssl_kinematic_d64,
         and raw_summary_d28.
       - Sweeps RBF bandwidth quantile q in {0.10, 0.25, 0.50, 0.90} (M=316) and Nystrom centers
         M in {316, 1000} (q=0.90) across key representatives.
  3. broken_aug_t128:
       - Single-variable T=128 control (ssl_legacy_aug_t128_d64): identical 900k SM data, T=128
         truncation, Pre-LN GELU 8L/8H SetTransformer, and 20-epoch schedule as ssl_kinematic_d64,
         changing ONLY the augmentation back to legacy global azimuthal rotation + independent
         additive Gaussian noise on standardized (d0, z0, p, pt) with zero angular smearing.
  4. pretrain_seed:
       - Trains 2 additional pretraining seeds (seeds 1, 2) at D=64 for ssl_kinematic_d64 (20 ep)
         and ssl_tokens_d64 (12 ep), extracts 1.05M embeddings, and runs paired probes + shape NPLM.
  5. finetune_curves:
       - Unfreezes pretrained ssl_kinematic_d64 and ssl_tokens_d64 end-to-end across
         N_labeled in {100, 1000, 10000, 100000} x 5 seeds on the exact splits of supervised_scratch_d64.
  6. summarize_all:
       - Consolidates all Phase 3 tables and 3-seed mean +/- SD metrics into phase3_summary.json.
"""
import argparse
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, label_binarize

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(REPO_ROOT / "Tokenization-scripts" / "scripts"))

from common import (
    PROCESS_NAMES,
    ProjectionHead,
    SM_LABELS,
    SetTransformerEncoder,
    augment_batch_kinematic_torch,
    fit_track_scaler,
    load_events_fast,
    raw6_to_feat7_np,
    raw6_to_vqvae6_np,
)
from evaluate_probes import make_stratified_split
from nplm_runner import (
    choose_centers_and_sigma,
    compute_z_scores,
    evaluate_null_calibration,
    rbf_features_gpu,
    train_kernel_lbfgs,
)
from phase2_nplm_calibrated import (
    ZCAShrinkageWhitener,
    clopper_pearson_ci,
    gpfs_lock,
    order_stat_threshold,
    run_null_toys_gpu,
    run_signal_toys_gpu,
)
from phase2_prepare_and_extract import compute_effective_rank, extract_set_transformer
from phase2_probes_and_scratch import train_probe_v2
from train_and_extract import set_seed, simclr_nt_xent_loss

DATA_DIR = Path("/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/data")
PRETRAIN_PARQUET = DATA_DIR / "pretrain_sm_900k.parquet"
CANON_EVAL_PARQUET = DATA_DIR / "eval_downstream_1050k.parquet"
RESULTS_BASE = Path("/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/results")
CKPT_V2_DIR = RESULTS_BASE / "checkpoints"
EMBED_V2_DIR = RESULTS_BASE / "embeddings_v2"
PROBE_V2_DIR = RESULTS_BASE / "probes_v2"
NPLM_V2_DIR = RESULTS_BASE / "nplm_v2"
CACHE_DIR = Path("/pscratch/sd/d/danieltm/fm_cache")

P3_DIR = RESULTS_BASE / "phase3_hardening"
P3_CKPT_DIR = P3_DIR / "checkpoints"
P3_EMBED_DIR = P3_DIR / "embeddings"
P3_PROBE_DIR = P3_DIR / "probes"
P3_NPLM_DIR = P3_DIR / "nplm"


def augment_batch_legacy_torch(
    raw6: torch.Tensor,
    mask: torch.Tensor,
    scaler_mean: torch.Tensor,
    scaler_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Exact GPU implementation of Aritra's legacy SSL augmentation (cont_learn/ssl/pretrain.py lines 91-110):
      1. Global event-level azimuthal rotation angle ~ U(0, 2*pi) applied rigidly to (tx, ty),
         with ZERO per-track angular smearing on phi or theta (leaving tz = cos(theta) and all
         pairwise Delta_phi exact invariants across positive views).
      2. Standardize [symlog(d0), symlog(z0), p, pt, tx_rot, ty_rot, tz] using pretrain scaler.
      3. Add independent Gaussian noise in standardized space ONLY to d0 (0.10), z0 (0.10),
         p (0.05), and pt (0.05), violating pT = p * sin(theta).
    """
    B, T, _ = raw6.shape
    device = raw6.device

    d0 = raw6[:, :, 0]
    z0 = raw6[:, :, 1]
    p = raw6[:, :, 2]
    pt = raw6[:, :, 3]
    phi = raw6[:, :, 4]
    theta = raw6[:, :, 5]

    angle = torch.rand((B, 1), device=device, dtype=raw6.dtype) * (2.0 * torch.pi)
    phi_rot = phi + angle
    sin_t = torch.sin(theta)
    tx_rot = torch.cos(phi_rot) * sin_t
    ty_rot = torch.sin(phi_rot) * sin_t
    tz_exact = torch.cos(theta)

    feat7 = torch.stack([d0, z0, p, pt, tx_rot, ty_rot, tz_exact], dim=-1)
    feat7_s = (feat7 - scaler_mean.view(1, 1, 7)) / scaler_scale.view(1, 1, 7)

    noise = torch.zeros_like(feat7_s)
    noise[:, :, 0] = torch.randn((B, T), device=device, dtype=raw6.dtype) * 0.10
    noise[:, :, 1] = torch.randn((B, T), device=device, dtype=raw6.dtype) * 0.10
    noise[:, :, 2] = torch.randn((B, T), device=device, dtype=raw6.dtype) * 0.05
    noise[:, :, 3] = torch.randn((B, T), device=device, dtype=raw6.dtype) * 0.05
    feat7_s = (feat7_s + noise).masked_fill(~mask.unsqueeze(-1), 0.0)
    return feat7_s, mask


def save_p3_embedding(name: str, z: np.ndarray, eval_data: dict[str, np.ndarray], provenance: dict) -> Path:
    P3_EMBED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = P3_EMBED_DIR / f"{name}.npz"
    sm_mask = np.isin(eval_data["labels"], [0, 1, 2])
    erank_sm = compute_effective_rank(z[sm_mask])
    embed_std_sm = float(z[sm_mask].std(axis=0).mean())
    prov = {
        "model": name,
        "latent_dim": int(z.shape[1]),
        "n_events": int(z.shape[0]),
        "erank_sm": erank_sm,
        "embed_std_sm": embed_std_sm,
        **provenance,
    }
    payload = {
        "z": z.astype(np.float32),
        "labels": eval_data["labels"].astype(np.int64),
        "event_ids": eval_data["event_ids"].astype(np.int64),
        "eval_uid": np.arange(len(eval_data["labels"]), dtype=np.int64),
        "n_tracks": eval_data["lengths"].astype(np.int32),
        "provenance_json": np.array(json.dumps(prov)),
    }
    tmp = out_path.with_suffix(".tmp.npz")
    np.savez(tmp, **payload)
    os.replace(tmp, out_path)
    print(f"[save_p3] {name:28s} | shape={str(z.shape):14s} | erank_SM={erank_sm:5.2f} | std_SM={embed_std_sm:.4f}", flush=True)
    return out_path


def evaluate_p3_probes_and_nplm(npz_path: Path, device: torch.device):
    """Runs the exact Phase 2 paired probes and paired shape-only NPLM on a Phase 3 embedding."""
    P3_PROBE_DIR.mkdir(parents=True, exist_ok=True)
    P3_NPLM_DIR.mkdir(parents=True, exist_ok=True)
    mname = npz_path.stem
    data = np.load(npz_path)
    z = data["z"].astype(np.float32)
    labels = data["labels"]
    D = z.shape[1]
    prov = json.loads(str(data["provenance_json"])) if "provenance_json" in data.files else {}

    # 1. Paired Probes (exact splits matching phase2_probes_and_scratch.py)
    probe_json = P3_PROBE_DIR / f"{mname}_probe_detail.json"
    if not probe_json.exists():
        t0 = time.time()
        sm_indices = np.where(np.isin(labels, [0, 1, 2]))[0]
        sm_labels = labels[sm_indices]
        sm_tr, sm_va, sm_te = make_stratified_split(sm_labels, 0.70, 0.10, seed=42)

        c5_indices = np.where(np.isin(labels, [0, 1, 2, 3, 4]))[0]
        c5_labels = labels[c5_indices]
        c5_tr, c5_va, c5_te = make_stratified_split(c5_labels, 0.70, 0.10, seed=42)

        rng_bin = np.random.default_rng(42)
        sm_sub = rng_bin.choice(sm_indices, size=60_000, replace=False)
        bin_results = {}
        for bsm_lbl, bsm_name in [(3, "sm_vs_higgs_portal_v2"), (4, "sm_vs_hidden_valley"), (5, "sm_vs_higgs_portal_v1")]:
            bsm_idx = np.where(labels == bsm_lbl)[0]
            idx_all = np.concatenate([sm_sub, bsm_idx])
            y_bin = np.concatenate([np.zeros(len(sm_sub), dtype=np.int64), np.ones(len(bsm_idx), dtype=np.int64)])
            b_tr, b_va, b_te = make_stratified_split(y_bin, 0.70, 0.10, seed=42 + bsm_lbl)
            z_b = z[idx_all]
            bin_results[bsm_name] = train_probe_v2(z_b[b_tr], y_bin[b_tr], z_b[b_va], y_bin[b_va], z_b[b_te], y_bin[b_te], 2, device, "linear", batch_size=1024)

        z_sm = z[sm_indices]
        res_sm3_lin = train_probe_v2(z_sm[sm_tr], sm_labels[sm_tr], z_sm[sm_va], sm_labels[sm_va], z_sm[sm_te], sm_labels[sm_te], 3, device, "linear")
        res_sm3_mlp = train_probe_v2(z_sm[sm_tr], sm_labels[sm_tr], z_sm[sm_va], sm_labels[sm_va], z_sm[sm_te], sm_labels[sm_te], 3, device, "mlp")

        z_c5 = z[c5_indices]
        res_c5_lin = train_probe_v2(z_c5[c5_tr], c5_labels[c5_tr], z_c5[c5_va], c5_labels[c5_va], z_c5[c5_te], c5_labels[c5_te], 5, device, "linear")

        row = {
            "model": mname,
            "latent_dim": D,
            "erank_sm": prov.get("erank_sm", np.nan),
            "sm3_linear_acc": res_sm3_lin["accuracy"],
            "sm3_linear_acc_se": res_sm3_lin["accuracy_se"],
            "sm3_linear_auc": res_sm3_lin["macro_auc"],
            "sm3_mlp_acc": res_sm3_mlp["accuracy"],
            "sm3_mlp_auc": res_sm3_mlp["macro_auc"],
            "c5_linear_acc": res_c5_lin["accuracy"],
            "c5_linear_auc": res_c5_lin["macro_auc"],
            "ood_hp_v2_auc": bin_results["sm_vs_higgs_portal_v2"]["macro_auc"],
            "ood_hp_v2_auc_se": bin_results["sm_vs_higgs_portal_v2"]["macro_auc_se"],
            "ood_hv_auc": bin_results["sm_vs_hidden_valley"]["macro_auc"],
            "ood_hv_auc_se": bin_results["sm_vs_hidden_valley"]["macro_auc_se"],
            "ood_hp_v1_auc": bin_results["sm_vs_higgs_portal_v1"]["macro_auc"],
            "ood_hp_v1_auc_se": bin_results["sm_vs_higgs_portal_v1"]["macro_auc_se"],
        }
        with open(probe_json, "w") as f:
            json.dump({"summary": row, "sm3_linear": res_sm3_lin, "sm3_mlp": res_sm3_mlp, "c5_linear": res_c5_lin, "binary_ood_linear": bin_results}, f, indent=2)

        csv_p = P3_PROBE_DIR / "p3_probe_summary.csv"
        with gpfs_lock(csv_p):
            df_new = pd.DataFrame([row])
            if csv_p.exists():
                old = pd.read_csv(csv_p)
                old = old[old["model"] != mname]
                df_new = pd.concat([old, df_new], ignore_index=True)
            df_new.to_csv(csv_p, index=False)
        print(f"[p3_probe] {mname:26s} | SM3={row['sm3_linear_acc']:.4f} (MLP {row['sm3_mlp_acc']:.4f}) | HP_v2={row['ood_hp_v2_auc']:.4f} | HV={row['ood_hv_auc']:.4f} ({time.time()-t0:.1f}s)", flush=True)

    # 2. Paired Shape-Only NPLM (exact seeds matching phase2_nplm_calibrated.py)
    nplm_json = P3_NPLM_DIR / f"{mname}_shape.json"
    if not nplm_json.exists():
        t0 = time.time()
        n_r, n_b, m_centers = 500_000, 100_000, 316
        n_null, n_sig = 250, 100
        fracs = [0.005, 0.01, 0.025, 0.05]

        basis = build_custom_basis(z, labels, n_r=n_r, m_centers=m_centers, ref_seed=42, device=device, sigma_quantile=0.90)
        t_null = run_null_toys_gpu(basis, n_r, n_b, n_null, toy_seed=1001)
        gof = evaluate_null_calibration(t_null)
        thr_01 = order_stat_threshold(t_null, 0.01)

        out_j = {
            "model": mname,
            "latent_dim": D,
            "sigma": basis["sigma"],
            "null_gof": gof,
            "signals": {},
        }
        rows_n = []
        for s_lbl in [3, 4, 5]:
            sname = PROCESS_NAMES[s_lbl]
            out_j["signals"][sname] = {}
            for frac in fracs:
                t_sig = run_signal_toys_gpu(basis, s_lbl, n_r, n_b, frac, n_sig, toy_seed=2000 + 100 * s_lbl + int(round(frac * 10000)), sample_mode="shape")
                t_med = float(np.median(t_sig))
                zd = compute_z_scores(t_med, t_null, gof)
                k_rej = int((t_sig > thr_01).sum())
                pow_01 = k_rej / len(t_sig)
                ci_lo, ci_hi = clopper_pearson_ci(k_rej, len(t_sig))
                rec = {
                    "model": mname,
                    "latent_dim": D,
                    "signal": sname,
                    "frac": frac,
                    "null_mean": gof["mean"],
                    "null_std": gof["std"],
                    "t_median": t_med,
                    "snr": zd["snr"],
                    "z_asymptotic": zd["z_asymptotic"],
                    "power_alpha_1e2": pow_01,
                    "power_alpha_1e2_ci_lo": ci_lo,
                    "power_alpha_1e2_ci_hi": ci_hi,
                }
                out_j["signals"][sname][str(frac)] = rec
                rows_n.append(rec)
        with open(nplm_json, "w") as f:
            json.dump(out_j, f, indent=2)
        csv_n = P3_NPLM_DIR / "p3_nplm_summary_shape.csv"
        with gpfs_lock(csv_n):
            df_n = pd.DataFrame(rows_n)
            if csv_n.exists():
                old = pd.read_csv(csv_n)
                old = old[old["model"] != mname]
                df_n = pd.concat([old, df_n], ignore_index=True)
            df_n.to_csv(csv_n, index=False)
        snr_hp = out_j["signals"]["higgs_portal_v2"]["0.01"]["snr"]
        snr_hv = out_j["signals"]["hidden_valley"]["0.01"]["snr"]
        print(f"[p3_nplm]  {mname:26s} | 1% HP_v2 SNR={snr_hp:5.2f} | 1% HV SNR={snr_hv:5.2f} ({time.time()-t0:.1f}s)", flush=True)


def build_custom_basis(
    z_all: np.ndarray,
    labels_all: np.ndarray,
    n_r: int,
    m_centers: int,
    ref_seed: int,
    device: torch.device,
    sigma_quantile: float = 0.90,
    custom_scaler=None,
    already_transformed: bool = False,
) -> dict:
    sm_mask = np.isin(labels_all, [0, 1, 2])
    z_bkg = z_all[sm_mask]

    rng_ref = np.random.default_rng(ref_seed)
    perm_bkg = rng_ref.permutation(len(z_bkg))
    idx_ref = perm_bkg[:n_r]
    idx_pool = perm_bkg[n_r:]

    if already_transformed:
        ref_scaled = z_bkg[idx_ref].astype(np.float32)
        bkg_pool_scaled = z_bkg[idx_pool].astype(np.float32)
        transform_fn = lambda x: x.astype(np.float32)
    else:
        scaler = custom_scaler if custom_scaler is not None else StandardScaler().fit(z_bkg[idx_ref])
        if hasattr(scaler, "mean_") and scaler.mean_ is None:
            scaler.fit(z_bkg[idx_ref])
        ref_scaled = scaler.transform(z_bkg[idx_ref]).astype(np.float32)
        bkg_pool_scaled = scaler.transform(z_bkg[idx_pool]).astype(np.float32)
        transform_fn = lambda x: scaler.transform(x).astype(np.float32)

    centers, sigma = choose_centers_and_sigma(ref_scaled, m_centers, sigma_quantile, rng_ref, device)

    dtype = torch.float64
    centers_t = torch.from_numpy(centers).to(device=device, dtype=dtype)
    ref_t = torch.from_numpy(ref_scaled).to(device=device, dtype=dtype)
    bkg_pool_t = torch.from_numpy(bkg_pool_scaled).to(device=device, dtype=dtype)

    k_mm = rbf_features_gpu(centers_t, centers_t, sigma) + 1e-6 * torch.eye(m_centers, device=device, dtype=dtype)
    phi_ref = rbf_features_gpu(ref_t, centers_t, sigma, chunk=131072)
    del ref_t

    sig_pools_t = {}
    for s_lbl in [3, 4, 5]:
        s_mask = labels_all == s_lbl
        if s_mask.any():
            sig_pools_t[s_lbl] = torch.from_numpy(transform_fn(z_all[s_mask])).to(device=device, dtype=dtype)

    return {
        "sigma": sigma,
        "centers_t": centers_t,
        "k_mm": k_mm,
        "phi_ref": phi_ref,
        "bkg_pool_t": bkg_pool_t,
        "n_pool": len(bkg_pool_scaled),
        "sig_pools_t": sig_pools_t,
    }


def step_supcon_subspace(device: torch.device):
    """
    Quantitatively diagnoses the supcon_d64 supervised-vs-NPLM paradox on HP_v2:
      1. Fits StandardScaler + full 64-D PCA strictly on the 500k SM reference sample R.
      2. Trains binary linear probes for SM vs HP_v2 and SM vs HV on standardized embeddings,
         decomposing probe weight norm ||w||^2 across SM PCA subspaces (PCs 1..2, 1..3, 4..16, 4..64).
      3. Evaluates BOTH binary linear probe AUC and paired shape-only Kernel NPLM (1% HP_v2 & HV)
         across subspaces (unwhitened PCs vs unit-variance whitened PCs, plus top-8 and top-16 whitened PCs).
    """
    out_dir = P3_DIR / "supcon_subspace"
    out_dir.mkdir(parents=True, exist_ok=True)
    done_json = out_dir / "supcon_subspace_results.json"
    if done_json.exists():
        print(f"[supcon_subspace] Already complete at {done_json}", flush=True)
        return

    data = np.load(EMBED_V2_DIR / "supcon_d64.npz")
    z = data["z"].astype(np.float32)
    labels = data["labels"]

    sm_mask = np.isin(labels, [0, 1, 2])
    z_bkg = z[sm_mask]
    rng_ref = np.random.default_rng(42)
    perm_bkg = rng_ref.permutation(len(z_bkg))
    idx_ref = perm_bkg[:500_000]

    std_scaler = StandardScaler().fit(z_bkg[idx_ref])
    z_std = std_scaler.transform(z).astype(np.float32)
    pca = PCA(n_components=64, random_state=42).fit(z_std[sm_mask][idx_ref])

    z_pca_raw = pca.transform(z_std).astype(np.float32)
    pc_scales = np.sqrt(np.maximum(pca.explained_variance_, 1e-12)).astype(np.float32)
    z_pca_whitened = (z_pca_raw / pc_scales.reshape(1, 64)).astype(np.float32)

    evr = pca.explained_variance_ratio_
    subspace_slices = {
        "full_d64_std": ("orig", None),
        "pcs_1_2_raw": ("raw", slice(0, 2)),
        "pcs_1_2_whitened": ("white", slice(0, 2)),
        "pcs_1_3_raw": ("raw", slice(0, 3)),
        "pcs_1_3_whitened": ("white", slice(0, 3)),
        "pcs_1_8_whitened": ("white", slice(0, 8)),
        "pcs_1_16_whitened": ("white", slice(0, 16)),
        "pcs_4_16_raw": ("raw", slice(3, 16)),
        "pcs_4_16_whitened": ("white", slice(3, 16)),
        "pcs_4_64_raw": ("raw", slice(3, 64)),
        "pcs_4_64_whitened": ("white", slice(3, 64)),
    }

    # Decompose linear probe weight vectors in standardized space into PCA eigenbasis
    sm_indices = np.where(sm_mask)[0]
    rng_bin = np.random.default_rng(42)
    sm_sub = rng_bin.choice(sm_indices, size=60_000, replace=False)

    probe_decomp = {}
    for bsm_lbl, bsm_name in [(3, "higgs_portal_v2"), (4, "hidden_valley")]:
        bsm_idx = np.where(labels == bsm_lbl)[0]
        idx_all = np.concatenate([sm_sub, bsm_idx])
        y_bin = np.concatenate([np.zeros(len(sm_sub), dtype=np.int64), np.ones(len(bsm_idx), dtype=np.int64)])
        b_tr, b_va, b_te = make_stratified_split(y_bin, 0.70, 0.10, seed=42 + bsm_lbl)

        # Fit logistic linear probe directly on z_std (without re-scaling per coordinate) to get exact direction w in z_std
        torch.manual_seed(42)
        lin = nn.Linear(64, 2).to(device)
        counts = np.bincount(y_bin[b_tr], minlength=2).astype(np.float32)
        w_cls = counts.sum() / (2.0 * np.maximum(counts, 1.0))
        crit = nn.CrossEntropyLoss(weight=torch.from_numpy(w_cls).to(device))
        opt = torch.optim.AdamW(lin.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40)
        x_tr_t = torch.from_numpy(z_std[idx_all][b_tr]).to(device)
        y_tr_t = torch.from_numpy(y_bin[b_tr]).to(device)
        x_va_t = torch.from_numpy(z_std[idx_all][b_va]).to(device)
        y_va_t = torch.from_numpy(y_bin[b_va]).to(device)
        x_te_t = torch.from_numpy(z_std[idx_all][b_te]).to(device)
        best_vl = float("inf")
        best_st = None
        for _ in range(40):
            lin.train()
            perm = torch.randperm(len(x_tr_t), device=device)
            for st in range(0, len(x_tr_t), 1024):
                bi = perm[st : st + 1024]
                opt.zero_grad(set_to_none=True)
                loss = crit(lin(x_tr_t[bi]), y_tr_t[bi])
                loss.backward()
                opt.step()
            sched.step()
            lin.eval()
            with torch.no_grad():
                vl = float(crit(lin(x_va_t), y_va_t).item())
            if vl < best_vl:
                best_vl = vl
                best_st = {k: v.detach().clone() for k, v in lin.state_dict().items()}
        lin.load_state_dict(best_st)
        w_vec = (lin.weight[1] - lin.weight[0]).detach().cpu().numpy().astype(np.float64)
        c_pca = pca.components_.astype(np.float64) @ w_vec
        tot_w2 = float(np.sum(c_pca ** 2))

        # Also measure Mahalanobis Fisher SNR^2 = sum_k (Delta mu_k)^2 / lambda_k across PC subspaces
        mu_sm = z_pca_raw[sm_mask][idx_ref].mean(axis=0).astype(np.float64)
        mu_sig = z_pca_raw[bsm_idx].mean(axis=0).astype(np.float64)
        dmu = mu_sig - mu_sm
        lam = np.maximum(pca.explained_variance_.astype(np.float64), 1e-12)
        euclid_dmu2 = dmu ** 2
        mahal_dmu2 = (dmu ** 2) / lam

        probe_decomp[bsm_name] = {
            "w2_frac_pcs_1_2": float(np.sum(c_pca[0:2] ** 2) / tot_w2),
            "w2_frac_pcs_1_3": float(np.sum(c_pca[0:3] ** 2) / tot_w2),
            "w2_frac_pcs_4_16": float(np.sum(c_pca[3:16] ** 2) / tot_w2),
            "w2_frac_pcs_4_64": float(np.sum(c_pca[3:64] ** 2) / tot_w2),
            "euclid_shift2_pcs_1_3": float(np.sum(euclid_dmu2[0:3])),
            "euclid_shift2_pcs_4_16": float(np.sum(euclid_dmu2[3:16])),
            "euclid_shift2_pcs_4_64": float(np.sum(euclid_dmu2[3:64])),
            "mahal_snr2_pcs_1_3": float(np.sum(mahal_dmu2[0:3])),
            "mahal_snr2_pcs_4_16": float(np.sum(mahal_dmu2[3:16])),
            "mahal_snr2_pcs_17_64": float(np.sum(mahal_dmu2[16:64])),
        }

    # Evaluate subspace binary probes + paired shape-only NPLM
    n_r, n_b, m_centers = 500_000, 100_000, 316
    n_null, n_sig = 250, 100
    rows = []
    for sub_name, (kind, sl) in subspace_slices.items():
        t0 = time.time()
        if kind == "orig":
            z_sub = z_std
        elif kind == "raw":
            z_sub = z_pca_raw[:, sl]
        else:
            z_sub = z_pca_whitened[:, sl]

        # Binary probe AUCs on this subspace
        aucs = {}
        for bsm_lbl, bsm_name in [(3, "higgs_portal_v2"), (4, "hidden_valley")]:
            bsm_idx = np.where(labels == bsm_lbl)[0]
            idx_all = np.concatenate([sm_sub, bsm_idx])
            y_bin = np.concatenate([np.zeros(len(sm_sub), dtype=np.int64), np.ones(len(bsm_idx), dtype=np.int64)])
            b_tr, b_va, b_te = make_stratified_split(y_bin, 0.70, 0.10, seed=42 + bsm_lbl)
            zb = z_sub[idx_all]
            r_p = train_probe_v2(zb[b_tr], y_bin[b_tr], zb[b_va], y_bin[b_va], zb[b_te], y_bin[b_te], 2, device, "linear", batch_size=1024, compute_boot=False)
            aucs[bsm_name] = r_p["macro_auc"]

        # Paired shape-only NPLM (without re-scaling raw PCs so raw vs whitened difference is preserved!)
        basis = build_custom_basis(z_sub, labels, n_r=n_r, m_centers=m_centers, ref_seed=42, device=device, sigma_quantile=0.90, already_transformed=True)
        t_null = run_null_toys_gpu(basis, n_r, n_b, n_null, toy_seed=1001)
        gof = evaluate_null_calibration(t_null)
        thr_01 = order_stat_threshold(t_null, 0.01)

        rec = {
            "subspace": sub_name,
            "dim": int(z_sub.shape[1]),
            "sigma": basis["sigma"],
            "null_mean": gof["mean"],
            "null_std": gof["std"],
            "probe_auc_hp_v2": aucs["higgs_portal_v2"],
            "probe_auc_hv": aucs["hidden_valley"],
        }
        for s_lbl, s_short in [(3, "hp_v2"), (4, "hv")]:
            t_sig = run_signal_toys_gpu(basis, s_lbl, n_r, n_b, 0.01, n_sig, toy_seed=2000 + 100 * s_lbl + 100, sample_mode="shape")
            t_med = float(np.median(t_sig))
            zd = compute_z_scores(t_med, t_null, gof)
            pow_01 = float((t_sig > thr_01).mean())
            rec[f"snr_1pct_{s_short}"] = zd["snr"]
            rec[f"power_1pct_{s_short}"] = pow_01
        rows.append(rec)
        print(
            f"[supcon_subspace] {sub_name:20s} (D={rec['dim']:2d}) | Probe HPv2={rec['probe_auc_hp_v2']:.4f} | "
            f"NPLM 1% HPv2 SNR={rec['snr_1pct_hp_v2']:5.2f} (Pow={rec['power_1pct_hp_v2']:.2f}) | "
            f"1% HV SNR={rec['snr_1pct_hv']:5.2f} ({time.time()-t0:.1f}s)",
            flush=True,
        )

    summary = {
        "sm_pca_explained_variance": {
            "pc1": float(evr[0]),
            "pc2": float(evr[1]),
            "pc3": float(evr[2]),
            "pcs_1_2_sum": float(evr[0:2].sum()),
            "pcs_1_3_sum": float(evr[0:3].sum()),
            "pcs_4_16_sum": float(evr[3:16].sum()),
            "pcs_4_64_sum": float(evr[3:64].sum()),
        },
        "linear_probe_pca_decomposition": probe_decomp,
        "subspace_nplm_and_probes": rows,
    }
    with open(done_json, "w") as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame(rows).to_csv(out_dir / "supcon_subspace_table.csv", index=False)
    print(f"[supcon_subspace] Saved {done_json}", flush=True)


def step_bandwidth_zca_sweep(device: torch.device):
    """
    Sweeps:
      1. ZCA regularization gamma in {1e-1, 1e-2, 1e-4, 1e-6} on supcon_d64, ssl_kinematic_d64, raw_summary_d28.
      2. RBF bandwidth quantile q in {0.10, 0.25, 0.50, 0.90} (M=316) and Nystrom centers M in {316, 1000} (q=0.90)
         across supcon_d64, ssl_kinematic_d64, ssl_tokens_d64, ssl_tokens_d8, ae_transformer_d32, vqvae_quant_d8, phys_handcrafted_d6.
    """
    out_dir = P3_DIR / "supcon_subspace"
    out_dir.mkdir(parents=True, exist_ok=True)
    done_json = out_dir / "bandwidth_zca_sweep.json"
    n_r, n_b = 500_000, 100_000
    n_null, n_sig = 250, 100
    configs = [
        (316, 0.10),
        (316, 0.25),
        (316, 0.50),
        (316, 0.90),
        (1000, 0.90),
    ]
    if done_json.exists():
        with open(done_json) as f:
            prev = json.load(f)
        bw_prev = prev.get("bandwidth_and_centers_sweep", [])
        if any(r.get("model") == "raw_summary_d28" for r in bw_prev):
            print(f"[bandwidth_zca_sweep] Already complete at {done_json}", flush=True)
            return
        d = np.load(EMBED_V2_DIR / "raw_summary_d28.npz")
        z, labels = d["z"].astype(np.float32), d["labels"]
        for m_c, q_bw in configs:
            t0 = time.time()
            basis = build_custom_basis(z, labels, n_r=n_r, m_centers=m_c, ref_seed=42, device=device, sigma_quantile=q_bw)
            t_null = run_null_toys_gpu(basis, n_r, n_b, n_null, toy_seed=1001)
            gof = evaluate_null_calibration(t_null)
            thr_01 = order_stat_threshold(t_null, 0.01)
            rec = {"model": "raw_summary_d28", "m_centers": m_c, "sigma_quantile": q_bw, "sigma": basis["sigma"], "null_mean": gof["mean"], "null_std": gof["std"]}
            for s_lbl, s_short in [(3, "hp_v2"), (4, "hv")]:
                t_sig = run_signal_toys_gpu(basis, s_lbl, n_r, n_b, 0.01, n_sig, toy_seed=2000 + 100 * s_lbl + 100, sample_mode="shape")
                zd = compute_z_scores(float(np.median(t_sig)), t_null, gof)
                rec[f"snr_1pct_{s_short}"] = zd["snr"]
                rec[f"power_1pct_{s_short}"] = float((t_sig > thr_01).mean())
            bw_prev.append(rec)
            print(f"[bw_sweep]  {'raw_summary_d28':20s} | M={m_c:4d} q={q_bw:.2f} | HPv2 SNR={rec['snr_1pct_hp_v2']:5.2f} | HV SNR={rec['snr_1pct_hv']:5.2f} ({time.time()-t0:.1f}s)", flush=True)
        prev["bandwidth_and_centers_sweep"] = bw_prev
        with open(done_json, "w") as f:
            json.dump(prev, f, indent=2)
        pd.DataFrame(bw_prev).to_csv(out_dir / "bandwidth_centers_sweep.csv", index=False)
        print(f"[bandwidth_zca_sweep] Appended raw_summary_d28 to {done_json}", flush=True)
        return

    # 1. ZCA gamma sweep
    zca_rows = []
    for mname in ["supcon_d64", "ssl_kinematic_d64", "raw_summary_d28"]:
        d = np.load(EMBED_V2_DIR / f"{mname}.npz")
        z, labels = d["z"].astype(np.float32), d["labels"]
        for mode_label, gamma in [("standard", None), ("zca_1e-1", 1e-1), ("zca_1e-2", 1e-2), ("zca_1e-4", 1e-4), ("zca_1e-6", 1e-6)]:
            t0 = time.time()
            sc = None if gamma is None else ZCAShrinkageWhitener(gamma=gamma)
            basis = build_custom_basis(z, labels, n_r=n_r, m_centers=316, ref_seed=42, device=device, sigma_quantile=0.90, custom_scaler=sc)
            t_null = run_null_toys_gpu(basis, n_r, n_b, n_null, toy_seed=1001)
            gof = evaluate_null_calibration(t_null)
            thr_01 = order_stat_threshold(t_null, 0.01)
            rec = {"model": mname, "mode": mode_label, "gamma": gamma, "null_mean": gof["mean"], "null_std": gof["std"]}
            for s_lbl, s_short in [(3, "hp_v2"), (4, "hv")]:
                t_sig = run_signal_toys_gpu(basis, s_lbl, n_r, n_b, 0.01, n_sig, toy_seed=2000 + 100 * s_lbl + 100, sample_mode="shape")
                zd = compute_z_scores(float(np.median(t_sig)), t_null, gof)
                rec[f"snr_1pct_{s_short}"] = zd["snr"]
                rec[f"power_1pct_{s_short}"] = float((t_sig > thr_01).mean())
            zca_rows.append(rec)
            print(f"[zca_sweep] {mname:20s} | {mode_label:10s} | HPv2 SNR={rec['snr_1pct_hp_v2']:5.2f} | HV SNR={rec['snr_1pct_hv']:5.2f} ({time.time()-t0:.1f}s)", flush=True)

    # 2. Bandwidth quantile q & Nystrom M sweep
    bw_models = [
        "supcon_d64", "ssl_kinematic_d64", "ssl_tokens_d64", "ssl_tokens_d8",
        "ae_transformer_d32", "vqvae_quant_d8", "phys_handcrafted_d6",
    ]
    configs = [
        (316, 0.10),
        (316, 0.25),
        (316, 0.50),
        (316, 0.90),
        (1000, 0.90),
    ]
    bw_rows = []
    for mname in bw_models:
        d = np.load(EMBED_V2_DIR / f"{mname}.npz")
        z, labels = d["z"].astype(np.float32), d["labels"]
        for m_c, q_bw in configs:
            t0 = time.time()
            basis = build_custom_basis(z, labels, n_r=n_r, m_centers=m_c, ref_seed=42, device=device, sigma_quantile=q_bw)
            t_null = run_null_toys_gpu(basis, n_r, n_b, n_null, toy_seed=1001)
            gof = evaluate_null_calibration(t_null)
            thr_01 = order_stat_threshold(t_null, 0.01)
            rec = {"model": mname, "m_centers": m_c, "sigma_quantile": q_bw, "sigma": basis["sigma"], "null_mean": gof["mean"], "null_std": gof["std"]}
            for s_lbl, s_short in [(3, "hp_v2"), (4, "hv")]:
                t_sig = run_signal_toys_gpu(basis, s_lbl, n_r, n_b, 0.01, n_sig, toy_seed=2000 + 100 * s_lbl + 100, sample_mode="shape")
                zd = compute_z_scores(float(np.median(t_sig)), t_null, gof)
                rec[f"snr_1pct_{s_short}"] = zd["snr"]
                rec[f"power_1pct_{s_short}"] = float((t_sig > thr_01).mean())
            bw_rows.append(rec)
            print(f"[bw_sweep]  {mname:20s} | M={m_c:4d} q={q_bw:.2f} | HPv2 SNR={rec['snr_1pct_hp_v2']:5.2f} | HV SNR={rec['snr_1pct_hv']:5.2f} ({time.time()-t0:.1f}s)", flush=True)

    with open(done_json, "w") as f:
        json.dump({"zca_gamma_sweep": zca_rows, "bandwidth_and_centers_sweep": bw_rows}, f, indent=2)
    pd.DataFrame(zca_rows).to_csv(out_dir / "zca_gamma_sweep.csv", index=False)
    pd.DataFrame(bw_rows).to_csv(out_dir / "bandwidth_centers_sweep.csv", index=False)
    print(f"[bandwidth_zca_sweep] Saved {done_json}", flush=True)


def run_resumable_continuous_ssl(
    tag: str,
    aug_mode: str,
    seed: int,
    epochs: int,
    device: torch.device,
    batch_size: int = 512,
):
    """
    Resumable continuous SetTransformer SSL pretraining (kinematic vs legacy augmentation)
    with concatenated (2B, T, 7) forward pass + TF32 enabled, followed by 1.05M extraction,
    paired probes, and paired shape-only NPLM.
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    save_dir = P3_CKPT_DIR / tag
    save_dir.mkdir(parents=True, exist_ok=True)
    npz_out = P3_EMBED_DIR / f"{tag}.npz"

    if not npz_out.exists():
        set_seed(seed)
        pre_data = load_events_fast(PRETRAIN_PARQUET, include_labels=SM_LABELS)
        scaler_path = save_dir / "scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
        else:
            feat7_pre = raw6_to_feat7_np(pre_data["raw6"], pre_data["mask"])
            scaler = fit_track_scaler(feat7_pre, pre_data["mask"])
            del feat7_pre
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)

        mean_t = torch.from_numpy(scaler.mean_.astype(np.float32)).to(device)
        scale_t = torch.from_numpy(scaler.scale_.astype(np.float32)).to(device)

        encoder = SetTransformerEncoder(latent_dim=64, norm_first=True, activation="gelu", use_out_norm=True).to(device)
        proj_head = ProjectionHead(in_dim=64, hidden_dim=64, out_dim=64).to(device)
        params = list(encoder.parameters()) + list(proj_head.parameters())
        optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-6)

        N = len(pre_data["labels"])
        steps_per_epoch = N // batch_size
        total_steps = epochs * steps_per_epoch
        warmup_steps = 500
        lr_lambda = lambda step: min(1.0, (step + 1) / warmup_steps) * 0.5 * (1.0 + math.cos(math.pi * max(0.0, step - warmup_steps) / max(1, total_steps - warmup_steps)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        raw6_gpu = torch.from_numpy(pre_data["raw6"][:, :128, :]).to(device)
        mask_gpu = torch.from_numpy(pre_data["mask"][:, :128]).to(device)
        lengths_gpu = torch.minimum(torch.from_numpy(pre_data["lengths"]).to(device), torch.tensor(128, device=device))
        del pre_data

        history = []
        start_epoch = 1
        resume_ckpt = save_dir / "resume.pt"
        if resume_ckpt.exists():
            rc = torch.load(resume_ckpt, map_location=device)
            encoder.load_state_dict(rc["encoder"])
            proj_head.load_state_dict(rc["proj_head"])
            optimizer.load_state_dict(rc["optimizer"])
            scheduler.load_state_dict(rc["scheduler"])
            history = rc["history"]
            start_epoch = int(rc["epoch"]) + 1
            print(f"[{tag}] Resuming from epoch {start_epoch}/{epochs}...", flush=True)

        aug_fn = augment_batch_kinematic_torch if aug_mode == "kinematic" else augment_batch_legacy_torch
        print(f"[{tag}] Training ({aug_mode} aug, seed={seed}) epochs {start_epoch}..{epochs} on {device}...", flush=True)
        for epoch in range(start_epoch, epochs + 1):
            t0 = time.time()
            set_seed(seed + 1000 * epoch)
            encoder.train()
            proj_head.train()
            perm = torch.randperm(N, device=device)
            epoch_loss = 0.0
            embed_std = 0.0

            for step in range(steps_per_epoch):
                idx = perm[step * batch_size : (step + 1) * batch_size]
                max_t = max(int(lengths_gpu[idx].max().item()), 1)
                r6_b = raw6_gpu[idx, :max_t, :]
                m_b = mask_gpu[idx, :max_t]

                x1, m1 = aug_fn(r6_b, m_b, mean_t, scale_t)
                x2, m2 = aug_fn(r6_b, m_b, mean_t, scale_t)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    z_2b = encoder(torch.cat([x1, x2], dim=0), torch.cat([m1, m2], dim=0))
                    p_2b = proj_head(z_2b)
                    B_cur = x1.shape[0]
                    z1 = z_2b[:B_cur]
                    loss = simclr_nt_xent_loss(p_2b[:B_cur], p_2b[B_cur:], temperature=0.5)

                if step == 0:
                    embed_std = float(z1.detach().float().std(dim=0).mean().item())

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                scheduler.step()
                epoch_loss += float(loss.item())

            avg_loss = epoch_loss / steps_per_epoch
            dt = time.time() - t0
            history.append({"epoch": epoch, "loss": avg_loss, "embed_std": embed_std, "sec": dt})
            print(f"[{tag}] Epoch {epoch:02d}/{epochs}: loss={avg_loss:.4f}, embed_std={embed_std:.4f} ({dt:.1f}s)", flush=True)
            tmp_r = resume_ckpt.with_suffix(".tmp.pt")
            torch.save({
                "epoch": epoch,
                "encoder": encoder.state_dict(),
                "proj_head": proj_head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "history": history,
            }, tmp_r)
            os.replace(tmp_r, resume_ckpt)
            torch.save(encoder.state_dict(), save_dir / "encoder.pt")

        del raw6_gpu, mask_gpu, lengths_gpu
        torch.cuda.empty_cache()
        torch.save(encoder.state_dict(), save_dir / "encoder.pt")
        torch.save(proj_head.state_dict(), save_dir / "proj_head.pt")
        pd.DataFrame(history).to_csv(save_dir / "training_history.csv", index=False)

        eval_data = load_events_fast(CANON_EVAL_PARQUET)
        feat7_eval = raw6_to_feat7_np(eval_data["raw6"][:, :128, :], eval_data["mask"][:, :128])
        z_eval = extract_set_transformer(encoder, feat7_eval, eval_data["mask"][:, :128], scaler, device, max_t_cap=128)
        save_p3_embedding(tag, z_eval, eval_data, {"family": tag, "aug_mode": aug_mode, "seed": seed, "max_t": 128, "epochs": epochs})

    evaluate_p3_probes_and_nplm(npz_out, device)


def run_resumable_token_ssl_seed(seed_idx: int, epochs: int, device: torch.device, batch_size: int = 512):
    """Resumable VQ-tokenized SSL pretraining for extra pretraining seeds (seed_idx in {1, 2})."""
    from vqvae_tracks import TrackVQVAE
    from pretrain_ssl_contrastive_tokens import TokenSetTransformerEncoder

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    tag = f"ssl_tokens_d64_s{seed_idx}"
    seed = 2042 + 100 * seed_idx
    save_dir = P3_CKPT_DIR / tag
    save_dir.mkdir(parents=True, exist_ok=True)
    npz_out = P3_EMBED_DIR / f"{tag}.npz"

    if not npz_out.exists():
        set_seed(seed)
        vq_dir = Path("/global/cfs/cdirs/m4958/usr/aritra08/token/vqvae_tracks/sm_only_v1")
        with open(vq_dir / "scaler.pkl", "rb") as f:
            vq_scaler = pickle.load(f)
        vq_ckpt = torch.load(vq_dir / "vqvae_best.pt", map_location=device)
        vqvae = TrackVQVAE(**vq_ckpt["cfg"]).to(device)
        vqvae.load_state_dict(vq_ckpt["state_dict"])
        vqvae.eval()
        for p in vqvae.parameters():
            p.requires_grad = False

        mean_t = torch.from_numpy(vq_scaler.mean_.astype(np.float32)).to(device).view(1, 1, 6)
        scale_t = torch.from_numpy(vq_scaler.scale_.astype(np.float32)).to(device).view(1, 1, 6)
        noise_vec = torch.tensor([0.05, 0.05, 0.03, 0.0, 0.0, 0.0], dtype=torch.float32, device=device).view(1, 1, 6)

        pre_data = load_events_fast(PRETRAIN_PARQUET, include_labels=SM_LABELS)
        vq6_pre, mask_pre = raw6_to_vqvae6_np(pre_data["raw6"], pre_data["mask"])
        N = len(pre_data["labels"])
        del pre_data

        encoder = TokenSetTransformerEncoder(codebook_size=8192, latent_dim=64).to(device)
        proj_head = ProjectionHead(in_dim=64, hidden_dim=64, out_dim=64).to(device)
        params = list(encoder.parameters()) + list(proj_head.parameters())
        optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-6)

        steps_per_epoch = N // batch_size
        total_steps = epochs * steps_per_epoch
        warmup_steps = 500
        lr_lambda = lambda step: min(1.0, (step + 1) / warmup_steps) * 0.5 * (1.0 + math.cos(math.pi * max(0.0, step - warmup_steps) / max(1, total_steps - warmup_steps)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        vq6_gpu = torch.from_numpy(vq6_pre).to(device)
        mask_gpu = torch.from_numpy(mask_pre).to(device)
        lengths_vq_gpu = mask_gpu.sum(dim=1)
        del vq6_pre, mask_pre

        history = []
        start_epoch = 1
        resume_ckpt = save_dir / "resume.pt"
        if resume_ckpt.exists():
            rc = torch.load(resume_ckpt, map_location=device)
            encoder.load_state_dict(rc["encoder"])
            proj_head.load_state_dict(rc["proj_head"])
            optimizer.load_state_dict(rc["optimizer"])
            scheduler.load_state_dict(rc["scheduler"])
            history = rc["history"]
            start_epoch = int(rc["epoch"]) + 1
            print(f"[{tag}] Resuming from epoch {start_epoch}/{epochs}...", flush=True)

        print(f"[{tag}] Training TokenSetTransformerEncoder (seed={seed}, epochs {start_epoch}..{epochs}) on {device}...", flush=True)
        for epoch in range(start_epoch, epochs + 1):
            t0 = time.time()
            set_seed(seed + 1000 * epoch)
            encoder.train()
            proj_head.train()
            perm = torch.randperm(N, device=device)
            epoch_loss = 0.0
            embed_std = 0.0

            for step in range(steps_per_epoch):
                idx = perm[step * batch_size : (step + 1) * batch_size]
                max_t = max(int(lengths_vq_gpu[idx].max().item()), 1)
                x_b = vq6_gpu[idx, :max_t, :]
                m_b = mask_gpu[idx, :max_t]

                B_cur, T_cur, _ = x_b.shape
                x_2b = torch.cat([x_b, x_b], dim=0)
                m_2b = torch.cat([m_b, m_b], dim=0)
                drop_prob = 0.10 + 0.10 * torch.rand((2 * B_cur, 1), device=device)
                keep_r = torch.rand((2 * B_cur, T_cur), device=device) >= drop_prob
                keep_r[:, 0] = True
                m_aug = m_2b & keep_r
                x_aug = x_2b + torch.randn_like(x_2b) * noise_vec
                x_scaled = ((x_aug - mean_t) / scale_t).masked_fill(~m_aug.unsqueeze(-1), 0.0)
                with torch.no_grad():
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                        z_e = vqvae.encode(x_scaled, m_aug)
                    _, tok_idx, _, _, _ = vqvae.vq(z_e.float(), m_aug)
                tok_2b = tok_idx.clamp(min=0)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    z_2b = encoder(tok_2b, m_aug)
                    p_2b = proj_head(z_2b)
                    z1 = z_2b[:B_cur]
                    loss = simclr_nt_xent_loss(p_2b[:B_cur], p_2b[B_cur:], temperature=0.5)

                if step == 0:
                    embed_std = float(z1.detach().float().std(dim=0).mean().item())

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                scheduler.step()
                epoch_loss += float(loss.item())

            avg_loss = epoch_loss / steps_per_epoch
            dt = time.time() - t0
            history.append({"epoch": epoch, "loss": avg_loss, "embed_std": embed_std, "sec": dt})
            print(f"[{tag}] Epoch {epoch:02d}/{epochs}: loss={avg_loss:.4f}, embed_std={embed_std:.4f} ({dt:.1f}s)", flush=True)
            tmp_r = resume_ckpt.with_suffix(".tmp.pt")
            torch.save({
                "epoch": epoch,
                "encoder": encoder.state_dict(),
                "proj_head": proj_head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "history": history,
            }, tmp_r)
            os.replace(tmp_r, resume_ckpt)
            torch.save(encoder.state_dict(), save_dir / "encoder.pt")

        del vq6_gpu, mask_gpu, lengths_vq_gpu
        torch.cuda.empty_cache()
        torch.save(encoder.state_dict(), save_dir / "encoder.pt")
        torch.save(proj_head.state_dict(), save_dir / "proj_head.pt")
        pd.DataFrame(history).to_csv(save_dir / "training_history.csv", index=False)

        eval_data = load_events_fast(CANON_EVAL_PARQUET)
        tc = np.load(CACHE_DIR / "eval_downstream_1050k_vq_tokens.npz")
        tokens_eval = tc["tokens"]
        mask_vq_eval = tc["mask"]
        encoder.eval()
        z_list = []
        with torch.no_grad():
            for st in range(0, len(tokens_eval), 2048):
                en = min(st + 2048, len(tokens_eval))
                tb = torch.from_numpy(tokens_eval[st:en].astype(np.int64)).clamp(min=0).to(device)
                mb = torch.from_numpy(mask_vq_eval[st:en]).to(device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    z_list.append(encoder(tb, mb).float().cpu().numpy())
        save_p3_embedding(tag, np.concatenate(z_list, axis=0), eval_data, {"family": "ssl_tokens", "seed": seed, "max_t": 128, "epochs": epochs})

    evaluate_p3_probes_and_nplm(npz_out, device)


def step_finetune_curves(family: str, device: torch.device):
    """
    End-to-end unfrozen fine-tuning across N_labeled in {100, 1000, 10000, 100000} x 5 seeds
    on the EXACT stratified splits and held-out sm_te (150,001 SM events) of supervised_scratch_d64.
    Uses LP-FT (5-epoch linear head warmup followed by full end-to-end differential-LR fine-tuning,
    lr_enc=2e-4, lr_head=1e-3, early stopping on exact budgeted validation loss).
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    out_dir = P3_DIR / "finetune_curves"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{family}_ft_d64"
    out_csv = out_dir / f"{tag}_curves.csv"

    rows = []
    done_pairs = set()
    if out_csv.exists():
        old_df = pd.read_csv(out_csv)
        rows = old_df.to_dict("records")
        done_pairs = set(zip(old_df["n_labeled"].astype(int), old_df["seed"].astype(int)))
        if len(done_pairs) >= 20:
            print(f"[{tag}] Already completed all 20 (n_labeled, seed) runs at {out_csv}", flush=True)
            return

    eval_data = load_events_fast(CANON_EVAL_PARQUET)
    labels = eval_data["labels"]
    sm_indices = np.where(np.isin(labels, [0, 1, 2]))[0]
    sm_labels = labels[sm_indices]
    sm_tr, sm_va, sm_te = make_stratified_split(sm_labels, 0.70, 0.10, seed=42)
    pool_idx = np.concatenate([sm_tr, sm_va])

    if family == "ssl_kinematic":
        ckpt_dir = CKPT_V2_DIR / "ssl_kinematic_d64"
        with open(ckpt_dir / "scaler.pkl", "rb") as f:
            sc = pickle.load(f)
        pre_state = torch.load(ckpt_dir / "encoder.pt", map_location="cpu")

        raw6_sm = eval_data["raw6"][sm_indices, :128, :]
        mask_sm = eval_data["mask"][sm_indices, :128]
        lengths_sm = mask_sm.sum(axis=1).astype(np.int32)
        del eval_data
        feat7_sm = raw6_to_feat7_np(raw6_sm, mask_sm)
        del raw6_sm
        mean = sc.mean_.astype(np.float32).reshape(1, 1, 7)
        scale = np.maximum(sc.scale_.astype(np.float32), 1e-6).reshape(1, 1, 7)
        x_all = ((feat7_sm - mean) / scale).astype(np.float32)
        x_all[~mask_sm] = 0.0
        del feat7_sm
        make_enc = lambda: SetTransformerEncoder(latent_dim=64, norm_first=True, activation="gelu", use_out_norm=True)
    elif family == "ssl_tokens":
        from pretrain_ssl_contrastive_tokens import TokenSetTransformerEncoder
        ckpt_dir = CKPT_V2_DIR / "ssl_tokens_d64"
        pre_state = torch.load(ckpt_dir / "encoder.pt", map_location="cpu")
        del eval_data
        tc = np.load(CACHE_DIR / "eval_downstream_1050k_vq_tokens.npz")
        x_all = np.maximum(tc["tokens"][sm_indices], 0).astype(np.int64)
        mask_sm = tc["mask"][sm_indices]
        lengths_sm = mask_sm.sum(axis=1).astype(np.int32)
        make_enc = lambda: TokenSetTransformerEncoder(codebook_size=8192, latent_dim=64)
    else:
        raise ValueError(f"Unknown family={family}")

    for n_labeled in [100, 1000, 10000, 100000]:
        if n_labeled == 100:
            epochs, bs = 60, 32
        elif n_labeled == 1000:
            epochs, bs = 50, 64
        elif n_labeled == 10000:
            epochs, bs = 35, 256
        else:
            epochs, bs = 25, 512

        for seed in range(5):
            if (n_labeled, seed) in done_pairs:
                continue
            t0 = time.time()
            torch.manual_seed(5000 + 100 * seed + n_labeled)
            np.random.seed(5000 + 100 * seed + n_labeled)

            sss_b = StratifiedShuffleSplit(n_splits=1, train_size=n_labeled, random_state=1000 * seed + n_labeled)
            sub_rel, _ = next(sss_b.split(np.zeros(len(pool_idx)), sm_labels[pool_idx]))
            sub_idx = pool_idx[sub_rel]

            sss_tv = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=2000 * seed + n_labeled)
            tr_sub, va_sub = next(sss_tv.split(np.zeros(len(sub_idx)), sm_labels[sub_idx]))
            idx_tr = sub_idx[tr_sub]
            idx_va = sub_idx[va_sub]

            x_tr_g = torch.from_numpy(x_all[idx_tr]).to(device)
            m_tr_g = torch.from_numpy(mask_sm[idx_tr]).to(device)
            l_tr_g = torch.from_numpy(lengths_sm[idx_tr]).to(device)
            y_tr_g = torch.from_numpy(sm_labels[idx_tr]).to(device)

            x_va_g = torch.from_numpy(x_all[idx_va]).to(device)
            m_va_g = torch.from_numpy(mask_sm[idx_va]).to(device)
            y_va_g = torch.from_numpy(sm_labels[idx_va]).to(device)

            enc = make_enc().to(device)
            enc.load_state_dict(pre_state)
            head = nn.Linear(64, 3).to(device)

            best_va_loss = float("inf")
            best_enc = {k: v.detach().clone() for k, v in enc.state_dict().items()}
            best_head = {k: v.detach().clone() for k, v in head.state_dict().items()}

            n_tr = len(idx_tr)
            n_batches = max(1, (n_tr + bs - 1) // bs)

            def eval_val() -> float:
                enc.eval()
                head.eval()
                va_losses = []
                with torch.no_grad():
                    for vs in range(0, len(idx_va), 1024):
                        ve = min(vs + 1024, len(idx_va))
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                            v_log = head(enc(x_va_g[vs:ve], m_va_g[vs:ve]))
                            va_losses.append(float(F.cross_entropy(v_log, y_va_g[vs:ve], reduction="sum").item()))
                return sum(va_losses) / len(idx_va)

            # Stage 1: 5-epoch linear head warmup (prevents random head gradient shock at step 1)
            for p in enc.parameters():
                p.requires_grad = False
            opt_lp = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
            for _ in range(5):
                enc.eval()
                head.train()
                perm = torch.randperm(n_tr, device=device)
                for b in range(n_batches):
                    bi = perm[b * bs : (b + 1) * bs]
                    mt = max(int(l_tr_g[bi].max().item()), 1)
                    opt_lp.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                        z_b = enc(x_tr_g[bi, :mt], m_tr_g[bi, :mt])
                        loss = F.cross_entropy(head(z_b), y_tr_g[bi])
                    loss.backward()
                    opt_lp.step()
                vl = eval_val()
                if vl < best_va_loss:
                    best_va_loss = vl
                    best_enc = {k: v.detach().clone() for k, v in enc.state_dict().items()}
                    best_head = {k: v.detach().clone() for k, v in head.state_dict().items()}

            # Stage 2: Full end-to-end unfrozen fine-tuning
            for p in enc.parameters():
                p.requires_grad = True
            params = [
                {"params": list(enc.parameters()), "lr": 2e-4},
                {"params": list(head.parameters()), "lr": 1e-3},
            ]
            all_params = list(enc.parameters()) + list(head.parameters())
            opt_ft = torch.optim.AdamW(params, weight_decay=1e-4)
            sched_ft = torch.optim.lr_scheduler.CosineAnnealingLR(opt_ft, T_max=epochs)

            for _ in range(epochs):
                enc.train()
                head.train()
                perm = torch.randperm(n_tr, device=device)
                for b in range(n_batches):
                    bi = perm[b * bs : (b + 1) * bs]
                    mt = max(int(l_tr_g[bi].max().item()), 1)
                    opt_ft.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                        logits = head(enc(x_tr_g[bi, :mt], m_tr_g[bi, :mt]))
                        loss = F.cross_entropy(logits, y_tr_g[bi])
                    loss.backward()
                    nn.utils.clip_grad_norm_(all_params, 1.0)
                    opt_ft.step()
                sched_ft.step()
                vl = eval_val()
                if vl < best_va_loss:
                    best_va_loss = vl
                    best_enc = {k: v.detach().clone() for k, v in enc.state_dict().items()}
                    best_head = {k: v.detach().clone() for k, v in head.state_dict().items()}

            # Evaluate best validation checkpoint on held-out sm_te (150,001 events)
            enc.load_state_dict(best_enc)
            head.load_state_dict(best_head)
            enc.eval()
            head.eval()
            probs_list = []
            with torch.no_grad():
                for ts in range(0, len(sm_te), 2048):
                    te_slice = sm_te[ts : ts + 2048]
                    mt = max(int(lengths_sm[te_slice].max()), 1)
                    xt = torch.from_numpy(x_all[te_slice, :mt]).to(device, non_blocking=True)
                    mt_g = torch.from_numpy(mask_sm[te_slice, :mt]).to(device, non_blocking=True)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                        lg = head(enc(xt, mt_g)).float()
                        probs_list.append(torch.softmax(lg, dim=-1).cpu().numpy())
            probs_te = np.concatenate(probs_list, axis=0)
            preds_te = probs_te.argmax(axis=-1)
            y_te = sm_labels[sm_te]
            y_bin_te = label_binarize(y_te, classes=[0, 1, 2])
            acc = float(accuracy_score(y_te, preds_te))
            auc = float(roc_auc_score(y_bin_te, probs_te, average="macro", multi_class="ovr"))

            rows.append({
                "model": tag,
                "probe_type": "finetune_e2e",
                "n_labeled": int(n_labeled),
                "seed": int(seed),
                "test_acc": acc,
                "test_auc": auc,
                "best_val_loss": float(best_va_loss),
            })
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            print(f"[{tag}] N={n_labeled:6d} seed={seed} | test_acc={acc:.4f} | test_auc={auc:.4f} ({time.time()-t0:.1f}s)", flush=True)


def step_summarize_all():
    """Consolidates all Phase 3 hardening results + 3-seed pretraining summaries into phase3_summary.json."""
    summary = {}

    # 1. Subspace & Bandwidth/ZCA
    sub_j = P3_DIR / "supcon_subspace" / "supcon_subspace_results.json"
    if sub_j.exists():
        with open(sub_j) as f:
            summary["supcon_subspace"] = json.load(f)
    bw_j = P3_DIR / "supcon_subspace" / "bandwidth_zca_sweep.json"
    if bw_j.exists():
        with open(bw_j) as f:
            summary["bandwidth_zca_sweep"] = json.load(f)

    # 2. Single-variable legacy aug T=128 vs legacy T=256 vs kinematic T=128
    p2_probes = pd.read_csv(PROBE_V2_DIR / "probe_summary.csv").set_index("model")
    p2_nplm = pd.read_csv(NPLM_V2_DIR / "nplm_summary_shape.csv")

    # Recover any completed Phase 3 probe/NPLM JSONs directly so no row is ever lost
    p3_p_rows = []
    for jf in sorted(P3_PROBE_DIR.glob("*_probe_detail.json")):
        with open(jf) as f:
            p3_p_rows.append(json.load(f)["summary"])
    p3_n_rows = []
    for jf in sorted(P3_NPLM_DIR.glob("*_shape.json")):
        with open(jf) as f:
            jd = json.load(f)
            for sdict in jd["signals"].values():
                for rec in sdict.values():
                    p3_n_rows.append(rec)

    if p3_p_rows and p3_n_rows:
        p3_p = pd.DataFrame(p3_p_rows).set_index("model")
        p3_n = pd.DataFrame(p3_n_rows)

        def has_both(mname: str) -> bool:
            return (mname in p3_p.index) and ((p3_n["model"] == mname).any())

        def get_metrics(mname: str, is_p3: bool) -> dict:
            pr = p3_p.loc[mname] if is_p3 else p2_probes.loc[mname]
            ndf = p3_n if is_p3 else p2_nplm
            sub_hp = ndf[(ndf["model"] == mname) & (ndf["signal"] == "higgs_portal_v2") & (np.isclose(ndf["frac"], 0.01))].iloc[0]
            sub_hv = ndf[(ndf["model"] == mname) & (ndf["signal"] == "hidden_valley") & (np.isclose(ndf["frac"], 0.01))].iloc[0]
            return {
                "erank_sm": float(pr["erank_sm"]),
                "sm3_linear_acc": float(pr["sm3_linear_acc"]),
                "sm3_mlp_acc": float(pr["sm3_mlp_acc"]),
                "ood_hp_v2_auc": float(pr["ood_hp_v2_auc"]),
                "ood_hv_auc": float(pr["ood_hv_auc"]),
                "nplm_1pct_hp_v2_snr": float(sub_hp["snr"]),
                "nplm_1pct_hp_v2_power": float(sub_hp["power_alpha_1e2"]),
                "nplm_1pct_hv_snr": float(sub_hv["snr"]),
                "nplm_1pct_hv_power": float(sub_hv["power_alpha_1e2"]),
            }

        if has_both("ssl_legacy_aug_t128_d64"):
            summary["legacy_aug_control"] = {
                "ssl_legacy_d64_t256": get_metrics("ssl_legacy_d64", False),
                "ssl_legacy_aug_t128_d64": get_metrics("ssl_legacy_aug_t128_d64", True),
                "ssl_kinematic_d64_t128": get_metrics("ssl_kinematic_d64", False),
            }

        # 3. 3-Seed Pretraining Uncertainty at D=64
        pretrain_3seed = {}
        for base_name in ["ssl_kinematic_d64", "ssl_tokens_d64"]:
            s1, s2 = f"{base_name}_s1", f"{base_name}_s2"
            if has_both(s1) and has_both(s2):
                recs = [get_metrics(base_name, False), get_metrics(s1, True), get_metrics(s2, True)]
                agg = {"per_seed": recs, "mean_sd": {}}
                for k in recs[0].keys():
                    vals = np.array([r[k] for r in recs], dtype=np.float64)
                    agg["mean_sd"][k] = {"mean": float(vals.mean()), "sd": float(vals.std(ddof=1))}
                pretrain_3seed[base_name] = agg
        summary["pretrain_3seed_d64"] = pretrain_3seed

    # 4. End-to-End Fine-Tuning vs Frozen Probes vs Scratch (deduplicating probe_type == 'e2e_scratch')
    ft_dir = P3_DIR / "finetune_curves"
    ft_summary = {}
    p2_curves = pd.read_csv(PROBE_V2_DIR / "label_efficiency_curves.csv")
    sc_curves = pd.read_csv(PROBE_V2_DIR / "supervised_scratch_curves.csv")
    if "probe_type" in sc_curves.columns:
        sc_curves = sc_curves[sc_curves["probe_type"] == "e2e_scratch"]
    for n_lab in [100, 1000, 10000, 100000]:
        row_n = {}
        sub_sc = sc_curves[sc_curves["n_labeled"] == n_lab].sort_values("seed")
        assert len(sub_sc) == 5, f"Expected 5 scratch seeds at N={n_lab}, got {len(sub_sc)}"
        row_n["supervised_scratch_d64"] = {
            "acc_mean": float(sub_sc["test_acc"].mean()),
            "acc_sd": float(sub_sc["test_acc"].std(ddof=1)),
            "auc_mean": float(sub_sc["test_auc"].mean()),
            "auc_sd": float(sub_sc["test_auc"].std(ddof=1)),
        }
        for m_frz in ["ssl_kinematic_d64", "ssl_tokens_d64", "phys_handcrafted_d6"]:
            for pt in ["linear", "mlp"]:
                sub = p2_curves[(p2_curves["model"] == m_frz) & (p2_curves["probe_type"] == pt) & (p2_curves["n_labeled"] == n_lab)]["test_acc"]
                if len(sub) > 0:
                    row_n[f"{m_frz}_{pt}"] = {"mean": float(sub.mean()), "sd": float(sub.std(ddof=1))}
        for fam in ["ssl_kinematic", "ssl_tokens"]:
            fc = ft_dir / f"{fam}_ft_d64_curves.csv"
            if fc.exists():
                df_fc = pd.read_csv(fc)
                sub_f = df_fc[df_fc["n_labeled"] == n_lab]["test_acc"]
                if len(sub_f) > 0:
                    row_n[f"{fam}_ft_d64"] = {"mean": float(sub_f.mean()), "sd": float(sub_f.std(ddof=1))}
        ft_summary[str(n_lab)] = row_n
    summary["label_efficiency_with_finetune"] = ft_summary

    out_path = P3_DIR / "phase3_summary.json"
    P3_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[summarize_all] Wrote consolidated summary to {out_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--step",
        type=str,
        required=True,
        choices=["supcon_subspace", "bandwidth_zca_sweep", "broken_aug_t128", "pretrain_seed", "finetune_curves", "summarize_all"],
    )
    parser.add_argument("--family", type=str, default="ssl_kinematic", choices=["ssl_kinematic", "ssl_tokens"])
    parser.add_argument("--seed-idx", type=int, default=1, choices=[1, 2])
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    if args.step == "summarize_all":
        step_summarize_all()
        return

    lock_key = f"{args.step}_{args.family}_{args.seed_idx}" if args.step in ("pretrain_seed", "finetune_curves") else args.step
    lock_dir = P3_DIR / "locks" / f"{lock_key}.lockdir"
    (P3_DIR / "locks").mkdir(parents=True, exist_ok=True)
    try:
        os.mkdir(lock_dir)
    except FileExistsError:
        print(f"[lock] {lock_key} is already running on another worker; skipping cleanly.", flush=True)
        return

    try:
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.set_device(device)

        if args.step == "supcon_subspace":
            step_supcon_subspace(device)
        elif args.step == "bandwidth_zca_sweep":
            step_bandwidth_zca_sweep(device)
        elif args.step == "broken_aug_t128":
            run_resumable_continuous_ssl("ssl_legacy_aug_t128_d64", aug_mode="legacy", seed=42 + 64, epochs=20, device=device)
        elif args.step == "pretrain_seed":
            if args.family == "ssl_kinematic":
                run_resumable_continuous_ssl(f"ssl_kinematic_d64_s{args.seed_idx}", aug_mode="kinematic", seed=1042 + 100 * args.seed_idx, epochs=20, device=device)
            else:
                run_resumable_token_ssl_seed(seed_idx=args.seed_idx, epochs=12, device=device)
        elif args.step == "finetune_curves":
            step_finetune_curves(args.family, device)
    finally:
        try:
            os.rmdir(lock_dir)
        except OSError:
            pass


if __name__ == "__main__":
    main()
