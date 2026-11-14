#!/usr/bin/env python3
"""
Workstreams 1 & 2 (Preparation & Extraction):
  1. Builds single canonical eval_downstream_1050k.parquet (1,050,000 rows, unique eval_uid, verified zero overlap
     with pretrain_sm_900k.parquet).
  2. Extracts clean, versioned embeddings in results/embeddings_v2/ through a single consistent pipeline:
       - Fixes the critical vqvae_code_d8 bug (pure pre-VQ z_e across all 1.05M events)
       - Adds vqvae_quant_d8 (pure post-VQ quantized z_q across all 1.05M events)
       - Adds 4 zero-NN handcrafted physics baselines fit strictly on pretrain_sm_900k:
           * raw_summary_d28 (D=28)
           * pca_summary_d8  (D=8, PCA fit strictly on pretrain_sm_900k)
           * phys_handcrafted_d6 (D=6: N_tracks, H_T, track_MET, RMS_d0, max_|d0|, N_|d0|>1mm)
           * mult_only_d1 (D=1: log1p(N_tracks))
       - Matches max_tracks=128 extraction for models trained with max_t<=128, and logs effective rank erank(Z).
  3. Trains unified ssl_tokens_d64 (15 epochs on pretrain_sm_900k, matching d8 and d32) and logs effective rank.
  4. Evaluates VQ-VAE fidelity including truth-defined displaced tails (|d0_true| > 1 mm, > 10 mm) on HP_v2.
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
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(REPO_ROOT / "Tokenization-scripts" / "scripts"))

from common import (
    MLPEncoder,
    PROCESS_NAMES,
    ProjectionHead,
    SM_LABELS,
    SetTransformerEncoder,
    compute_summaries28_np,
    fit_track_scaler,
    load_events_fast,
    raw6_to_feat7_np,
    raw6_to_vqvae6_np,
    symlog_np,
)
from train_and_extract import set_seed, simclr_nt_xent_loss

DATA_DIR = Path("/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/data")
PRETRAIN_PARQUET = DATA_DIR / "pretrain_sm_900k.parquet"
OLD_EVAL_PARQUET = DATA_DIR / "eval_downstream.parquet"
EXT_HP_PARQUET = DATA_DIR / "hp_v2_ext_90k.parquet"
CANON_EVAL_PARQUET = DATA_DIR / "eval_downstream_1050k.parquet"

RESULTS_BASE = Path("/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/results")
CKPT_BASE = RESULTS_BASE / "checkpoints"
EMBED_V2_DIR = RESULTS_BASE / "embeddings_v2"
FID_V2_DIR = RESULTS_BASE / "vqvae_fidelity_v2"
CACHE_DIR = Path("/pscratch/sd/d/danieltm/fm_cache")


def inv_symlog(y: np.ndarray) -> np.ndarray:
    return np.sign(y) * np.expm1(np.minimum(np.abs(y), 20.0))


def compute_effective_rank(z: np.ndarray, max_rows: int = 100_000) -> float:
    """Roy & Vetterli (2007) spectral entropy effective rank on centered embeddings."""
    if len(z) > max_rows:
        rng = np.random.default_rng(42)
        z = z[rng.choice(len(z), size=max_rows, replace=False)]
    zc = z.astype(np.float64) - z.mean(axis=0, dtype=np.float64)
    s = np.linalg.svd(zc, full_matrices=False, compute_uv=False)
    p = (s ** 2)
    tot = p.sum()
    if tot <= 1e-18:
        return 0.0
    p = p / tot
    p = p[p > 1e-18]
    return float(np.exp(-(p * np.log(p)).sum()))


def build_canonical_eval_parquet() -> Path:
    if CANON_EVAL_PARQUET.exists():
        pf = pq.ParquetFile(CANON_EVAL_PARQUET)
        if pf.metadata.num_rows == 1_050_000:
            print(f"[canonical] Verified {CANON_EVAL_PARQUET} (N=1,050,000).")
            return CANON_EVAL_PARQUET

    print("[canonical] Merging eval_downstream.parquet + hp_v2_ext_90k.parquet -> eval_downstream_1050k.parquet...")
    cols = ["event_id", "label", "d0", "z0", "p", "pt", "phi", "theta"]
    t1 = pq.read_table(OLD_EVAL_PARQUET, columns=cols)
    t2 = pq.read_table(EXT_HP_PARQUET, columns=cols)

    # Align schema types cleanly
    schema_fields = [
        pa.field("eval_uid", pa.int64()),
        pa.field("event_id", pa.int64()),
        pa.field("label", pa.int64()),
    ]
    for c in ["d0", "z0", "p", "pt", "phi", "theta"]:
        schema_fields.append(pa.field(c, t1.schema.field(c).type))

    def cast_tbl(t: pa.Table, offset: int) -> pa.Table:
        n = t.num_rows
        arrays = [
            pa.array(np.arange(offset, offset + n, dtype=np.int64)),
            pa.array(t["event_id"].to_numpy().astype(np.int64)),
            pa.array(t["label"].to_numpy().astype(np.int64)),
        ]
        for c in ["d0", "z0", "p", "pt", "phi", "theta"]:
            col = t[c]
            if col.type != t1.schema.field(c).type:
                col = col.cast(t1.schema.field(c).type)
            arrays.append(col)
        return pa.Table.from_arrays(arrays, schema=pa.schema(schema_fields))

    tc1 = cast_tbl(t1, 0)
    tc2 = cast_tbl(t2, t1.num_rows)
    full = pa.concat_tables([tc1, tc2])
    assert full.num_rows == 1_050_000, f"Expected 1,050,000 rows, got {full.num_rows}"

    # Verify strict composite (label, event_id) disjointness against pretrain_sm_900k.parquet
    pre_meta = pq.read_table(PRETRAIN_PARQUET, columns=["label", "event_id"])
    pre_keys = set(zip(pre_meta["label"].to_numpy().tolist(), pre_meta["event_id"].to_numpy().tolist()))
    ev_lbl = full["label"].to_numpy()
    ev_eid = full["event_id"].to_numpy()
    sm_mask = np.isin(ev_lbl, [0, 1, 2])
    eval_sm_keys = set(zip(ev_lbl[sm_mask].tolist(), ev_eid[sm_mask].tolist()))
    overlap = len(pre_keys.intersection(eval_sm_keys))
    print(f"[canonical] Composite (label, event_id) overlap between pretrain_sm_900k and eval SM = {overlap}")
    assert overlap == 0, f"CRITICAL: Found {overlap} overlapping SM events!"

    counts = {int(k): int(v) for k, v in zip(*np.unique(ev_lbl, return_counts=True))}
    print(f"[canonical] Class counts in eval_downstream_1050k: {counts}")

    tmp = CANON_EVAL_PARQUET.with_suffix(".tmp.parquet")
    pq.write_table(full, tmp, compression="snappy")
    os.replace(tmp, CANON_EVAL_PARQUET)
    print(f"[canonical] Saved {CANON_EVAL_PARQUET}")
    return CANON_EVAL_PARQUET


@torch.no_grad()
def extract_set_transformer(
    encoder: nn.Module,
    feat7_unscaled: np.ndarray,
    mask: np.ndarray,
    scaler: StandardScaler,
    device: torch.device,
    max_t_cap: int = 256,
    batch_size: int = 1024,
) -> np.ndarray:
    encoder.eval()
    N, T, F_dim = feat7_unscaled.shape
    mean = scaler.mean_.astype(np.float32).reshape(1, 1, F_dim)
    scale = scaler.scale_.astype(np.float32).reshape(1, 1, F_dim)

    all_z = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        m_b = mask[start:end, :max_t_cap]
        max_t = max(int(m_b.sum(axis=1).max()), 1)
        m_b = m_b[:, :max_t]
        x_b = (feat7_unscaled[start:end, :max_t, :] - mean) / scale
        x_b[~m_b] = 0.0
        x_t = torch.from_numpy(x_b).to(device, non_blocking=True)
        m_t = torch.from_numpy(m_b).to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            z = encoder(x_t, m_t)
        all_z.append(z.float().cpu().numpy())
    return np.concatenate(all_z, axis=0)


def compute_handcrafted6_np(raw6: np.ndarray, mask: np.ndarray, chunk_size: int = 32768) -> np.ndarray:
    """
    6-D zero-NN handcrafted collider + LLP summary vector per event:
      0: log1p(N_tracks)
      1: log1p(H_T [GeV])
      2: log1p(track_MET [GeV])
      3: symlog(RMS_d0 [mm])
      4: symlog(max_|d0| [mm])
      5: log1p(N_tracks with |d0| > 1.0 mm)
    Note: raw6[:, :, 0] is symlog(d0 [mm]), raw6[:, :, 3] is pt [GeV], raw6[:, :, 4] is phi [rad].
    """
    N = raw6.shape[0]
    out = np.zeros((N, 6), dtype=np.float32)
    thr_symlog_1mm = float(np.log1p(1.0))

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        m = mask[start:end]
        max_t = max(int(m.sum(axis=1).max()), 1)
        m = m[:, :max_t]
        r = raw6[start:end, :max_t, :]

        n_trk = m.sum(axis=1).astype(np.float32)
        pt = np.where(m, r[:, :, 3], 0.0)
        phi = r[:, :, 4]
        ht = pt.sum(axis=1)
        px = (pt * np.cos(phi)).sum(axis=1)
        py = (pt * np.sin(phi)).sum(axis=1)
        met = np.sqrt(px ** 2 + py ** 2)

        abs_symlog_d0 = np.where(m, np.abs(r[:, :, 0]), 0.0)
        d0_mm_abs = np.expm1(np.minimum(abs_symlog_d0, 20.0))
        d0_mm_abs = np.where(m, d0_mm_abs, 0.0)
        rms_d0 = np.sqrt((d0_mm_abs ** 2).sum(axis=1) / np.maximum(n_trk, 1.0))
        max_d0 = d0_mm_abs.max(axis=1)
        n_disp_1mm = ((abs_symlog_d0 > thr_symlog_1mm) & m).sum(axis=1).astype(np.float32)

        out[start:end, 0] = np.log1p(n_trk)
        out[start:end, 1] = np.log1p(np.maximum(ht, 0.0))
        out[start:end, 2] = np.log1p(np.maximum(met, 0.0))
        out[start:end, 3] = symlog_np(rms_d0)
        out[start:end, 4] = symlog_np(max_d0)
        out[start:end, 5] = np.log1p(n_disp_1mm)
    return out


def save_v2_npz(
    name: str,
    z: np.ndarray,
    eval_data: dict[str, np.ndarray],
    provenance: dict,
    extra: dict | None = None,
):
    EMBED_V2_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EMBED_V2_DIR / f"{name}.npz"
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
    if extra:
        payload.update(extra)
    tmp = out_path.with_suffix(".tmp.npz")
    np.savez(tmp, **payload)
    os.replace(tmp, out_path)
    print(f"[save_v2] {name:28s} | shape={str(z.shape):14s} | erank_SM={erank_sm:5.2f} | std_SM={embed_std_sm:.4f}", flush=True)
    return prov


def run_all_extractions(device: torch.device, force: bool = False):
    canon_pq = build_canonical_eval_parquet()
    eval_data = load_events_fast(canon_pq)
    feat7_eval = raw6_to_feat7_np(eval_data["raw6"], eval_data["mask"])
    mask_eval = eval_data["mask"]

    pre_data = load_events_fast(PRETRAIN_PARQUET, include_labels=SM_LABELS)
    feat7_pre = raw6_to_feat7_np(pre_data["raw6"], pre_data["mask"])
    mask_pre = pre_data["mask"]

    manifest = []

    # -------------------------------------------------------------------------
    # A. Zero-NN Handcrafted Physics Baselines (fit strictly on pretrain_sm_900k)
    # -------------------------------------------------------------------------
    print("\n=== Extracting Zero-NN Handcrafted Physics Baselines ===")
    trk_scaler_pre = fit_track_scaler(feat7_pre, mask_pre)
    sum28_pre = compute_summaries28_np(feat7_pre, mask_pre, scaler=trk_scaler_pre)
    sum28_scaler = StandardScaler().fit(sum28_pre)
    sum28_pre_s = sum28_scaler.transform(sum28_pre).astype(np.float32)

    pca8 = PCA(n_components=8, random_state=42).fit(sum28_pre_s)
    pca8_pre = pca8.transform(sum28_pre_s)
    pca8_scaler = StandardScaler().fit(pca8_pre)

    hc6_pre = compute_handcrafted6_np(pre_data["raw6"], mask_pre)
    hc6_scaler = StandardScaler().fit(hc6_pre)
    mult1_scaler = StandardScaler().fit(hc6_pre[:, 0:1])

    # Save baseline scalers for controls (2x2 & masking)
    base_ckpt_dir = CKPT_BASE / "physics_baselines"
    base_ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(base_ckpt_dir / "scalers.pkl", "wb") as f:
        pickle.dump({
            "trk_scaler": trk_scaler_pre,
            "sum28_scaler": sum28_scaler,
            "pca8": pca8,
            "pca8_scaler": pca8_scaler,
            "hc6_scaler": hc6_scaler,
            "mult1_scaler": mult1_scaler,
        }, f)

    sum28_eval = compute_summaries28_np(feat7_eval, mask_eval, scaler=trk_scaler_pre)
    sum28_eval_s = sum28_scaler.transform(sum28_eval).astype(np.float32)
    pca8_eval_s = pca8_scaler.transform(pca8.transform(sum28_eval_s)).astype(np.float32)

    hc6_eval = compute_handcrafted6_np(eval_data["raw6"], mask_eval)
    hc6_eval_s = hc6_scaler.transform(hc6_eval).astype(np.float32)
    mult1_eval_s = mult1_scaler.transform(hc6_eval[:, 0:1]).astype(np.float32)

    manifest.append(save_v2_npz("raw_summary_d28", sum28_eval_s, eval_data, {"family": "physics_baseline", "max_t": 256}))
    manifest.append(save_v2_npz("pca_summary_d8", pca8_eval_s, eval_data, {
        "family": "physics_baseline", "max_t": 256, "pca_explained_var_sum": float(pca8.explained_variance_ratio_.sum())
    }))
    manifest.append(save_v2_npz("phys_handcrafted_d6", hc6_eval_s, eval_data, {"family": "physics_baseline", "max_t": 256}))
    manifest.append(save_v2_npz("mult_only_d1", mult1_eval_s, eval_data, {"family": "physics_baseline", "max_t": 256}))

    del sum28_pre, sum28_pre_s, pca8_pre, hc6_pre, feat7_pre, mask_pre, pre_data

    # -------------------------------------------------------------------------
    # B. VQ-VAE Contextual Poolings (Fixing vqvae_code_d8 & adding vqvae_quant_d8)
    #    + Caching canonical VQ tokens & computing displaced-tail VQ fidelity
    # -------------------------------------------------------------------------
    print("\n=== Extracting VQ-VAE Poolings (Pure Pre-VQ z_e & Post-VQ z_q) + Displaced-Tail Fidelity ===")
    from vqvae_tracks import TrackVQVAE
    from pretrain_ssl_contrastive_tokens import TokenSetTransformerEncoder

    vq_dir = Path("/global/cfs/cdirs/m4958/usr/aritra08/token/vqvae_tracks/sm_only_v1")
    with open(vq_dir / "scaler.pkl", "rb") as f:
        vq_scaler = pickle.load(f)
    vq_ckpt = torch.load(vq_dir / "vqvae_best.pt", map_location=device)
    vq_cfg = vq_ckpt["cfg"]
    vqvae = TrackVQVAE(**vq_cfg).to(device)
    vqvae.load_state_dict(vq_ckpt["state_dict"])
    vqvae.eval()

    vq6_eval, mask_vq_eval = raw6_to_vqvae6_np(eval_data["raw6"], mask_eval)
    vq_mean = vq_scaler.mean_.astype(np.float32).reshape(1, 1, 6)
    vq_scale = vq_scaler.scale_.astype(np.float32).reshape(1, 1, 6)

    tok_legacy_ckpt = Path("/global/cfs/cdirs/m4958/usr/aritra08/token/ssl_tokens/sm_only_v1/encoder_ssl_tokens.pt")
    tok_legacy_enc64 = TokenSetTransformerEncoder(codebook_size=8192, latent_dim=64).to(device)
    tok_legacy_enc64.load_state_dict(torch.load(tok_legacy_ckpt, map_location=device))
    tok_legacy_enc64.eval()

    N = len(eval_data["labels"])
    z_vq_enc128 = []
    z_vq_pre8 = []
    z_vq_quant8 = []
    z_tok_leg64 = []
    tokens_eval = np.full((N, vq6_eval.shape[1]), -1, dtype=np.int16)

    with torch.no_grad():
        for start in range(0, N, 1024):
            end = min(start + 1024, N)
            m_b = mask_vq_eval[start:end]
            x_b = (vq6_eval[start:end] - vq_mean) / vq_scale
            x_b[~m_b] = 0.0
            x_t = torch.from_numpy(x_b).to(device, non_blocking=True)
            m_t = torch.from_numpy(m_b).to(device, non_blocking=True)

            h = vqvae.encoder(vqvae.enc_in(x_t), src_key_padding_mask=~m_t)
            h_norm = vqvae.enc_out[0](h)
            z_e = vqvae.enc_out[1](h_norm)
            z_q, idx, _, _, _ = vqvae.vq(z_e, m_t)

            m_f = m_t.unsqueeze(-1).float()
            denom = m_f.sum(dim=1).clamp(min=1.0)

            z_vq_enc128.append(((h_norm * m_f).sum(dim=1) / denom).cpu().numpy())
            z_vq_pre8.append(((z_e * m_f).sum(dim=1) / denom).cpu().numpy())
            z_vq_quant8.append(((z_q * m_f).sum(dim=1) / denom).cpu().numpy())
            tokens_eval[start:end] = idx.cpu().numpy().astype(np.int16)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                z_tl = tok_legacy_enc64(idx.clamp(min=0), m_t)
            z_tok_leg64.append(z_tl.float().cpu().numpy())

    tok_cache = CACHE_DIR / "eval_downstream_1050k_vq_tokens.npz"
    np.savez(tok_cache, tokens=tokens_eval, mask=mask_vq_eval)
    print(f"[vqvae] Saved canonical token cache {tok_cache}")

    manifest.append(save_v2_npz("vqvae_encoder_d128", np.concatenate(z_vq_enc128, axis=0), eval_data, {"family": "vqvae_pooling", "max_t": 128, "pooling": "h_norm"}))
    manifest.append(save_v2_npz("vqvae_code_d8", np.concatenate(z_vq_pre8, axis=0), eval_data, {"family": "vqvae_pooling", "max_t": 128, "pooling": "pre_vq_ze"}))
    manifest.append(save_v2_npz("vqvae_quant_d8", np.concatenate(z_vq_quant8, axis=0), eval_data, {"family": "vqvae_pooling", "max_t": 128, "pooling": "post_vq_zq"}))
    manifest.append(save_v2_npz("ssl_tokens_legacy_d64", np.concatenate(z_tok_leg64, axis=0), eval_data, {"family": "ssl_tokens_legacy", "max_t": 128}))

    # Displaced-tail VQ-VAE fidelity on ALL 100,000 higgs_portal_v2 events vs held-out SM subsample
    FID_V2_DIR.mkdir(parents=True, exist_ok=True)
    hp_idx = np.where(eval_data["labels"] == 3)[0]
    m_hp = mask_vq_eval[hp_idx]
    vq_hp_true = vq6_eval[hp_idx]
    vq_hp_rec = np.zeros_like(vq_hp_true)
    with torch.no_grad():
        for s in range(0, len(hp_idx), 1024):
            e = min(s + 1024, len(hp_idx))
            mb = m_hp[s:e]
            xb = (vq_hp_true[s:e] - vq_mean) / vq_scale
            xb[~mb] = 0.0
            xt = torch.from_numpy(xb).to(device)
            mt = torch.from_numpy(mb).to(device)
            xh_s, _, _, _, _, _ = vqvae(xt, mt)
            xh = xh_s.cpu().numpy() * vq_scale + vq_mean
            xh[~mb] = 0.0
            vq_hp_rec[s:e] = xh

    sl_d0_t = vq_hp_true[:, :, 0][m_hp]
    sl_d0_r = vq_hp_rec[:, :, 0][m_hp]
    d0_mm_t = inv_symlog(sl_d0_t)
    d0_mm_r = inv_symlog(sl_d0_r)
    abs_t = np.abs(d0_mm_t)
    abs_r = np.abs(d0_mm_r)

    tail_stats = {"n_total_hp_v2_tracks": int(len(d0_mm_t))}
    for thr in [0.5, 1.0, 5.0, 10.0, 20.0]:
        in_tail = abs_t > thr
        not_tail = ~in_tail
        n_tail = int(in_tail.sum())
        recall = float((abs_r[in_tail] > thr).mean()) if n_tail > 0 else np.nan
        fpr = float((abs_r[not_tail] > thr).mean()) if int(not_tail.sum()) > 0 else np.nan
        mae_mm = float(np.mean(np.abs(d0_mm_r[in_tail] - d0_mm_t[in_tail]))) if n_tail > 0 else np.nan
        med_rel_err = float(np.median(np.abs(d0_mm_r[in_tail] - d0_mm_t[in_tail]) / np.maximum(abs_t[in_tail], 1e-6))) if n_tail > 0 else np.nan
        mae_symlog = float(np.mean(np.abs(sl_d0_r[in_tail] - sl_d0_t[in_tail]))) if n_tail > 0 else np.nan
        bias_mm = float(np.mean(abs_r[in_tail] - abs_t[in_tail])) if n_tail > 0 else np.nan
        tail_stats[f"abs_d0_gt_{thr}mm"] = {
            "n_tracks": n_tail,
            "frac_of_hp_tracks": float(n_tail / len(d0_mm_t)),
            "threshold_recall_P_rec_gt_thr_given_true_gt_thr": recall,
            "threshold_fpr_P_rec_gt_thr_given_true_le_thr": fpr,
            "mae_d0_mm": mae_mm,
            "mean_signed_abs_bias_mm": bias_mm,
            "median_rel_abs_err": med_rel_err,
            "mae_symlog_d0": mae_symlog,
        }
    with open(FID_V2_DIR / "hp_v2_displaced_tail_fidelity.json", "w") as f:
        json.dump(tail_stats, f, indent=2)
    print("[vqvae] Saved displaced-tail fidelity:", json.dumps(tail_stats["abs_d0_gt_1.0mm"], indent=2))
    del vq_hp_true, vq_hp_rec

    # -------------------------------------------------------------------------
    # C. Extract Existing & Unified Checkpoints Cleanly
    # -------------------------------------------------------------------------
    print("\n=== Extracting All SetTransformer, Token, and Summary-AE Checkpoints ===")
    # 1. SupCon (trained in pretrain_supervised.py without max_t=128 cap, unaugmented views, tau=0.1)
    for d in [8, 32, 64]:
        cdir = Path(f"/global/cfs/cdirs/m4958/usr/aritra08/cont_learn/supervised/results/d{d}/checkpoints")
        with open(cdir / "scaler.pkl", "rb") as f:
            sc = pickle.load(f)
        enc = SetTransformerEncoder(latent_dim=d, norm_first=True, activation="gelu", use_out_norm=True).to(device)
        enc.load_state_dict(torch.load(cdir / "encoder_supervised_simclr.pt", map_location=device))
        z = extract_set_transformer(enc, feat7_eval, mask_eval, sc, device, max_t_cap=256)
        manifest.append(save_v2_npz(f"supcon_d{d}", z, eval_data, {"family": "supcon", "max_t": 256, "source": str(cdir)}))

    # 2. SSL Legacy (shortcut)
    ssl_paths = {
        8: Path("/global/cfs/cdirs/m4958/usr/aritra08/cont_learn/ssl/results/d8/checkpoints"),
        32: Path("/global/cfs/cdirs/m4958/usr/aritra08/cont_learn/ssl/results/d32/checkpoints"),
        64: Path("/global/cfs/cdirs/m4958/usr/aritra08/token/vqvae_tracks/self_supervised/sm_only_v1/checkpoints"),
    }
    for d, cdir in ssl_paths.items():
        with open(cdir / "scaler.pkl", "rb") as f:
            sc = pickle.load(f)
        enc = SetTransformerEncoder(latent_dim=d, norm_first=True, activation="gelu", use_out_norm=True).to(device)
        enc.load_state_dict(torch.load(cdir / "encoder_ssl_simclr.pt", map_location=device))
        z = extract_set_transformer(enc, feat7_eval, mask_eval, sc, device, max_t_cap=256)
        manifest.append(save_v2_npz(f"ssl_legacy_d{d}", z, eval_data, {"family": "ssl_legacy", "max_t": 256, "source": str(cdir)}))

    # 3. AE MLP
    mlp_runs = {8: "sm_only_v1_run2", 32: "sm_only_v2_run2", 64: "sm_only_v3_run2"}
    for d, rname in mlp_runs.items():
        cdir = Path(f"/global/cfs/cdirs/m4958/usr/aritra08/autoencoders/mlp/results/{rname}/checkpoints")
        with open(cdir / "scaler.pkl", "rb") as f:
            tsc = pickle.load(f)
        with open(cdir / "target_scaler.pkl", "rb") as f:
            tgsc = pickle.load(f)
        s28 = compute_summaries28_np(feat7_eval, mask_eval, scaler=tsc)
        s28_s = tgsc.transform(s28).astype(np.float32)
        enc = MLPEncoder(latent_dim=d).to(device)
        enc.load_state_dict(torch.load(cdir / "encoder_ae_mlp.pt", map_location=device))
        enc.eval()
        z_chunks = []
        with torch.no_grad():
            for st in range(0, N, 16384):
                z_chunks.append(enc(torch.from_numpy(s28_s[st:st + 16384]).to(device)).cpu().numpy())
        manifest.append(save_v2_npz(f"ae_mlp_d{d}", np.concatenate(z_chunks, axis=0), eval_data, {"family": "ae_mlp", "max_t": 256, "source": str(cdir)}))

    # 4. AE Transformer Post-LN (legacy)
    tr_runs = {8: "sm_only_v4", 32: "sm_only_v2", 64: "sm_only_v3"}
    for d, rname in tr_runs.items():
        cdir = Path(f"/global/cfs/cdirs/m4958/usr/aritra08/autoencoders/transformer/results/{rname}/checkpoints")
        with open(cdir / "scaler.pkl", "rb") as f:
            sc = pickle.load(f)
        enc = SetTransformerEncoder(latent_dim=d, norm_first=False, activation="relu", use_out_norm=False).to(device)
        enc.load_state_dict(torch.load(cdir / "encoder_ae_transformer.pt", map_location=device))
        z = extract_set_transformer(enc, feat7_eval, mask_eval, sc, device, max_t_cap=256)
        manifest.append(save_v2_npz(f"ae_transformer_postln_d{d}", z, eval_data, {"family": "ae_transformer_postln", "max_t": 256, "source": str(cdir)}))

    # 5. Random Init Baselines (max_t=128 matching unified SetTransformers)
    for d in [8, 32, 64]:
        set_seed(1000 + d)
        enc = SetTransformerEncoder(latent_dim=d, norm_first=True, activation="gelu", use_out_norm=True).to(device)
        z = extract_set_transformer(enc, feat7_eval, mask_eval, trk_scaler_pre, device, max_t_cap=128)
        manifest.append(save_v2_npz(f"random_init_d{d}", z, eval_data, {"family": "random_init", "max_t": 128}))

    # 6. Unified Suite Checkpoints (trained with max_t <= 128 -> extract with max_t_cap=128!)
    unified_st_models = [
        "ssl_kinematic_d8", "ssl_kinematic_d32", "ssl_kinematic_d64", "ssl_kinematic_drop_d64",
        "ae_transformer_d8", "ae_transformer_d32", "ae_transformer_d64",
        "supervised_e2e_d64",
    ]
    for mname in unified_st_models:
        cdir = CKPT_BASE / mname
        d = int(mname.split("_d")[-1])
        with open(cdir / "scaler.pkl", "rb") as f:
            sc = pickle.load(f)
        enc = SetTransformerEncoder(latent_dim=d, norm_first=True, activation="gelu", use_out_norm=True).to(device)
        enc.load_state_dict(torch.load(cdir / "encoder.pt", map_location=device))
        z = extract_set_transformer(enc, feat7_eval, mask_eval, sc, device, max_t_cap=128)
        extra = None
        if mname == "supervised_e2e_d64":
            head = nn.Linear(d, 3).to(device)
            head.load_state_dict(torch.load(cdir / "head.pt", map_location=device))
            head.eval()
            with torch.no_grad():
                logits = head(torch.from_numpy(z).to(device)).cpu().numpy().astype(np.float32)
            extra = {"sm3_logits": logits}
        manifest.append(save_v2_npz(mname, z, eval_data, {"family": mname.rsplit("_d", 1)[0], "max_t": 128, "source": str(cdir)}, extra=extra))

    # 7. Unified Token SSL d8, d32 (and d64 if already trained)
    for d in [8, 32, 64]:
        mname = f"ssl_tokens_d{d}"
        cdir = CKPT_BASE / mname
        if not (cdir / "encoder.pt").exists():
            print(f"[skip] {cdir / 'encoder.pt'} not trained yet; will extract after training.")
            continue
        tok_enc = TokenSetTransformerEncoder(codebook_size=8192, latent_dim=d).to(device)
        tok_enc.load_state_dict(torch.load(cdir / "encoder.pt", map_location=device))
        tok_enc.eval()
        z_list = []
        with torch.no_grad():
            for st in range(0, N, 2048):
                en = min(st + 2048, N)
                tb = torch.from_numpy(tokens_eval[st:en].astype(np.int64)).clamp(min=0).to(device)
                mb = torch.from_numpy(mask_vq_eval[st:en]).to(device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    z_list.append(tok_enc(tb, mb).float().cpu().numpy())
        manifest.append(save_v2_npz(mname, np.concatenate(z_list, axis=0), eval_data, {"family": "ssl_tokens", "max_t": 128, "source": str(cdir)}))

    with open(EMBED_V2_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[phase2_prepare_and_extract] Saved manifest with {len(manifest)} models to {EMBED_V2_DIR / 'manifest.json'}")


def train_unified_ssl_tokens_d64(epochs: int = 12, batch_size: int = 512, device: torch.device = torch.device("cuda")):
    from vqvae_tracks import TrackVQVAE
    from pretrain_ssl_contrastive_tokens import TokenSetTransformerEncoder

    tag = "ssl_tokens_d64"
    save_dir = CKPT_BASE / tag
    save_dir.mkdir(parents=True, exist_ok=True)
    set_seed(242 + 64)

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
    del pre_data["raw6"]

    encoder = TokenSetTransformerEncoder(codebook_size=8192, latent_dim=64).to(device)
    proj_head = ProjectionHead(in_dim=64, hidden_dim=64, out_dim=64).to(device)
    params = list(encoder.parameters()) + list(proj_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-6)

    N = len(pre_data["labels"])
    steps_per_epoch = N // batch_size
    total_steps = epochs * steps_per_epoch
    warmup_steps = 500
    lr_lambda = lambda step: min(1.0, (step + 1) / warmup_steps) * 0.5 * (1.0 + math.cos(math.pi * max(0.0, step - warmup_steps) / max(1, total_steps - warmup_steps)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def make_tokenized_pair(x_b: torch.Tensor, m_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, _ = x_b.shape
        x_2b = torch.cat([x_b, x_b], dim=0)
        m_2b = torch.cat([m_b, m_b], dim=0)
        drop_prob = 0.10 + 0.10 * torch.rand((2 * B, 1), device=device)
        keep_r = torch.rand((2 * B, T), device=device) >= drop_prob
        keep_r[:, 0] = True
        m_aug = m_2b & keep_r
        x_aug = x_2b + torch.randn_like(x_2b) * noise_vec
        x_scaled = ((x_aug - mean_t) / scale_t).masked_fill(~m_aug.unsqueeze(-1), 0.0)
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                z_e = vqvae.encode(x_scaled, m_aug)
            _, idx, _, _, _ = vqvae.vq(z_e.float(), m_aug)
        tok = idx.clamp(min=0)
        return tok[:B], m_aug[:B], tok[B:], m_aug[B:]

    vq6_gpu = torch.from_numpy(vq6_pre).to(device)
    mask_gpu = torch.from_numpy(mask_pre).to(device)
    lengths_vq_gpu = mask_gpu.sum(dim=1)
    del vq6_pre, mask_pre, pre_data

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

    print(f"[{tag}] Training unified TokenSetTransformerEncoder (epochs {start_epoch}..{epochs}) on {device}...", flush=True)
    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
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

            tok1, m1, tok2, m2 = make_tokenized_pair(x_b, m_b)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                z_2b = encoder(torch.cat([tok1, tok2], dim=0), torch.cat([m1, m2], dim=0))
                p_2b = proj_head(z_2b)
                B_cur = tok1.shape[0]
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
        torch.save({
            "epoch": epoch,
            "encoder": encoder.state_dict(),
            "proj_head": proj_head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "history": history,
        }, resume_ckpt)
        torch.save(encoder.state_dict(), save_dir / "encoder.pt")

    del vq6_gpu, mask_gpu, lengths_vq_gpu
    torch.cuda.empty_cache()

    torch.save(encoder.state_dict(), save_dir / "encoder.pt")
    torch.save(proj_head.state_dict(), save_dir / "proj_head.pt")
    pd.DataFrame(history).to_csv(save_dir / "training_history.csv", index=False)

    eval_data = load_events_fast(CANON_EVAL_PARQUET)
    tok_cache = CACHE_DIR / "eval_downstream_1050k_vq_tokens.npz"
    tc = np.load(tok_cache)
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
    save_v2_npz(tag, np.concatenate(z_list, axis=0), eval_data, {"family": "ssl_tokens", "max_t": 128, "source": str(save_dir), "epochs": epochs})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=str, required=True, choices=["extract_all", "train_ssl_tokens_d64"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    if args.step == "extract_all":
        run_all_extractions(device)
    elif args.step == "train_ssl_tokens_d64":
        train_unified_ssl_tokens_d64(epochs=args.epochs, device=device)


if __name__ == "__main__":
    main()
