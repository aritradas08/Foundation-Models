#!/usr/bin/env python3
"""
Phase 2.2: VQ-VAE Tokenizer Audit & Derived-Observable Fidelity Evaluation.
Evaluates the K=8192 VQ-VAE track tokenizer on strictly held-out SM and BSM events
from eval_downstream.parquet, measuring:
  1. Codebook utilization (active codes / 8192) and perplexity exp(H(q)) across SM & BSM channels
  2. Track-level reconstruction fidelity (RMSE, R^2, Wasserstein-1 distance) on the 6 tokenized
     features [symlog(d0), symlog(z0), log(pT), eta, cos(dphi), sin(dphi)] and derived physical
     track coordinates [d0, z0, pT, px, py, pz]
  3. Event-level derived observable fidelity: H_T = sum(pT), leading track pT, |pT_miss|,
     and RMS transverse impact parameter d0.
"""
import json
import pickle
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT / "Tokenization-scripts" / "scripts"))

from common import PROCESS_NAMES, load_events_fast, raw6_to_vqvae6_np
from vqvae_tracks import TrackVQVAE

EVAL_PARQUET = Path("/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/data/eval_downstream.parquet")
VQVAE_DIR = Path("/global/cfs/cdirs/m4958/usr/aritra08/token/vqvae_tracks/sm_only_v1")
OUT_DIR = Path("/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/results/vqvae_fidelity")


def inv_symlog(y: np.ndarray) -> np.ndarray:
    return np.sign(y) * np.expm1(np.minimum(np.abs(y), 20.0))


def compute_codebook_stats(indices: np.ndarray, mask: np.ndarray, codebook_size: int = 8192) -> dict:
    valid_codes = indices[mask]
    counts = np.bincount(valid_codes, minlength=codebook_size).astype(np.float64)
    active = int((counts > 0).sum())
    probs = counts / max(counts.sum(), 1.0)
    nz = probs[probs > 0]
    entropy = float(-(nz * np.log(nz)).sum())
    perplexity = float(np.exp(entropy))
    return {
        "n_valid_tracks": int(len(valid_codes)),
        "active_codes": active,
        "utilization_pct": float(100.0 * active / codebook_size),
        "perplexity": perplexity,
        "entropy_nats": entropy,
    }


def metric_1d(true_arr: np.ndarray, recon_arr: np.ndarray, q_clip: float = 0.995) -> dict:
    lo = np.quantile(true_arr, 1.0 - q_clip)
    hi = np.quantile(true_arr, q_clip)
    sel = (true_arr >= lo) & (true_arr <= hi) & np.isfinite(true_arr) & np.isfinite(recon_arr)
    t_c = true_arr[sel]
    r_c = recon_arr[sel]
    rmse = float(np.sqrt(np.mean((t_c - r_c) ** 2)))
    mae = float(np.mean(np.abs(t_c - r_c)))
    iqr = float(np.percentile(t_c, 75) - np.percentile(t_c, 25))
    w1 = float(stats.wasserstein_distance(t_c, r_c))
    corr = float(np.corrcoef(t_c, r_c)[0, 1]) if len(t_c) > 2 else np.nan
    ss_res = np.sum((t_c - r_c) ** 2)
    ss_tot = np.sum((t_c - np.mean(t_c)) ** 2)
    r2 = float(1.0 - ss_res / max(ss_tot, 1e-12))
    return {
        "rmse": rmse,
        "mae": mae,
        "w1": w1,
        "w1_over_iqr": float(w1 / max(iqr, 1e-8)),
        "pearson_r": corr,
        "r2": r2,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(VQVAE_DIR / "scaler.pkl", "rb") as f:
        vq_scaler = pickle.load(f)
    ckpt = torch.load(VQVAE_DIR / "vqvae_best.pt", map_location=device)
    cfg = ckpt["cfg"]
    vqvae = TrackVQVAE(**cfg).to(device)
    vqvae.load_state_dict(ckpt["state_dict"])
    vqvae.eval()

    eval_data = load_events_fast(EVAL_PARQUET)
    labels_all = eval_data["labels"]
    raw6_all = eval_data["raw6"]
    mask_all = eval_data["mask"]

    # Subsample up to 15,000 events per process for fast, high-precision fidelity statistics
    rng = np.random.default_rng(42)
    sel_idx = []
    for lbl in sorted(np.unique(labels_all)):
        idx_l = np.where(labels_all == lbl)[0]
        take = rng.choice(idx_l, size=min(15_000, len(idx_l)), replace=False)
        sel_idx.append(take)
    sel_idx = np.sort(np.concatenate(sel_idx))

    raw6 = raw6_all[sel_idx]
    mask256 = mask_all[sel_idx]
    labels = labels_all[sel_idx]
    del eval_data, raw6_all, mask_all

    vq6_true, mask_vq = raw6_to_vqvae6_np(raw6, mask256)
    phi_lead = raw6[:, 0:1, 4]  # leading-track phi per event

    mean = vq_scaler.mean_.astype(np.float32).reshape(1, 1, 6)
    scale = vq_scaler.scale_.astype(np.float32).reshape(1, 1, 6)

    N, T, _ = vq6_true.shape
    vq6_recon = np.zeros_like(vq6_true)
    codes = np.full((N, T), -1, dtype=np.int32)

    with torch.no_grad():
        for start in range(0, N, 1024):
            end = min(start + 1024, N)
            m_b = mask_vq[start:end]
            x_b = (vq6_true[start:end] - mean) / scale
            x_b[~m_b] = 0.0
            x_t = torch.from_numpy(x_b).to(device)
            m_t = torch.from_numpy(m_b).to(device)
            x_hat_s, idx, _, _, _, _ = vqvae(x_t, m_t)
            x_hat = x_hat_s.cpu().numpy() * scale + mean
            x_hat[~m_b] = 0.0
            vq6_recon[start:end] = x_hat
            codes[start:end] = idx.cpu().numpy()

    # 1. Codebook statistics on SM vs BSM
    sm_ev = np.isin(labels, [0, 1, 2])
    codebook_stats = {
        "SM_combined": compute_codebook_stats(codes[sm_ev], mask_vq[sm_ev], cfg["codebook_size"]),
    }
    for lbl, pname in PROCESS_NAMES.items():
        ev_m = labels == lbl
        if ev_m.any():
            codebook_stats[pname] = compute_codebook_stats(codes[ev_m], mask_vq[ev_m], cfg["codebook_size"])

    # 2. Track-level metrics on held-out SM events
    m_sm = mask_vq[sm_ev]
    t_sm = vq6_true[sm_ev][m_sm]
    r_sm = vq6_recon[sm_ev][m_sm]

    feat_names = ["symlog_d0", "symlog_z0", "log_pt", "eta", "cos_dphi", "sin_dphi"]
    track_metrics = {}
    for i, fname in enumerate(feat_names):
        track_metrics[fname] = metric_1d(t_sm[:, i], r_sm[:, i])

    # Derived physical track coordinates on SM
    d0_t, d0_r = inv_symlog(t_sm[:, 0]), inv_symlog(r_sm[:, 0])
    z0_t, z0_r = inv_symlog(t_sm[:, 1]), inv_symlog(r_sm[:, 1])
    pt_t, pt_r = np.exp(np.clip(t_sm[:, 2], -10, 10)), np.exp(np.clip(r_sm[:, 2], -10, 10))
    eta_t, eta_r = t_sm[:, 3], np.clip(r_sm[:, 3], -6.0, 6.0)
    dphi_t = np.arctan2(t_sm[:, 5], t_sm[:, 4])
    dphi_r = np.arctan2(r_sm[:, 5], r_sm[:, 4])

    track_metrics["d0_mm"] = metric_1d(d0_t, d0_r)
    track_metrics["z0_mm"] = metric_1d(z0_t, z0_r)
    track_metrics["pt_GeV"] = metric_1d(pt_t, pt_r)
    track_metrics["dphi_rad"] = metric_1d(dphi_t, dphi_r)
    track_metrics["pz_GeV"] = metric_1d(pt_t * np.sinh(eta_t), pt_r * np.sinh(eta_r))

    # 3. Event-level derived observables on held-out SM events
    vq_t_sm = vq6_true[sm_ev]
    vq_r_sm = vq6_recon[sm_ev]
    phi_lead_sm = phi_lead[sm_ev]

    pt_ev_t = np.where(m_sm, np.exp(np.clip(vq_t_sm[:, :, 2], -10, 10)), 0.0)
    pt_ev_r = np.where(m_sm, np.exp(np.clip(vq_r_sm[:, :, 2], -10, 10)), 0.0)

    ht_t = pt_ev_t.sum(axis=1)
    ht_r = pt_ev_r.sum(axis=1)
    lead_pt_t = pt_ev_t.max(axis=1)
    lead_pt_r = pt_ev_r.max(axis=1)

    phi_ev_t = phi_lead_sm + np.arctan2(vq_t_sm[:, :, 5], vq_t_sm[:, :, 4])
    phi_ev_r = phi_lead_sm + np.arctan2(vq_r_sm[:, :, 5], vq_r_sm[:, :, 4])
    px_t = (pt_ev_t * np.cos(phi_ev_t)).sum(axis=1)
    py_t = (pt_ev_t * np.sin(phi_ev_t)).sum(axis=1)
    px_r = (pt_ev_r * np.cos(phi_ev_r)).sum(axis=1)
    py_r = (pt_ev_r * np.sin(phi_ev_r)).sum(axis=1)
    mpt_t = np.sqrt(px_t ** 2 + py_t ** 2)
    mpt_r = np.sqrt(px_r ** 2 + py_r ** 2)

    d0_sq_t = np.where(m_sm, inv_symlog(vq_t_sm[:, :, 0]) ** 2, 0.0)
    d0_sq_r = np.where(m_sm, inv_symlog(vq_r_sm[:, :, 0]) ** 2, 0.0)
    n_trk_sm = np.maximum(m_sm.sum(axis=1), 1)
    rms_d0_t = np.sqrt(d0_sq_t.sum(axis=1) / n_trk_sm)
    rms_d0_r = np.sqrt(d0_sq_r.sum(axis=1) / n_trk_sm)

    event_metrics = {
        "HT_GeV": metric_1d(ht_t, ht_r),
        "leading_pT_GeV": metric_1d(lead_pt_t, lead_pt_r),
        "track_MET_GeV": metric_1d(mpt_t, mpt_r),
        "rms_d0_mm": metric_1d(rms_d0_t, rms_d0_r),
    }

    summary = {
        "vqvae_cfg": cfg,
        "codebook_stats": codebook_stats,
        "track_level_fidelity_SM": track_metrics,
        "event_level_fidelity_SM": event_metrics,
    }
    with open(OUT_DIR / "vqvae_fidelity_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[evaluate_vqvae_fidelity] Saved {OUT_DIR / 'vqvae_fidelity_summary.json'}")
    print("SM Codebook stats:", codebook_stats["SM_combined"])
    for k, v in event_metrics.items():
        print(f"  Event observable {k:16s}: R2={v['r2']:.4f}, r={v['pearson_r']:.4f}, W1/IQR={v['w1_over_iqr']:.4f}")

    # Plot 4-panel derived event-observable & track-feature comparison
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.0))
    panels = [
        (t_sm[:, 0], r_sm[:, 0], r"Track $\mathrm{symlog}(d_0)$", (-2.0, 2.0), False),
        (t_sm[:, 2], r_sm[:, 2], r"Track $\log(p_T/\mathrm{GeV})$", (-0.5, 4.5), False),
        (t_sm[:, 3], r_sm[:, 3], r"Track $\eta$", (-3.5, 3.5), False),
        (dphi_t, dphi_r, r"Track $\Delta\phi(\mathrm{lead})$ [rad]", (-np.pi, np.pi), False),
        (ht_t, ht_r, r"Event $H_T = \sum p_T$ [GeV]", (0, 600), True),
        (lead_pt_t, lead_pt_r, r"Event Leading Track $p_T$ [GeV]", (0, 180), True),
    ]
    for ax, (arr_t, arr_r, xlabel, rng_x, logy) in zip(axes.flat, panels):
        ax.hist(arr_t, bins=60, range=rng_x, density=True, histtype="step", lw=1.8, color="black", label="Ground Truth")
        ax.hist(arr_r, bins=60, range=rng_x, density=True, alpha=0.45, color="royalblue", label="VQ-VAE ($K=8192$)")
        m = metric_1d(arr_t, arr_r)
        ax.set_xlabel(xlabel, fontsize=10.5)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(f"$R^2 = {m['r2']:.3f}$, $W_1/\\mathrm{{IQR}} = {m['w1_over_iqr']:.3f}$", fontsize=10)
        if logy:
            ax.set_yscale("log")
        ax.legend(frameon=False, fontsize=8.5)

    fig.suptitle("VQ-VAE ($K=8192$) Track & Derived Event-Observable Reconstruction Fidelity (Held-Out SM)", fontsize=12.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "vqvae_fidelity_panel.pdf")
    fig.savefig(OUT_DIR / "vqvae_fidelity_panel.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
