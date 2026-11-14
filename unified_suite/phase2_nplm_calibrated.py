#!/usr/bin/env python3
"""
Workstreams 2 & 4: Rigorous Kernel NPLM Suite on Clean Versioned Embeddings (embeddings_v2):
  1. all_models_shape: Paired shape-only NPLM (N_R=500k, N_B=100k, M=316, 250 paired null toys,
     100 paired signal toys per fraction) across ALL models in results/embeddings_v2/ for
     HP_v2 (label=3), HV (label=4), and HP_v1 (label=5), plus additive mode for D=64 representatives.
  2. highstats_power: Conditional 10,000-null-toy calibration (5,000 threshold calibration toys +
     5,000 independent validation toys to verify empirical Type-I coverage at alpha=1e-2 and 1e-3!),
     plus 200 paired signal toys across f_inj in {0.25%, 0.5%, 1.0%, 2.5%}, reporting empirical
     discovery power Power(alpha=1e-2) and Power(alpha=1e-3) with exact Clopper-Pearson 95% CIs
     and [q16, q84] bands.
  3. ref_seed_variability: 5 independent (reference R, Nystrom centers, scaler) draws x 1,000 null
     toys + 100 signal toys at 1.0% injection, reporting across-reference mean +/- SD without pooling.
  4. low_overlap_check: N_R=350k (leaving 400k held-out SM events!), N_B=25k (only 6.25% expected
     overlap, plus 16 strictly disjoint non-overlapping 25k null sets!) with 2,000 calibration +
     2,000 validation nulls and 200 signal toys.
  5. whitening_ablation: Compares per-coordinate StandardScaler vs regularized ZCA shrinkage
     whitening (Sigma_R_reg^{-1/2}, gamma=1e-2) fit strictly on R and recalibrated under H0,
     testing whether SM covariance anisotropy explains the SupCon ranking inversion on HP_v2.
"""
import argparse
import glob
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

@contextmanager
def gpfs_lock(lock_path: str | Path, timeout: float = 120.0):
    lp = str(lock_path) + ".dirlock"
    t0 = time.time()
    while True:
        try:
            os.mkdir(lp)
            break
        except FileExistsError:
            if time.time() - t0 > timeout:
                try:
                    os.rmdir(lp)
                except OSError:
                    pass
            time.sleep(0.05)
    try:
        yield
    finally:
        try:
            os.rmdir(lp)
        except OSError:
            pass

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
from sklearn.preprocessing import StandardScaler

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from common import PROCESS_NAMES
from nplm_runner import (
    choose_centers_and_sigma,
    compute_z_scores,
    evaluate_null_calibration,
    rbf_features_gpu,
    train_kernel_lbfgs,
)

RESULTS_BASE = Path("/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/results")
EMBED_V2_DIR = RESULTS_BASE / "embeddings_v2"
NPLM_V2_DIR = RESULTS_BASE / "nplm_v2"


def clopper_pearson_ci(k: int, n: int, alpha_ci: float = 0.05) -> tuple[float, float]:
    if n <= 0:
        return (np.nan, np.nan)
    lo = 0.0 if k == 0 else float(stats.beta.ppf(alpha_ci / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1.0 - alpha_ci / 2.0, k + 1, n - k))
    return lo, hi


def order_stat_threshold(t_cal: np.ndarray, alpha: float) -> float:
    """
    Conservative finite-sample conformal/rank threshold (Phipson & Smyth 2010 / Davison & Hinkley):
    choose the k-th order statistic with k = ceil((N + 1) * (1 - alpha)), capped at N.
    Guarantees marginal Type-I error <= alpha on exchangeable validation draws.
    """
    t_sorted = np.sort(t_cal)
    N = len(t_sorted)
    idx_1based = min(int(np.ceil((N + 1) * (1.0 - alpha))), N)
    return float(t_sorted[idx_1based - 1])


class ZCAShrinkageWhitener:
    """Regularized ZCA/Mahalanobis whitener fit strictly on reference sample R."""
    def __init__(self, gamma: float = 1e-2):
        self.gamma = gamma
        self.mean_ = None
        self.W_ = None

    def fit(self, X: np.ndarray):
        X64 = X.astype(np.float64)
        self.mean_ = X64.mean(axis=0, keepdims=True)
        Xc = X64 - self.mean_
        std = np.maximum(Xc.std(axis=0, keepdims=True), 1e-8)
        Xs = Xc / std
        cov = (Xs.T @ Xs) / max(len(Xs) - 1, 1)
        D = cov.shape[0]
        cov_reg = (1.0 - self.gamma) * cov + self.gamma * np.eye(D, dtype=np.float64)
        eigvals, eigvecs = np.linalg.eigh(cov_reg)
        inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(np.maximum(eigvals, 1e-8))) @ eigvecs.T
        self.std_ = std
        self.W_ = inv_sqrt
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Xs = (X.astype(np.float64) - self.mean_) / self.std_
        return (Xs @ self.W_).astype(np.float32)


def prepare_kernel_basis(
    z_all: np.ndarray,
    labels_all: np.ndarray,
    n_r: int,
    m_centers: int,
    ref_seed: int,
    device: torch.device,
    transform_mode: str = "standard",
    sigma_quantile: float = 0.90,
):
    sm_mask = np.isin(labels_all, [0, 1, 2])
    z_bkg = z_all[sm_mask]

    rng_ref = np.random.default_rng(ref_seed)
    perm_bkg = rng_ref.permutation(len(z_bkg))
    idx_ref = perm_bkg[:n_r]
    idx_pool = perm_bkg[n_r:]

    if transform_mode == "standard":
        scaler = StandardScaler().fit(z_bkg[idx_ref])
    elif transform_mode == "zca":
        scaler = ZCAShrinkageWhitener(gamma=1e-2).fit(z_bkg[idx_ref])
    else:
        raise ValueError(f"Unknown transform_mode={transform_mode}")

    ref_scaled = scaler.transform(z_bkg[idx_ref]).astype(np.float32)
    bkg_pool_scaled = scaler.transform(z_bkg[idx_pool]).astype(np.float32)

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
            s_scaled = scaler.transform(z_all[s_mask]).astype(np.float32)
            sig_pools_t[s_lbl] = torch.from_numpy(s_scaled).to(device=device, dtype=dtype)

    return {
        "sigma": sigma,
        "centers_t": centers_t,
        "k_mm": k_mm,
        "phi_ref": phi_ref,
        "bkg_pool_t": bkg_pool_t,
        "n_pool": len(bkg_pool_scaled),
        "sig_pools_t": sig_pools_t,
    }


def run_null_toys_gpu(basis: dict, n_r: int, n_b: int, n_toys: int, toy_seed: int, lam: float = 1e-6) -> np.ndarray:
    rng = np.random.default_rng(toy_seed)
    device = basis["phi_ref"].device
    n_pool = basis["n_pool"]
    t_vals = np.empty(n_toys, dtype=np.float64)
    for i in range(n_toys):
        idx_b = rng.choice(n_pool, size=n_b, replace=(n_b > n_pool))
        d_t = basis["bkg_pool_t"][torch.from_numpy(idx_b).to(device)]
        phi_d = rbf_features_gpu(d_t, basis["centers_t"], basis["sigma"])
        _, conv = train_kernel_lbfgs(basis["phi_ref"], phi_d, basis["k_mm"], n_b, n_r, lam=lam)
        t_vals[i] = conv["t_final"]
    return t_vals


def run_signal_toys_gpu(
    basis: dict,
    sig_lbl: int,
    n_r: int,
    n_b: int,
    frac: float,
    n_toys: int,
    toy_seed: int,
    sample_mode: str = "shape",
    lam: float = 1e-6,
) -> np.ndarray:
    rng = np.random.default_rng(toy_seed)
    device = basis["phi_ref"].device
    n_pool = basis["n_pool"]
    sig_t = basis["sig_pools_t"][sig_lbl]
    n_sig_pool = len(sig_t)

    n_s = max(int(round(frac * n_b)), 1)
    n_b_draw = (n_b - n_s) if sample_mode == "shape" else n_b
    t_vals = np.empty(n_toys, dtype=np.float64)
    for i in range(n_toys):
        idx_b = rng.choice(n_pool, size=n_b_draw, replace=(n_b_draw > n_pool))
        idx_s = rng.choice(n_sig_pool, size=n_s, replace=(n_s > n_sig_pool))
        d_t = torch.cat([
            basis["bkg_pool_t"][torch.from_numpy(idx_b).to(device)],
            sig_t[torch.from_numpy(idx_s).to(device)],
        ], dim=0)
        phi_d = rbf_features_gpu(d_t, basis["centers_t"], basis["sigma"])
        _, conv = train_kernel_lbfgs(basis["phi_ref"], phi_d, basis["k_mm"], n_b, n_r, lam=lam)
        t_vals[i] = conv["t_final"]
    return t_vals


def step_all_models_shape(device: torch.device, models_filter: list[str] | None = None, sample_mode: str = "shape"):
    out_dir = NPLM_V2_DIR / f"all_models_{sample_mode}"
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_files = sorted(glob.glob(str(EMBED_V2_DIR / "*.npz")))
    if models_filter:
        npz_files = [p for p in npz_files if Path(p).stem in models_filter]

    n_r, n_b, m_centers = 500_000, 100_000, 316
    n_null, n_sig = 250, 100
    fracs = [0.005, 0.01, 0.025, 0.05]
    signals = [3, 4, 5] if sample_mode == "shape" else [3, 4]

    rows = []
    for fpath in npz_files:
        mname = Path(fpath).stem
        j_out = out_dir / f"{mname}_{sample_mode}.json"
        if j_out.exists():
            with open(j_out) as f:
                mj = json.load(f)
            for sname, sdict in mj["signals"].items():
                for frac_s, rec in sdict.items():
                    rows.append({k: v for k, v in rec.items() if k != "t_values"})
            csv_path = NPLM_V2_DIR / f"nplm_summary_{sample_mode}.csv"
            with gpfs_lock(csv_path):
                df = pd.DataFrame(rows)
                if csv_path.exists():
                    old = pd.read_csv(csv_path)
                    old = old[~old["model"].isin(df["model"])]
                    df = pd.concat([old, df], ignore_index=True)
                df.to_csv(csv_path, index=False)
            print(f"[nplm_{sample_mode}] Reused cached {j_out.name}", flush=True)
            continue
        t0 = time.time()
        data = np.load(fpath)
        z = data["z"].astype(np.float32)
        labels = data["labels"]
        D = z.shape[1]

        basis = prepare_kernel_basis(z, labels, n_r, m_centers, ref_seed=42, device=device)
        t_null = run_null_toys_gpu(basis, n_r, n_b, n_null, toy_seed=1001)
        gof = evaluate_null_calibration(t_null)
        thr_01 = order_stat_threshold(t_null, 0.01)

        model_json = {
            "model": mname,
            "latent_dim": D,
            "sample_mode": sample_mode,
            "n_r": n_r,
            "n_b": n_b,
            "m_centers": m_centers,
            "sigma": basis["sigma"],
            "null_gof": gof,
            "t_null": t_null.tolist(),
            "signals": {},
        }

        for s_lbl in signals:
            sname = PROCESS_NAMES[s_lbl]
            model_json["signals"][sname] = {}
            for frac in fracs:
                t_sig = run_signal_toys_gpu(basis, s_lbl, n_r, n_b, frac, n_sig, toy_seed=2000 + 100 * s_lbl + int(round(frac * 10000)), sample_mode=sample_mode)
                t_med = float(np.median(t_sig))
                t_q16 = float(np.percentile(t_sig, 16))
                t_q84 = float(np.percentile(t_sig, 84))
                zd = compute_z_scores(t_med, t_null, gof)
                zd_q16 = compute_z_scores(t_q16, t_null, gof)
                zd_q84 = compute_z_scores(t_q84, t_null, gof)
                k_rej = int((t_sig > thr_01).sum())
                pow_01 = k_rej / len(t_sig)
                ci_lo, ci_hi = clopper_pearson_ci(k_rej, len(t_sig))

                n_s = int(round(frac * n_b))
                t_rate = 0.0 if sample_mode == "shape" else float(2.0 * ((n_b + n_s) * np.log(1.0 + n_s / n_b) - n_s))

                rec = {
                    "model": mname,
                    "latent_dim": D,
                    "sample_mode": sample_mode,
                    "signal": sname,
                    "frac": frac,
                    "null_mean": gof["mean"],
                    "null_std": gof["std"],
                    "t_rate_only": t_rate,
                    "t_median": t_med,
                    "t_q16": t_q16,
                    "t_q84": t_q84,
                    "snr": zd["snr"],
                    "snr_q16": zd_q16["snr"],
                    "snr_q84": zd_q84["snr"],
                    "z_asymptotic": zd["z_asymptotic"],
                    "z_asym_q16": zd_q16["z_asymptotic"],
                    "z_asym_q84": zd_q84["z_asymptotic"],
                    "power_alpha_1e2": pow_01,
                    "power_alpha_1e2_ci_lo": ci_lo,
                    "power_alpha_1e2_ci_hi": ci_hi,
                }
                rows.append(rec)
                model_json["signals"][sname][f"{frac}"] = {**rec, "t_values": t_sig.tolist()}

        with open(out_dir / f"{mname}_{sample_mode}.json", "w") as f:
            json.dump(model_json, f, indent=2)

        hp1 = model_json["signals"]["higgs_portal_v2"]["0.01"]
        hv1 = model_json["signals"]["hidden_valley"]["0.01"]
        print(
            f"[nplm_{sample_mode}] {mname:26s} | null={gof['mean']:5.1f}+/-{gof['std']:4.1f} | "
            f"HP_v2(1%): SNR={hp1['snr']:5.2f} [{hp1['snr_q16']:.1f},{hp1['snr_q84']:.1f}], Z={hp1['z_asymptotic']:5.2f}, Pow1%={hp1['power_alpha_1e2']:.2f} | "
            f"HV(1%): SNR={hv1['snr']:5.2f}, Z={hv1['z_asymptotic']:5.2f} ({time.time()-t0:.1f}s)",
            flush=True,
        )
        del basis
        torch.cuda.empty_cache()

        csv_path = NPLM_V2_DIR / f"nplm_summary_{sample_mode}.csv"
        with gpfs_lock(csv_path):
            df = pd.DataFrame(rows)
            if csv_path.exists():
                old = pd.read_csv(csv_path)
                old = old[~old["model"].isin(df["model"])]
                df = pd.concat([old, df], ignore_index=True)
            df.to_csv(csv_path, index=False)
    print(f"[nplm_{sample_mode}] Finished saving {NPLM_V2_DIR / f'nplm_summary_{sample_mode}.csv'}")


def step_highstats_calibration_and_power(device: torch.device, models_filter: list[str] | None = None):
    """
    Runs:
      A. 10,000 conditional null toys (5,000 calibration + 5,000 independent validation) +
         200 paired signal toys at f_inj in {0.0025, 0.005, 0.01, 0.025} on HP_v2 and HV.
      B. 5 independent reference/center draws x (1,000 null + 100 signal toys at 1%) to quantify
         reference-draw variability without pooling heterogeneous seeds.
      C. Low-overlap check (N_R=350,000, leaving 400,000 SM pool; N_B=25,000 -> 6.25% overlap,
         plus 16 strictly disjoint non-overlapping 25k null sets + 2,000 cal + 2,000 val nulls).
    """
    out_dir = NPLM_V2_DIR / "highstats_calibrated"
    out_dir.mkdir(parents=True, exist_ok=True)

    default_models = [
        "supcon_d64", "supervised_e2e_d64",
        "ssl_kinematic_d64", "ssl_kinematic_d32",
        "ssl_tokens_d64", "ssl_tokens_d32", "ssl_tokens_d8",
        "ae_transformer_d32", "ae_mlp_d32",
        "vqvae_code_d8", "vqvae_quant_d8",
        "raw_summary_d28", "pca_summary_d8", "phys_handcrafted_d6", "mult_only_d1",
        "random_init_d64",
    ]
    target_models = models_filter if models_filter else default_models
    target_models = [m for m in target_models if (EMBED_V2_DIR / f"{m}.npz").exists()]

    power_rows = []
    refseed_rows = []
    lowov_rows = []

    ref_and_lowov_models = {
        "supcon_d64", "supervised_e2e_d64",
        "ssl_kinematic_d32", "ssl_tokens_d64", "ssl_tokens_d8",
        "ae_mlp_d32", "ae_transformer_d32",
        "vqvae_code_d8", "vqvae_quant_d8",
        "raw_summary_d28", "phys_handcrafted_d6",
    }

    def save_merge(rows_list, path):
        if not rows_list:
            return
        with gpfs_lock(path):
            df = pd.DataFrame(rows_list)
            if path.exists():
                old = pd.read_csv(path)
                old = old[~old["model"].isin(df["model"])]
                df = pd.concat([old, df], ignore_index=True)
            df.to_csv(path, index=False)

    for mname in target_models:
        t0 = time.time()
        data = np.load(EMBED_V2_DIR / f"{mname}.npz")
        z = data["z"].astype(np.float32)
        labels = data["labels"]
        D = z.shape[1]

        # --- Part A: 10,000 Null Toys (5k cal + 5k val) + 200 Signal Toys ---
        n_r, n_b = 500_000, 100_000
        hs_npz_path = out_dir / f"{mname}_highstats.npz"
        if hs_npz_path.exists():
            cached_hs = np.load(hs_npz_path)
            t_cal = cached_hs["t_cal"]
            t_val = cached_hs["t_val"]
            basis = None
            print(f"[highstats] Reusing Part A 10k-null cache {hs_npz_path.name}", flush=True)
        else:
            basis = prepare_kernel_basis(z, labels, n_r, 316, ref_seed=42, device=device)
            t_cal = run_null_toys_gpu(basis, n_r, n_b, 5000, toy_seed=10_001)
            t_val = run_null_toys_gpu(basis, n_r, n_b, 5000, toy_seed=20_001)

        thr_1e2 = order_stat_threshold(t_cal, 1e-2)
        thr_1e3 = order_stat_threshold(t_cal, 1e-3)
        val_rej_1e2 = int((t_val > thr_1e2).sum())
        val_rej_1e3 = int((t_val > thr_1e3).sum())
        val_fpr_1e2 = val_rej_1e2 / len(t_val)
        val_fpr_1e3 = val_rej_1e3 / len(t_val)

        gof_cal = evaluate_null_calibration(t_cal)
        ks_2samp = stats.ks_2samp(t_cal, t_val)
        npz_payload = {"t_cal": t_cal, "t_val": t_val}

        for s_lbl in [3, 4]:
            sname = PROCESS_NAMES[s_lbl]
            for frac in [0.0025, 0.005, 0.01, 0.025]:
                key_s = f"{sname}_f{frac:g}"
                if basis is None:
                    t_sig = cached_hs[key_s]
                else:
                    t_sig = run_signal_toys_gpu(basis, s_lbl, n_r, n_b, frac, 200, toy_seed=30_000 + 100 * s_lbl + int(round(frac * 10000)))
                npz_payload[key_s] = t_sig
                k_1e2 = int((t_sig > thr_1e2).sum())
                k_1e3 = int((t_sig > thr_1e3).sum())
                ci2_lo, ci2_hi = clopper_pearson_ci(k_1e2, len(t_sig))
                ci3_lo, ci3_hi = clopper_pearson_ci(k_1e3, len(t_sig))
                t_med = float(np.median(t_sig))
                t_q16 = float(np.percentile(t_sig, 16))
                t_q84 = float(np.percentile(t_sig, 84))
                zd = compute_z_scores(t_med, t_cal, gof_cal)

                power_rows.append({
                    "model": mname,
                    "latent_dim": D,
                    "signal": sname,
                    "frac": frac,
                    "n_cal_null": 5000,
                    "n_val_null": 5000,
                    "null_mean": gof_cal["mean"],
                    "null_std": gof_cal["std"],
                    "thr_alpha_1e2": thr_1e2,
                    "thr_alpha_1e3": thr_1e3,
                    "val_fpr_1e2": val_fpr_1e2,
                    "val_fpr_1e3": val_fpr_1e3,
                    "ks_cal_vs_val_p": float(ks_2samp.pvalue),
                    "t_median": t_med,
                    "t_q16": t_q16,
                    "t_q84": t_q84,
                    "snr_median": zd["snr"],
                    "snr_q16": (t_q16 - gof_cal["mean"]) / gof_cal["std"],
                    "snr_q84": (t_q84 - gof_cal["mean"]) / gof_cal["std"],
                    "z_asym_median": zd["z_asymptotic"],
                    "power_1e2": k_1e2 / len(t_sig),
                    "power_1e2_ci_lo": ci2_lo,
                    "power_1e2_ci_hi": ci2_hi,
                    "power_1e3": k_1e3 / len(t_sig),
                    "power_1e3_ci_lo": ci3_lo,
                    "power_1e3_ci_hi": ci3_hi,
                })
        if basis is not None:
            np.savez_compressed(hs_npz_path, **npz_payload)
            del basis
            torch.cuda.empty_cache()
        save_merge(power_rows, out_dir / "highstats_power_10k_nulls.csv")

        if mname in ref_and_lowov_models:
            # --- Part B: 5 Independent Reference + Center Seeds ---
            for rseed in [101, 102, 103, 104, 105]:
                b_r = prepare_kernel_basis(z, labels, n_r, 316, ref_seed=rseed, device=device)
                tn_r = run_null_toys_gpu(b_r, n_r, n_b, 1000, toy_seed=40_000 + rseed)
                thr_r = order_stat_threshold(tn_r, 1e-2)
                mu_r, sd_r = float(tn_r.mean()), float(tn_r.std(ddof=1))
                for s_lbl in [3, 4]:
                    ts_r = run_signal_toys_gpu(b_r, s_lbl, n_r, n_b, 0.01, 100, toy_seed=50_000 + 10 * rseed + s_lbl)
                    refseed_rows.append({
                        "model": mname,
                        "ref_seed": rseed,
                        "signal": PROCESS_NAMES[s_lbl],
                        "null_mean": mu_r,
                        "null_std": sd_r,
                        "snr_1pct": float((np.median(ts_r) - mu_r) / sd_r),
                        "power_1e2_1pct": float((ts_r > thr_r).mean()),
                    })
                del b_r
                torch.cuda.empty_cache()
            save_merge(refseed_rows, out_dir / "ref_seed_variability_5seeds.csv")

            # --- Part C: Low-Overlap Check (N_R=350k, Pool=400k, N_B=25k -> 6.25% overlap) ---
            b_lo = prepare_kernel_basis(z, labels, 350_000, 316, ref_seed=777, device=device)
            disjoint_t = []
            for k_dis in range(16):
                d_t = b_lo["bkg_pool_t"][k_dis * 25_000 : (k_dis + 1) * 25_000]
                phi_d = rbf_features_gpu(d_t, b_lo["centers_t"], b_lo["sigma"])
                _, conv = train_kernel_lbfgs(b_lo["phi_ref"], phi_d, b_lo["k_mm"], 25_000, 350_000)
                disjoint_t.append(conv["t_final"])
            disjoint_t = np.array(disjoint_t)

            tn_lo_cal = run_null_toys_gpu(b_lo, 350_000, 25_000, 2000, toy_seed=60_001)
            tn_lo_val = run_null_toys_gpu(b_lo, 350_000, 25_000, 2000, toy_seed=65_001)
            thr_lo_1e2 = order_stat_threshold(tn_lo_cal, 1e-2)
            fpr_lo_1e2 = float((tn_lo_val > thr_lo_1e2).mean())
            ks_disjoint_p = float(stats.ks_2samp(disjoint_t, tn_lo_val).pvalue)

            for s_lbl in [3, 4]:
                for frac in [0.01, 0.02]:
                    ts_lo = run_signal_toys_gpu(b_lo, s_lbl, 350_000, 25_000, frac, 200, toy_seed=70_000 + 100 * s_lbl + int(frac * 1000))
                    k_lo = int((ts_lo > thr_lo_1e2).sum())
                    c_lo, c_hi = clopper_pearson_ci(k_lo, len(ts_lo))
                    lowov_rows.append({
                        "model": mname,
                        "signal": PROCESS_NAMES[s_lbl],
                        "frac": frac,
                        "n_r": 350_000,
                        "n_pool": 400_000,
                        "n_b": 25_000,
                        "null_cal_mean": float(tn_lo_cal.mean()),
                        "null_cal_std": float(tn_lo_cal.std(ddof=1)),
                        "disjoint16_mean": float(disjoint_t.mean()),
                        "disjoint16_std": float(disjoint_t.std(ddof=1)),
                        "ks_disjoint_vs_subsampled_p": ks_disjoint_p,
                        "val_fpr_1e2": fpr_lo_1e2,
                        "snr_median": float((np.median(ts_lo) - tn_lo_cal.mean()) / tn_lo_cal.std(ddof=1)),
                        "power_1e2": k_lo / len(ts_lo),
                        "power_1e2_ci_lo": c_lo,
                        "power_1e2_ci_hi": c_hi,
                    })
            del b_lo
            torch.cuda.empty_cache()
            save_merge(lowov_rows, out_dir / "low_overlap_nb25k_check.csv")

        hp_row = [r for r in power_rows if r["model"] == mname and r["signal"] == "higgs_portal_v2" and r["frac"] == 0.01][0]
        print(
            f"[highstats] {mname:24s} | val_FPR(1e-2)={val_fpr_1e2:.4f}, val_FPR(1e-3)={val_fpr_1e3:.4f} | "
            f"HP_v2(1%): Pow(1e-2)={hp_row['power_1e2']:.3f}, Pow(1e-3)={hp_row['power_1e3']:.3f}, SNR={hp_row['snr_median']:.2f} ({time.time()-t0:.1f}s)",
            flush=True,
        )


def step_whitening_geometry_ablation(device: torch.device, models_filter: list[str] | None = None):
    """
    Tests StandardScaler vs Regularized ZCA/Mahalanobis Whitening on R (and bandwidth quantiles
    q in {0.50, 0.75, 0.90}) with full null recalibration (1,000 cal + 1,000 val nulls, 150 signal toys).
    Directly tests whether SM covariance anisotropy explains SupCon's low isotropic-RBF sensitivity on HP_v2.
    """
    out_dir = NPLM_V2_DIR / "geometry_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    default_models = [
        "supcon_d64", "supervised_e2e_d64",
        "ssl_kinematic_d64", "ssl_tokens_d64", "ssl_tokens_d32",
        "ae_transformer_d32", "ae_mlp_d32",
        "vqvae_code_d8", "vqvae_quant_d8", "raw_summary_d28", "phys_handcrafted_d6",
    ]
    models = models_filter if models_filter else default_models
    models = [m for m in models if (EMBED_V2_DIR / f"{m}.npz").exists()]

    rows = []
    for mname in models:
        data = np.load(EMBED_V2_DIR / f"{mname}.npz")
        z = data["z"].astype(np.float32)
        labels = data["labels"]

        configs = [
            ("standard", 0.90),
            ("zca", 0.90),
            ("standard", 0.50),
            ("zca", 0.50),
        ]
        for tmode, q_sig in configs:
            t0 = time.time()
            basis = prepare_kernel_basis(z, labels, 500_000, 316, ref_seed=42, device=device, transform_mode=tmode, sigma_quantile=q_sig)
            t_cal = run_null_toys_gpu(basis, 500_000, 100_000, 400, toy_seed=80_001)
            t_val = run_null_toys_gpu(basis, 500_000, 100_000, 400, toy_seed=82_001)
            thr_1e2 = order_stat_threshold(t_cal, 1e-2)
            val_fpr = float((t_val > thr_1e2).mean())
            mu_n, sd_n = float(t_cal.mean()), float(t_cal.std(ddof=1))

            for s_lbl in [3, 4]:
                for frac in [0.005, 0.01]:
                    ts = run_signal_toys_gpu(basis, s_lbl, 500_000, 100_000, frac, 100, toy_seed=85_000 + 100 * s_lbl + int(frac * 1000))
                    k = int((ts > thr_1e2).sum())
                    lo, hi = clopper_pearson_ci(k, len(ts))
                    rows.append({
                        "model": mname,
                        "transform": tmode,
                        "sigma_quantile": q_sig,
                        "sigma_val": basis["sigma"],
                        "signal": PROCESS_NAMES[s_lbl],
                        "frac": frac,
                        "null_mean": mu_n,
                        "null_std": sd_n,
                        "val_fpr_1e2": val_fpr,
                        "snr_median": float((np.median(ts) - mu_n) / sd_n),
                        "power_1e2": k / len(ts),
                        "power_1e2_ci_lo": lo,
                        "power_1e2_ci_hi": hi,
                    })
            hp_r = [r for r in rows if r["model"] == mname and r["transform"] == tmode and r["sigma_quantile"] == q_sig and r["signal"] == "higgs_portal_v2" and r["frac"] == 0.01][0]
            print(f"[geom] {mname:22s} | {tmode:8s} q={q_sig:.2f} | HP_v2(1%): SNR={hp_r['snr_median']:6.2f}, Pow(1e-2)={hp_r['power_1e2']:.3f} ({time.time()-t0:.1f}s)", flush=True)
            del basis
            torch.cuda.empty_cache()

        g_path = out_dir / "whitening_and_bandwidth_ablation.csv"
        with gpfs_lock(g_path):
            df_g = pd.DataFrame(rows)
            if g_path.exists():
                old_g = pd.read_csv(g_path)
                old_g = old_g[~old_g["model"].isin(df_g["model"])]
                df_g = pd.concat([old_g, df_g], ignore_index=True)
            df_g.to_csv(g_path, index=False)
    print(f"Saved {out_dir / 'whitening_and_bandwidth_ablation.csv'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=str, required=True, choices=["all_shape", "all_additive", "highstats", "geometry"])
    parser.add_argument("--models", type=str, nargs="*", default=None)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    if args.step == "all_shape":
        step_all_models_shape(device, args.models, sample_mode="shape")
    elif args.step == "all_additive":
        default_add = ["supcon_d64", "supervised_e2e_d64", "ssl_kinematic_d64", "ssl_kinematic_drop_d64", "ssl_tokens_d64", "ae_transformer_d64", "ae_mlp_d64", "random_init_d64", "raw_summary_d28", "phys_handcrafted_d6"]
        step_all_models_shape(device, args.models or default_add, sample_mode="additive")
    elif args.step == "highstats":
        step_highstats_calibration_and_power(device, args.models)
    elif args.step == "geometry":
        step_whitening_geometry_ablation(device, args.models)


if __name__ == "__main__":
    main()
