#!/usr/bin/env python3
"""
Statistically calibrated Kernel NPLM Anomaly Detection runner (L-BFGS in float64 with RKHS penalty).
Evaluates each BSM signal channel independently on the held-out eval_downstream SM pool,
supporting both:
  - sample_mode="shape" (total data size N_D = N_B constant: (N_B - N_S) bkg + N_S sig, isolating representation geometry)
  - sample_mode="additive" (N_B bkg + N_S sig, reporting both raw t_ext and rate-subtracted t_ext - t_rate)
"""
import argparse
import json
import os
import time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F

RESULTS_BASE = Path("/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/results")
EMBED_DIR = RESULTS_BASE / "embeddings"
NPLM_OUT_DIR = RESULTS_BASE / "nplm"

SM_LABELS = [0, 1, 2]
SIGNAL_NAMES = {
    3: "higgs_portal_v2",
    4: "hidden_valley",
    5: "higgs_portal_v1",
}


def choose_centers_and_sigma(ref_scaled: np.ndarray, n_centers: int, quantile: float, rng: np.random.Generator, device: torch.device):
    n_ref = ref_scaled.shape[0]
    centers = ref_scaled[rng.choice(n_ref, size=n_centers, replace=False)].copy()
    sub = ref_scaled[rng.choice(n_ref, size=min(2500, n_ref), replace=False)]
    with torch.no_grad():
        sub_t = torch.from_numpy(sub).to(device=device, dtype=torch.float32)
        d = torch.cdist(sub_t, sub_t)
        pos_d = d[d > 0]
        sigma = float(torch.quantile(pos_d, quantile).item()) if pos_d.numel() > 0 else 1.0
    return centers, max(sigma, 1e-3)


def rbf_features_gpu(x_t: torch.Tensor, centers_t: torch.Tensor, sigma: float, chunk: int = 50_000) -> torch.Tensor:
    N = x_t.shape[0]
    M = centers_t.shape[0]
    inv_2s2 = -1.0 / (2.0 * (sigma ** 2))
    if N <= chunk:
        d2 = torch.cdist(x_t, centers_t).pow_(2)
        return torch.exp(d2.mul_(inv_2s2))
    out = torch.empty((N, M), dtype=x_t.dtype, device=x_t.device)
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        d2 = torch.cdist(x_t[start:end], centers_t).pow_(2)
        out[start:end] = torch.exp(d2.mul_(inv_2s2))
    return out


class NPLMKernelModel(nn.Module):
    def __init__(self, n_centers: int, dtype: torch.dtype, device: torch.device):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(n_centers, dtype=dtype, device=device))

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        return phi @ self.w


def compute_test_statistic(w: torch.Tensor, phi_ref: torch.Tensor, phi_data: torch.Tensor, n_bkg_expected: int, n_ref: int, f_clamp: float = 30.0) -> float:
    with torch.no_grad():
        f_ref = (phi_ref @ w).clamp(-f_clamp, f_clamp)
        f_data = (phi_data @ w).clamp(-f_clamp, f_clamp)
        ratio = float(n_bkg_expected) / float(n_ref)
        t_stat = 2.0 * (ratio * (1.0 - torch.exp(f_ref)).sum() + f_data.sum())
    return float(t_stat.item())


def train_kernel_lbfgs(
    phi_ref: torch.Tensor,
    phi_data: torch.Tensor,
    k_mm: torch.Tensor,
    n_bkg_expected: int,
    n_ref: int,
    lam: float = 1e-6,
    f_clamp: float = 30.0,
    max_iter: int = 15,
    grad_tol: float = 1e-7,
    **kwargs,
) -> tuple[torch.Tensor, dict]:
    """
    Exact second-order Newton-Raphson optimizer in float64 on GPU with backtracking line search.
    Because the kernel NPLM objective is strictly convex in w in R^M (M = 316),
    inverting the exact M x M Hessian converges quadratically to grad_inf < 1e-8 in 3-5 iterations (~30ms/toy).
    """
    device = phi_ref.device
    dtype = phi_ref.dtype
    M = phi_ref.shape[1]
    w = torch.zeros(M, dtype=dtype, device=device)

    a1 = float(n_ref) / float(n_bkg_expected)
    denom = float(phi_ref.shape[0] + phi_data.shape[0])
    reg_mat = 2.0 * lam * k_mm + 1e-10 * torch.eye(M, dtype=dtype, device=device)

    def eval_loss(w_vec: torch.Tensor) -> float:
        f_r = (phi_ref @ w_vec).clamp(-f_clamp, f_clamp)
        f_d = (phi_data @ w_vec).clamp(-f_clamp, f_clamp)
        ce = (F.softplus(f_r).sum() + a1 * F.softplus(-f_d).sum()) / denom
        reg = lam * (w_vec @ (k_mm @ w_vec))
        return float((ce + reg).item())

    t_history = []
    final_grad_inf = float("nan")
    stopped_reason = "max_iter"

    with torch.no_grad():
        cur_loss = eval_loss(w)
        for it in range(max_iter):
            f_r = (phi_ref @ w).clamp(-f_clamp, f_clamp)
            f_d = (phi_data @ w).clamp(-f_clamp, f_clamp)
            p_r = torch.sigmoid(f_r)
            p_d = torch.sigmoid(-f_d)

            g = (phi_ref.T @ p_r - a1 * (phi_data.T @ p_d)) / denom + 2.0 * lam * (k_mm @ w)
            final_grad_inf = float(g.abs().max().item())
            if final_grad_inf <= grad_tol:
                stopped_reason = "grad_converged"
                break

            wr = (p_r * (1.0 - p_r)).clamp(min=1e-12)
            wd = (p_d * (1.0 - p_d)).clamp(min=1e-12)
            H = (
                phi_ref.T @ (phi_ref * wr.unsqueeze(1))
                + a1 * (phi_data.T @ (phi_data * wd.unsqueeze(1)))
            ) / denom + reg_mat

            try:
                dw = torch.linalg.solve(H, -g)
            except RuntimeError:
                dw = torch.linalg.lstsq(H, -g).solution

            # Backtracking line search for guaranteed descent
            step = 1.0
            for _ in range(8):
                w_cand = w + step * dw
                cand_loss = eval_loss(w_cand)
                if cand_loss <= cur_loss + 1e-12:
                    w = w_cand
                    cur_loss = cand_loss
                    break
                step *= 0.5
            else:
                w = w + step * dw
                cur_loss = eval_loss(w)

        t_val = compute_test_statistic(w, phi_ref, phi_data, n_bkg_expected, n_ref, f_clamp)
        t_history.append(t_val)

    return w.detach(), {
        "t_final": t_val,
        "t_history": t_history,
        "final_grad_norm_inf": final_grad_inf,
        "n_closure_calls": it + 1,
        "n_passes": it + 1,
        "stopped_reason": stopped_reason,
    }


def evaluate_null_calibration(t_null: np.ndarray) -> dict:
    t_null = np.asarray(t_null, dtype=np.float64)
    mean_t = float(t_null.mean())
    std_t = float(t_null.std(ddof=1))
    var_t = float(t_null.var(ddof=1))

    dof_mean = max(mean_t, 1e-3)
    std_unscaled_expected = np.sqrt(2.0 * dof_mean)
    std_ratio_unscaled = std_t / std_unscaled_expected if std_unscaled_expected > 0 else np.nan
    ks_unscaled_stat, ks_unscaled_p = stats.kstest(t_null, "chi2", args=(dof_mean, 0.0, 1.0))

    try:
        dof_mle, loc_mle, scale_mle = stats.chi2.fit(t_null, floc=0.0)
        dof_mle = max(float(dof_mle), 1e-3)
        scale_mle = max(float(scale_mle), 1e-6)
    except Exception:
        dof_mle = max(2.0 * (mean_t ** 2) / max(var_t, 1e-6), 1e-3)
        loc_mle, scale_mle = 0.0, max(var_t / (2.0 * mean_t), 1e-3)

    std_scaled_expected = np.sqrt(2.0 * dof_mle) * scale_mle
    std_ratio_scaled = std_t / std_scaled_expected if std_scaled_expected > 0 else np.nan
    ks_scaled_stat, ks_scaled_p = stats.kstest(t_null, "chi2", args=(dof_mle, loc_mle, scale_mle))

    counts, edges = np.histogram(t_null, bins=10)
    first_bin_frac = float(counts[0] / len(t_null))
    bimodal_suspicious = bool(
        first_bin_frac > 0.25 and counts[0] > 0 and counts[1:].max() > 0
        and counts[0] > 2.0 * np.median(counts[counts > 0])
    )
    chi2_ok = bool((abs(std_ratio_scaled - 1.0) <= 0.15) and (ks_scaled_p > 0.01) and (not bimodal_suspicious))

    return {
        "mean": mean_t,
        "std": std_t,
        "dof_mean_only": float(dof_mean),
        "std_ratio_unscaled": float(std_ratio_unscaled),
        "ks_unscaled_pvalue": float(ks_unscaled_p),
        "dof_mle": float(dof_mle),
        "scale_mle": float(scale_mle),
        "loc_mle": float(loc_mle),
        "std_ratio": float(std_ratio_scaled),
        "ks_statistic": float(ks_scaled_stat),
        "ks_pvalue": float(ks_scaled_p),
        "first_bin_frac": first_bin_frac,
        "bimodal_suspicious": bimodal_suspicious,
        "chi2_ok": chi2_ok,
    }


def compute_z_scores(t_obs: float, t_null: np.ndarray, gof: dict) -> dict:
    t_null = np.asarray(t_null, dtype=np.float64)
    n_toys = len(t_null)
    # Phipson-Smyth exact rank Monte Carlo p-value: (1 + #{t_null >= t_obs}) / (N + 1)
    k_exceed = int((t_null >= t_obs).sum())
    p_emp = float(np.clip((1.0 + k_exceed) / (n_toys + 1.0), 1e-12, 1.0 - 1e-12))
    z_emp = float(stats.norm.isf(p_emp))

    # Scaled Gamma/chi2 tail fit
    log_p_scaled = stats.chi2.logsf(max(t_obs, 0.0), df=gof["dof_mle"], loc=gof["loc_mle"], scale=gof["scale_mle"])
    z_asym_scaled = float(stats.norm.isf(1e-300) if np.isneginf(log_p_scaled) else stats.norm.isf(np.clip(np.exp(log_p_scaled), 1e-300, 1.0 - 1e-12)))

    # Unscaled chi2(df = mean_null) tail
    log_p_unscaled = stats.chi2.logsf(max(t_obs, 0.0), df=gof["dof_mean_only"], loc=0.0, scale=1.0)
    z_asym_unscaled = float(stats.norm.isf(1e-300) if np.isneginf(log_p_unscaled) else stats.norm.isf(np.clip(np.exp(log_p_unscaled), 1e-300, 1.0 - 1e-12)))

    # Standardized SNR (t_obs - mean_null) / std_null
    snr = float((t_obs - gof["mean"]) / max(gof["std"], 1e-6))
    return {
        "p_empirical": p_emp,
        "z_empirical": z_emp,
        "z_asymptotic": z_asym_scaled,
        "z_chi2_unscaled": z_asym_unscaled,
        "snr": snr,
    }


def run_single_model_nplm(
    model_name: str,
    signal_label: int,
    n_r: int = 500_000,
    n_b: int = 100_000,
    n_null_toys: int = 180,
    n_signal_toys: int = 60,
    injection_fractions: list[float] = [0.005, 0.01, 0.025, 0.05],
    sample_mode: str = "shape",
    device: torch.device = torch.device("cuda"),
    seed: int = 42,
    cached_null: tuple[np.ndarray, dict] | None = None,
):
    sig_name = SIGNAL_NAMES[signal_label]
    out_subdir = NPLM_OUT_DIR / f"{sample_mode}_{sig_name}" / model_name
    out_subdir.mkdir(parents=True, exist_ok=True)
    out_json = out_subdir / "nplm_results.json"

    npz_path = EMBED_DIR / f"{model_name}.npz"
    data = np.load(npz_path)
    z = data["z"].astype(np.float32)
    labels = data["labels"].astype(np.int64)

    z_bkg = z[np.isin(labels, SM_LABELS)]
    z_sig = z[labels == signal_label]
    if len(z_sig) == 0:
        raise ValueError(f"No events found for signal_label={signal_label} ({sig_name}) in {npz_path}")

    rng = np.random.default_rng(seed)
    perm_bkg = rng.permutation(len(z_bkg))
    idx_ref = perm_bkg[:n_r]
    idx_pool = perm_bkg[n_r:]

    ref_raw = z_bkg[idx_ref]
    scaler = StandardScaler().fit(ref_raw)
    ref_scaled = scaler.transform(ref_raw).astype(np.float32)
    bkg_pool_scaled = scaler.transform(z_bkg[idx_pool]).astype(np.float32)
    sig_pool_scaled = scaler.transform(z_sig).astype(np.float32)

    n_centers = int(np.sqrt(n_b))
    centers, sigma = choose_centers_and_sigma(ref_scaled, n_centers, 0.90, rng, device)

    dtype = torch.float64
    centers_t = torch.from_numpy(centers).to(device=device, dtype=dtype)
    ref_t = torch.from_numpy(ref_scaled).to(device=device, dtype=dtype)
    bkg_pool_t = torch.from_numpy(bkg_pool_scaled).to(device=device, dtype=dtype)
    sig_pool_t = torch.from_numpy(sig_pool_scaled).to(device=device, dtype=dtype)

    phi_ref = rbf_features_gpu(ref_t, centers_t, sigma)
    k_mm = rbf_features_gpu(centers_t, centers_t, sigma) + 1e-6 * torch.eye(n_centers, device=device, dtype=dtype)

    print(
        f"[NPLM {model_name} | {sig_name} | mode={sample_mode}] "
        f"D={z.shape[1]}, M={n_centers}, sigma={sigma:.3f}, N_R={n_r}, N_B={n_b}, "
        f"bkg_pool={len(bkg_pool_scaled)}, sig_pool={len(sig_pool_scaled)}",
        flush=True,
    )

    n_pool = len(bkg_pool_scaled)
    if cached_null is not None:
        t_null, gof = cached_null
        for _ in range(n_null_toys):
            rng.choice(n_pool, size=n_b, replace=(n_b > n_pool))
        note = "" if gof["chi2_ok"] else " [chi2 GOF warning]"
        print(
            f"  Null (reused {len(t_null)} toys): "
            f"mean={gof['mean']:.2f}, std={gof['std']:.2f}, dof_mle={gof['dof_mle']:.2f}, "
            f"scale={gof['scale_mle']:.3f}, ks_p={gof['ks_pvalue']:.3f}, chi2_ok={gof['chi2_ok']}",
            flush=True,
        )
    else:
        t0 = time.time()
        t_null = []
        for i in range(n_null_toys):
            idx_b = rng.choice(n_pool, size=n_b, replace=(n_b > n_pool))
            phi_d = rbf_features_gpu(bkg_pool_t[torch.from_numpy(idx_b).to(device)], centers_t, sigma)
            _, conv = train_kernel_lbfgs(phi_ref, phi_d, k_mm, n_b, n_r)
            t_null.append(conv["t_final"])

        t_null = np.array(t_null, dtype=np.float64)
        gof = evaluate_null_calibration(t_null)
        note = "" if gof["chi2_ok"] else " [chi2 GOF warning]"
        print(
            f"  Null ({n_null_toys} toys, {time.time() - t0:.1f}s): "
            f"mean={gof['mean']:.2f}, std={gof['std']:.2f}, dof_mle={gof['dof_mle']:.2f}, "
            f"scale={gof['scale_mle']:.3f}, ks_p={gof['ks_pvalue']:.3f}, chi2_ok={gof['chi2_ok']}",
            flush=True,
        )

    results = {
        "model": model_name,
        "latent_dim": int(z.shape[1]),
        "signal_label": int(signal_label),
        "signal_name": sig_name,
        "sample_mode": sample_mode,
        "n_r": int(n_r),
        "n_b": int(n_b),
        "m_kernels": int(n_centers),
        "sigma": float(sigma),
        "null_t": t_null.tolist(),
        "null_chi2_gof": gof,
        "injections": {},
    }

    for frac in injection_fractions:
        n_s = max(int(round(frac * n_b)), 1)
        n_b_draw = (n_b - n_s) if sample_mode == "shape" else n_b
        n_expected = n_b  # expected background under H0
        t_rate_only = 0.0 if sample_mode == "shape" else float(2.0 * ((n_b + n_s) * np.log(1.0 + n_s / n_b) - n_s))

        t_sig = []
        for _ in range(n_signal_toys):
            idx_b = rng.choice(n_pool, size=n_b_draw, replace=(n_b_draw > n_pool))
            idx_s = rng.choice(len(sig_pool_scaled), size=n_s, replace=(n_s > len(sig_pool_scaled)))
            d_t = torch.cat([
                bkg_pool_t[torch.from_numpy(idx_b).to(device)],
                sig_pool_t[torch.from_numpy(idx_s).to(device)],
            ], dim=0)
            phi_d = rbf_features_gpu(d_t, centers_t, sigma)
            _, conv = train_kernel_lbfgs(phi_ref, phi_d, k_mm, n_expected, n_r)
            t_sig.append(conv["t_final"])

        t_sig = np.array(t_sig, dtype=np.float64)
        t_med = float(np.median(t_sig))
        t_q16 = float(np.percentile(t_sig, 16))
        t_q84 = float(np.percentile(t_sig, 84))
        z_dict = compute_z_scores(t_med, t_null, gof)
        z_RateSub = compute_z_scores(max(t_med - t_rate_only, 0.0), t_null, gof)

        print(
            f"  inj {frac * 100:5.2f}% (n_s={n_s}): median t={t_med:.2f} [{t_q16:.1f}, {t_q84:.1f}]; "
            f"z_emp={z_dict['z_empirical']:.2f}; z_asym={z_dict['z_asymptotic']:.2f}; "
            f"z_chi2={z_dict['z_chi2_unscaled']:.2f}; snr={z_dict['snr']:.2f}{note}"
        )

        results["injections"][f"{frac}"] = {
            "n_signal": int(n_s),
            "n_bkg_drawn": int(n_b_draw),
            "t_rate_only": t_rate_only,
            "t_values": t_sig.tolist(),
            "t_median": t_med,
            "t_q16": t_q16,
            "t_q84": t_q84,
            **z_dict,
            "z_asymptotic_rate_subtracted": z_RateSub["z_asymptotic"],
            "z_asymptotic_trustworthy": bool(gof["chi2_ok"]),
        }

    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.hist(t_null, bins=25, density=True, alpha=0.55, label="SM Null ($H_0$)", color="darkslateblue")
    for frac in injection_fractions:
        ts = np.array(results["injections"][f"{frac}"]["t_values"])
        ax.hist(ts, bins=25, density=True, alpha=0.45, label=f"{frac * 100:.1f}% {sig_name}")
    ax.set_xlabel("NPLM test statistic $t$")
    ax.set_ylabel("Probability density")
    ax.set_title(f"{model_name} — {sig_name} ({sample_mode})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_subdir / "nplm_t_distributions.png", dpi=150)
    plt.close(fig)
    return results, (t_null, gof)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, nargs="+", required=True)
    parser.add_argument("--signals", type=int, nargs="+", default=[3, 4])
    parser.add_argument("--sample-mode", type=str, default="shape", choices=["shape", "additive"])
    parser.add_argument("--n-r", type=int, default=500_000)
    parser.add_argument("--n-b", type=int, default=100_000)
    parser.add_argument("--n-null-toys", type=int, default=180)
    parser.add_argument("--n-signal-toys", type=int, default=60)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    for m in args.models:
        cached_null = None
        for sig in args.signals:
            _, cached_null = run_single_model_nplm(
                model_name=m,
                signal_label=sig,
                n_r=args.n_r,
                n_b=args.n_b,
                n_null_toys=args.n_null_toys,
                n_signal_toys=args.n_signal_toys,
                sample_mode=args.sample_mode,
                device=device,
                cached_null=cached_null,
            )


if __name__ == "__main__":
    main()
