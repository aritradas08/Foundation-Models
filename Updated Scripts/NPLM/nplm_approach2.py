import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from scipy import stats
import matplotlib.pyplot as plt

embeddings_path = "/global/cfs/cdirs/m4958/usr/aritra08/cont_learn/supervised/results/d64/embeddings/z_embeddings_all_events.npz"
out_dir = "/global/cfs/cdirs/m4958/usr/aritra08/cont_learn/supervised/results/nplm/d64/"

sm_labels = [0, 1, 2]
bsm_labels = [3, 4]
process_names = ["ttbar", "ggf", "dihiggs", "higgs_portal", "hidden_valley"]

n_r = 500_000
n_b = 100_000
injection_fractions = [0.01, 0.025, 0.05]
n_null_toys = 180
n_signal_toys = 60

m_kernels = None  # None means by default sqrt(n_b)
sigma_quantile = 0.90
l2_reg = 1e-6
f_clamp = 30.0
kmm_jitter = 1e-6

lbfgs_max_iter = 1000
lbfgs_history_size = 50
lbfgs_grad_tol = 1e-6
lbfgs_lr = 1.0
lbfgs_tol_change = 1e-12
lbfgs_min_restarts = 3
lbfgs_n_restarts = 6
delta_t_tol = 0.05

std_ratio_tol = 0.15
ks_pvalue_min = 0.01

random_seed = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dtype = torch.float64


def load_pools(path):
    data = np.load(path)
    z, labels = data["z"], data["labels"]

    bkg_mask = np.isin(labels, sm_labels)
    sig_mask = np.isin(labels, bsm_labels)

    z_bkg, labels_bkg = z[bkg_mask], labels[bkg_mask]
    z_sig, labels_sig = z[sig_mask], labels[sig_mask]

    print(f"background pool: {z_bkg.shape[0]} events, dim={z_bkg.shape[1]}")
    for lv in sm_labels:
        print(f" {lv} ({process_names[lv]}): {(labels_bkg == lv).sum()}")
    print(f"signal pool: {z_sig.shape[0]} events")
    for lv in bsm_labels:
        print(f" {lv} ({process_names[lv]}): {(labels_sig == lv).sum()}")

    return z_bkg, labels_bkg, z_sig, labels_sig


def build_reference(z_bkg, n_ref, rng):
    idx = rng.permutation(len(z_bkg))
    idx_ref = idx[:n_ref]
    idx_rest = idx[n_ref:]
    return z_bkg[idx_ref], idx_rest


def sample_null(z_bkg, idx_pool, n_bkg, rng):
    idx = rng.choice(idx_pool, size=n_bkg, replace=False)
    return z_bkg[idx]


def sample_signal(z_bkg, idx_pool, z_sig, n_bkg, n_sig, rng):
    idx_b = rng.choice(idx_pool, size=n_bkg, replace=False)
    idx_s = rng.choice(len(z_sig), size=n_sig, replace=(n_sig > len(z_sig)))
    return np.concatenate([z_bkg[idx_b], z_sig[idx_s]], axis=0)


def choose_centers_and_sigma(ref_scaled, n_centers, quantile, rng):
    n_ref = ref_scaled.shape[0]
    centers = ref_scaled[rng.choice(n_ref, size=n_centers, replace=False)]

    sub = ref_scaled[rng.choice(n_ref, size=min(2000, n_ref), replace=False)]
    with torch.no_grad():
        d = torch.cdist(torch.from_numpy(sub).float(), torch.from_numpy(sub).float())
        sigma = torch.quantile(d[d > 0], quantile).item()
    return centers, sigma


def rbf_features(x, centers, sigma, device, dtype=torch.float32, chunk=20_000):
    centers_t = torch.from_numpy(centers).to(dtype=dtype, device=device)
    x_t = torch.from_numpy(x).to(dtype=dtype)
    out = torch.empty(x.shape[0], centers.shape[0], dtype=dtype)
    for start in range(0, x.shape[0], chunk):
        end = min(start + chunk, x.shape[0])
        xb = x_t[start:end].to(device)
        d2 = torch.cdist(xb, centers_t) ** 2
        out[start:end] = torch.exp(-d2/(2 * sigma ** 2)).cpu()
    return out


def kernel_gram_matrix(centers, sigma, device, dtype=torch.float32, jitter=kmm_jitter):
    k = rbf_features(centers, centers, sigma, device, dtype=dtype).to(device)
    return k + jitter * torch.eye(k.shape[0], device=device, dtype=k.dtype)


class NPLMKernelModel(nn.Module):
    def __init__(self, n_centers):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(n_centers))

    def forward(self, phi):
        return phi @ self.w


def class_imbalance_weight(n_ref, n_bkg):
    return float(n_ref) / float(n_bkg)


def weighted_cross_entropy_loss(f_ref, f_data, a1, a0=1.0):
    f_ref = f_ref.clamp(-f_clamp, f_clamp)
    f_data = f_data.clamp(-f_clamp, f_clamp)
    loss_ref = a0 * F.softplus(f_ref).sum()
    loss_data = a1 * F.softplus(-f_data).sum()
    return (loss_ref + loss_data)/(f_ref.shape[0] + f_data.shape[0])


def rkhs_regularizer(w, k_mm):
    return w @ (k_mm @ w)


def compute_test_statistic(model, phi_ref, phi_data, n_bkg, n_ref, device):
    with torch.no_grad():
        f_ref = model(phi_ref.to(device)).clamp(-f_clamp, f_clamp)
        f_data = model(phi_data.to(device)).clamp(-f_clamp, f_clamp)
        ratio = float(n_bkg) / float(n_ref)
        t_stat = 2.0 * (ratio * (1.0 - torch.exp(f_ref)).sum() + f_data.sum())
    return float(t_stat.item())


def train_kernel_model(phi_ref, phi_data, k_mm, a1, lam, n_bkg, n_ref, device,
                       max_iter=lbfgs_max_iter,
                       history_size=lbfgs_history_size,
                       grad_tol=lbfgs_grad_tol,
                       lr=lbfgs_lr):
    
    phi_ref = phi_ref.to(device)
    phi_data = phi_data.to(device)
    k_mm = k_mm.to(device)

    model = NPLMKernelModel(phi_ref.shape[1]).to(device=device, dtype=phi_ref.dtype)

    def loss_fn():
        return (weighted_cross_entropy_loss(model(phi_ref), model(phi_data), a1) + lam * rkhs_regularizer(model.w, k_mm))

    total_calls = 0
    final_grad_inf = float("nan")
    hit_max_iter = False
    n_restarts_used = 0
    t_history = []
    stopped_reason = "max_restarts"

    for restart_idx in range(lbfgs_n_restarts):
        n_restarts_used = restart_idx + 1
        optimizer = torch.optim.LBFGS(
            model.parameters(), lr=lr, max_iter=max_iter,
            history_size=history_size, tolerance_grad=grad_tol,
            tolerance_change=lbfgs_tol_change,
            line_search_fn="strong_wolfe",
        )
        calls = {"n": 0}

        def closure():
            optimizer.zero_grad()
            loss = loss_fn()
            loss.backward()
            calls["n"] += 1
            return loss

        optimizer.step(closure)
        total_calls += calls["n"]
        hit_max_iter = calls["n"] >= max_iter

        if model.w.grad is not None:
            final_grad_inf = float(model.w.grad.detach().abs().max().item())

        t_history.append(compute_test_statistic(model, phi_ref, phi_data, n_bkg, n_ref, device))

        if n_restarts_used >= lbfgs_min_restarts and len(t_history) >= 2:
            if abs(t_history[-1] - t_history[-2]) < delta_t_tol:
                stopped_reason = "t_stable"
                break

    return model, {
        "t_final": t_history[-1],
        "t_history": t_history,
        "final_grad_norm_inf": final_grad_inf,
        "n_closure_calls": total_calls,
        "n_restarts_used": n_restarts_used,
        "hit_max_iter": bool(hit_max_iter),
        "t_stable": stopped_reason == "t_stable",
        "stopped_reason": stopped_reason,
    }


def check_chi2_goodness_of_fit(t_null, std_tol=std_ratio_tol, ks_min=ks_pvalue_min):
    t_null = np.asarray(t_null)
    mean_t = t_null.mean()
    std_t = t_null.std()
    var_t = t_null.var()

    dof_mean_only = max(mean_t, 1e-3)
    dof_mom = max(2 * mean_t ** 2 / var_t, 1e-3) if var_t > 0 else dof_mean_only

    try:
        dof_mle, loc_mle, scale_mle = stats.chi2.fit(t_null, floc=0)
        dof_mle = max(dof_mle, 1e-3)
    except Exception as e:
        print(f"  chi2 fit failed ({e})")
        dof_mle, loc_mle, scale_mle = dof_mom, 0.0, 1.0

    dof_hat = dof_mle
    std_expected = np.sqrt(2 * dof_hat) * scale_mle
    std_ratio = std_t / std_expected if std_expected > 0 else np.nan

    ks_stat, ks_pvalue = stats.kstest(t_null, "chi2", args=(dof_hat, loc_mle, scale_mle))
    chi2_ok = (abs(std_ratio - 1.0) <= std_tol) and (ks_pvalue > ks_min)

    print("\nchi2 goodness of fit")
    print(f" mean={mean_t:.2f}, std={std_t:.2f}")
    print(f" dof (mle)={dof_mle:.2f}, scale={scale_mle:.3f}")
    print(f" std_ratio={std_ratio:.2f}")
    print(f" ks stat={ks_stat:.4f}, p={ks_pvalue:.2e}")
    print("  chi2 ok" if chi2_ok else "chi2 not ok")

    return {
        "mean": float(mean_t), "std": float(std_t),
        "dof_mean_only": float(dof_mean_only), "dof_mom": float(dof_mom),
        "dof_mle": float(dof_mle), "scale_mle": float(scale_mle), "loc_mle": float(loc_mle),
        "dof_hat": float(dof_hat), "std_ratio": float(std_ratio),
        "ks_statistic": float(ks_stat), "ks_pvalue": float(ks_pvalue), "chi2_ok": bool(chi2_ok),
    }


def check_bimodality(t_null, n_bins=10):
    t_null = np.asarray(t_null)
    counts, edges = np.histogram(t_null, bins=n_bins)
    first_bin_frac = counts[0] / len(t_null)
    suspicious = bool(first_bin_frac > 0.25 and counts[0] > 0 and counts[1:].max() > 0
                      and counts[0] > 2 * np.median(counts[counts > 0]))
    print(f"{'Suspicious' if suspicious else 'Not Suspicious'}")
    return {"first_bin_frac": float(first_bin_frac), "first_bin_edge": float(edges[1]),
            "suspicious": suspicious}


def z_score_from_null(t_obs, t_null, dof_hat, scale_hat=1.0, loc_hat=0.0):
    t_null = np.asarray(t_null)
    n_toys = len(t_null)

    p_empirical = max((t_null >= t_obs).sum() / n_toys, 1.0 / (n_toys + 1))
    z_empirical = stats.norm.isf(p_empirical)

    log_p = stats.chi2.logsf(t_obs, df=dof_hat, loc=loc_hat, scale=scale_hat)
    if np.isneginf(log_p):
        z_asymptotic = stats.norm.isf(1e-300)
    else:
        z_asymptotic = stats.norm.isf(np.exp(log_p))

    return float(z_empirical), float(z_asymptotic)


def run_nplm_experiment():
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(random_seed)

    z_bkg, labels_bkg, z_sig, labels_sig = load_pools(embeddings_path)

    ref_raw, idx_pool = build_reference(z_bkg, n_r, rng)
    scaler = StandardScaler().fit(ref_raw)
    ref = scaler.transform(ref_raw).astype(np.float32)

    n_centers = m_kernels or int(np.sqrt(n_b))
    print(f"M={n_centers}, n_r={n_r}, n_b={n_b}, dtype={train_dtype}")

    centers, sigma = choose_centers_and_sigma(ref, n_centers, sigma_quantile, rng)
    print(f"sigma = {sigma:.3f}")

    phi_ref = rbf_features(ref, centers, sigma, device, dtype=train_dtype)
    k_mm = kernel_gram_matrix(centers, sigma, device, dtype=train_dtype)
    a1 = class_imbalance_weight(n_r, n_b)
    print(f"a1/a0 = {a1:.4f}")

    print(f"\n{n_null_toys} background-only toys")
    t_null = []
    convergence_log = []
    for i in range(n_null_toys):
        d_null = scaler.transform(sample_null(z_bkg, idx_pool, n_b, rng)).astype(np.float32)
        phi_data = rbf_features(d_null, centers, sigma, device, dtype=train_dtype)

        model, conv = train_kernel_model(phi_ref, phi_data, k_mm, a1, l2_reg, n_b, n_r, device)
        t_null.append(conv["t_final"])
        convergence_log.append(conv)

        if (i + 1) % 20 == 0:
            print(f" toy {i + 1}/{n_null_toys}: t={conv['t_final']:.2f}"
                  f"stable={conv['t_stable']}  restarts={conv['n_restarts_used']}"
                  f"calls={conv['n_closure_calls']}  grad_inf={conv['final_grad_norm_inf']:.2e}")

    t_null = np.array(t_null)
    n_t_stable = sum(c["t_stable"] for c in convergence_log)
    mean_calls = float(np.mean([c["n_closure_calls"] for c in convergence_log]))
    mean_restarts = float(np.mean([c["n_restarts_used"] for c in convergence_log]))

    bimod = check_bimodality(t_null)
    gof = check_chi2_goodness_of_fit(t_null)
    dof_hat, scale_hat, loc_hat = gof["dof_hat"], gof["scale_mle"], gof["loc_mle"]

    results = {
        "embeddings_path": embeddings_path,
        "null_t": t_null.tolist(),
        "null_chi2_gof": gof,
        "null_bimodality": bimod,
        "null_convergence_summary": {
            "n_toys": n_null_toys,
            "n_t_stable": n_t_stable,
            "mean_closure_calls": mean_calls,
            "mean_restarts": mean_restarts,
        },
        "injections": {},
    }

    for frac in injection_fractions:
        n_s = max(int(round(frac * n_b)), 1)
        print(f"\ninjection {frac * 100:.2f}% ,{n_signal_toys} toys")
        t_sig = []
        for _ in range(n_signal_toys):
            d_sig = scaler.transform(
                sample_signal(z_bkg, idx_pool, z_sig, n_b, n_s, rng)).astype(np.float32)
            phi_data = rbf_features(d_sig, centers, sigma, device, dtype=train_dtype)

            model, conv = train_kernel_model(phi_ref, phi_data, k_mm, a1, l2_reg, n_b, n_r, device)
            t_sig.append(conv["t_final"])
        t_sig = np.array(t_sig)

        t_median = np.median(t_sig)
        z_emp, z_asym = z_score_from_null(t_median, t_null, dof_hat, scale_hat, loc_hat)

        print(f" median t={t_median:.2f}; z_emp={z_emp:.2f}; z_asym={z_asym:.2f}{note}")

        results["injections"][f"{frac}"] = {
            "n_signal": n_s,
            "t_values": t_sig.tolist(),
            "t_median": float(t_median),
            "z_empirical": z_emp,
            "z_asymptotic": z_asym,
            "z_asymptotic_trustworthy": bool(gof["chi2_ok"]),
        }

    with open(os.path.join(out_dir, "nplm_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(t_null, bins=30, density=True, alpha=0.5, label="background-only (null)", color="purple")
    for frac in injection_fractions:
        t_sig = np.array(results["injections"][f"{frac}"]["t_values"])
        ax.hist(t_sig, bins=30, density=True, alpha=0.5, label=f"{frac * 100:.1f}% signal")
    ax.set_xlabel(f"NPLM test statistic t")
    ax.set_ylabel("probability density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "nplm_t_distributions.png"), dpi=150)
    plt.close(fig)

    return results


if __name__ == "__main__":
    run_nplm_experiment()
