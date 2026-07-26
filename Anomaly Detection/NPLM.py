import os
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from scipy import stats
import matplotlib.pyplot as plt

EMBEDDINGS_PATH = "/global/cfs/cdirs/m4958/usr/aritra08/autoencoders/transformer/results/sm_only_v3/embeddings/z_embeddings_all_events.npz"
OUT_DIR = "/global/cfs/cdirs/m4958/usr/aritra08/autoencoders/transformer/results_nplm/sm_only_v3/"

SM_LABELS = [0, 1, 2]          # background events
BSM_LABELS = [3, 4]            # signal events
ALL_PROCESS_NAMES = ["ttbar", "ggf", "dihiggs", "higgs_portal", "hidden_valley"]

N_R = 500_000         
N_B = 100_000           
INJECTION_FRACTIONS = [0.001,0.005,0.01, 0.025, 0.05]   
N_NULL_TOYS = 5000      
N_SIGNAL_TOYS = 500     

M_KERNELS = None       
SIGMA_QUANTILE = 0.90  
L2_REG = 1e-6
LR = 5e-3
N_EPOCHS = 400
F_CLAMP = 15.0         

STD_RATIO_TOL = 0.15    
KS_PVALUE_MIN = 0.01    

RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_pools(path):
    data = np.load(path)
    z, labels = data["z"], data["labels"]

    bkg_mask = np.isin(labels, SM_LABELS)
    sig_mask = np.isin(labels, BSM_LABELS)

    z_bkg, labels_bkg = z[bkg_mask], labels[bkg_mask]
    z_sig, labels_sig = z[sig_mask], labels[sig_mask]

    print(f"Background pool: {z_bkg.shape[0]} events, dim={z_bkg.shape[1]}")
    for lv in SM_LABELS:
        print(f"  label={lv} ({ALL_PROCESS_NAMES[lv]}): {(labels_bkg==lv).sum()}")
    print(f"Signal pool: {z_sig.shape[0]} events")
    for lv in BSM_LABELS:
        print(f"  label={lv} ({ALL_PROCESS_NAMES[lv]}): {(labels_sig==lv).sum()}")

    return z_bkg, labels_bkg, z_sig, labels_sig



def build_reference(z_bkg, n_r, rng):
    idx = rng.permutation(len(z_bkg))
    idx_R = idx[:n_r]
    idx_rest = idx[n_r:]
    assert len(idx_rest) > 0
    return z_bkg[idx_R], idx_rest


def sample_D_null(z_bkg, idx_pool, n_b, rng):
    idx = rng.choice(idx_pool, size=n_b, replace=False)
    return z_bkg[idx]


def sample_D_signal(z_bkg, idx_pool, z_sig, n_b, n_s, rng):
    idx_b = rng.choice(idx_pool, size=n_b, replace=False)
    idx_s = rng.choice(len(z_sig), size=n_s, replace=(n_s > len(z_sig)))
    D = np.concatenate([z_bkg[idx_b], z_sig[idx_s]], axis=0)
    return D


def choose_centers_and_sigma(R_scaled, m_kernels, quantile, rng):
    n_r = R_scaled.shape[0]
    center_idx = rng.choice(n_r, size=m_kernels, replace=False)
    centers = R_scaled[center_idx]

    sub_idx = rng.choice(n_r, size=min(2000, n_r), replace=False)
    sub = R_scaled[sub_idx]
    with torch.no_grad():
        d = torch.cdist(torch.from_numpy(sub).float(), torch.from_numpy(sub).float())
        sigma = torch.quantile(d[d > 0], quantile).item()
    return centers, sigma


def rbf_features(X, centers, sigma, device, chunk=20_000):
    centers_t = torch.from_numpy(centers).float().to(device)
    X_t = torch.from_numpy(X).float()
    out = torch.empty(X.shape[0], centers.shape[0], dtype=torch.float32)
    for start in range(0, X.shape[0], chunk):
        end = min(start + chunk, X.shape[0])
        xb = X_t[start:end].to(device)
        d2 = torch.cdist(xb, centers_t) ** 2
        out[start:end] = torch.exp(-d2 / (2 * sigma ** 2)).cpu()
    return out


class NPLMKernelModel(nn.Module):
    def __init__(self, m_kernels):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(m_kernels))

    def forward(self, phi):
        return phi @ self.w

def train_and_get_t(phi_R, phi_D, n_b, w_r, lam, lr, n_epochs, device):
    phi_R = phi_R.to(device)
    phi_D = phi_D.to(device)

    model = NPLMKernelModel(phi_R.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(n_epochs):
        optimizer.zero_grad()

        f_R = model(phi_R).clamp(-F_CLAMP, F_CLAMP)
        f_D = model(phi_D).clamp(-F_CLAMP, F_CLAMP)

        n_hw = w_r * torch.exp(f_R).sum()
        objective = n_b - n_hw + f_D.sum()          
        reg = lam * (model.w ** 2).sum()

        loss = -(objective) + reg                   
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        f_R = model(phi_R).clamp(-F_CLAMP, F_CLAMP)
        f_D = model(phi_D).clamp(-F_CLAMP, F_CLAMP)
        n_hw = w_r * torch.exp(f_R).sum()
        t_stat = 2.0 * (n_b - n_hw + f_D.sum()).item()

    return t_stat, model

#Statistical Checks

def check_chi2_goodness_of_fit(t_null, std_ratio_tol=STD_RATIO_TOL,
                                ks_pvalue_min=KS_PVALUE_MIN):

    t_null = np.asarray(t_null)
    mean_t = t_null.mean()
    std_t = t_null.std()

    dof_mean_only = max(mean_t, 1e-3)

    var_t = t_null.var()
    dof_mom = max(2 * mean_t ** 2 / var_t, 1e-3) if var_t > 0 else dof_mean_only

    try:
        dof_mle, loc_mle, scale_mle = stats.chi2.fit(t_null, floc=0)
        dof_mle = max(dof_mle, 1e-3)
    except Exception as e:
        dof_mle, loc_mle, scale_mle = dof_mom, 0.0, 1.0

    dof_hat = dof_mle  

    std_expected = np.sqrt(2 * dof_hat) * scale_mle
    std_ratio = std_t / std_expected if std_expected > 0 else np.nan

    ks_stat, ks_pvalue = stats.kstest(t_null, "chi2", args=(dof_hat, loc_mle, scale_mle))

    chi2_ok = (abs(std_ratio - 1.0) <= std_ratio_tol) and (ks_pvalue > ks_pvalue_min)

    print(f"\nchi^2 goodness-of-fit check")
    print(f"observed  mean={mean_t:.2f}, std={std_t:.2f}")
    print(f"dof estimate (mean only, OLD) = {dof_mean_only:.2f}")
    print(f"dof estimate (method-of-moments) = {dof_mom:.2f}")
    print(f"dof estimate (full MLE fit) = {dof_mle:.2f}  (scale={scale_mle:.3f})")
    print(f"  std_ratio (observed/predicted, using MLE fit) = {std_ratio:.2f}  ")
    print(f"  KS test vs fitted chi^2 (dof={dof_hat:.1f}, scale={scale_mle:.3f}): "
          f"statistic={ks_stat:.4f}, p-value={ks_pvalue:.2e}")
    

    return {
        "mean": float(mean_t),
        "std": float(std_t),
        "dof_mean_only": float(dof_mean_only),
        "dof_mom": float(dof_mom),
        "dof_mle": float(dof_mle),
        "scale_mle": float(scale_mle),
        "loc_mle": float(loc_mle),
        "dof_hat": float(dof_hat),           # downstream code uses this
        "std_expected_chi2": float(std_expected),
        "std_ratio": float(std_ratio),
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_pvalue),
        "chi2_ok": bool(chi2_ok),
    }


def z_score_from_null(t_obs, t_null, dof_hat, scale_hat=1.0, loc_hat=0.0, use_asymptotic=True):
    t_null = np.asarray(t_null)
    n_toys = len(t_null)

    p_empirical = max((t_null >= t_obs).sum()/n_toys, 1.0/(n_toys + 1))
    z_empirical = stats.norm.isf(p_empirical)

    z_asymptotic = None
    if use_asymptotic:
        log_p_asymp = stats.chi2.logsf(t_obs, df=dof_hat, loc=loc_hat, scale=scale_hat)
        if np.isneginf(log_p_asymp):
            z_asymptotic = stats.norm.isf(1e-300)
        else:
            p_asymp = np.exp(log_p_asymp)
            z_asymptotic = stats.norm.isf(p_asymp)

    return z_empirical, z_asymptotic

def run_nplm_experiment():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    z_bkg, labels_bkg, z_sig, labels_sig = load_pools(EMBEDDINGS_PATH)

    # Build R once, fit standardization on R only
    R_raw, idx_pool = build_reference(z_bkg, N_R, rng)
    scaler = StandardScaler().fit(R_raw)
    R = scaler.transform(R_raw).astype(np.float32)

    m_kernels = M_KERNELS or int(np.sqrt(N_B))
    print(f"Using M={m_kernels} kernel centers, N_R={N_R}, N_B={N_B}")

    centers, sigma = choose_centers_and_sigma(R, m_kernels, SIGMA_QUANTILE, rng)
    print(f"Kernel width sigma = {sigma:.3f}")

    phi_R = rbf_features(R, centers, sigma, DEVICE)
    w_r = N_B/N_R

    # Null distribution: pure-background pseudoexperiments
    print(f"\nRunning {N_NULL_TOYS} background-only pseudoexperiments for the null distribution")
    t_null = []
    for i in range(N_NULL_TOYS):
        D_null_raw = sample_D_null(z_bkg, idx_pool, N_B, rng)
        D_null = scaler.transform(D_null_raw).astype(np.float32)
        phi_D = rbf_features(D_null, centers, sigma, DEVICE)
        t, _ = train_and_get_t(phi_R, phi_D, N_B, w_r, L2_REG, LR, N_EPOCHS, DEVICE)
        t_null.append(t)
        if (i + 1) % 20 == 0:
            print(f"  toy {i+1}/{N_NULL_TOYS}: t = {t:.2f}")
    t_null = np.array(t_null)
    print(f"Null t-statistic: mean={t_null.mean():.2f}, std={t_null.std():.2f}")

    gof = check_chi2_goodness_of_fit(t_null)
    dof_hat = gof["dof_hat"]
    scale_hat = gof["scale_mle"]
    loc_hat = gof["loc_mle"]

    results = {"null_t": t_null.tolist(), "null_chi2_gof": gof, "injections": {}}
    for frac in INJECTION_FRACTIONS:
        n_s = max(int(round(frac * N_B)), 1)
        print(f"\nInjection fraction {frac*100:.2f}% -> N(S)={n_s}, running {N_SIGNAL_TOYS} experiments")
     
        t_sig = []
        for i in range(N_SIGNAL_TOYS):
            D_sig_raw = sample_D_signal(z_bkg, idx_pool, z_sig, N_B, n_s, rng)
            D_sig = scaler.transform(D_sig_raw).astype(np.float32)
            phi_D = rbf_features(D_sig, centers, sigma, DEVICE)
            t, _ = train_and_get_t(phi_R, phi_D, N_B, w_r, L2_REG, LR, N_EPOCHS, DEVICE)
            t_sig.append(t)
        t_sig = np.array(t_sig)

        t_median = np.median(t_sig)
        z_emp, z_asym = z_score_from_null(
            t_median, t_null, dof_hat=dof_hat, scale_hat=scale_hat, loc_hat=loc_hat
        )
        asym_flag = "" if gof["chi2_ok"] else "  [chi2 GOF FAILED -- treat as qualitative]"
        print(f" median t = {t_median:.2f} -> Z_empirical = {z_emp:.2f}, "
              f"Z_asymptotic = {z_asym:.2f}{asym_flag}")

        results["injections"][f"{frac}"] = {
            "n_signal": n_s,
            "t_values": t_sig.tolist(),
            "t_median": float(t_median),
            "z_empirical": float(z_emp),
            "z_asymptotic": float(z_asym),
            "z_asymptotic_trustworthy": bool(gof["chi2_ok"]),
        }

    with open(os.path.join(OUT_DIR, "nplm_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(t_null, bins=30, density=True, alpha=0.5, label="background-only (null)", color="purple")
    for frac in INJECTION_FRACTIONS:
        t_sig = np.array(results["injections"][f"{frac}"]["t_values"])
        ax.hist(t_sig, bins=30, density=True, alpha=0.5, label=f"{frac*100:.1f}% signal")
    ax.set_xlabel("NPLM test statistic t")
    ax.set_ylabel("probability density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "nplm_t_distributions.png"), dpi=150)
    plt.close(fig)

    print(f"\nSaved results to {OUT_DIR}")
    return results


if __name__ == "__main__":
    run_nplm_experiment()
