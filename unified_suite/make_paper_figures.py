#!/usr/bin/env python3
"""
Generate all publication-quality figures for the Foundation Models manuscript
and copy them to both results/figures/ and Foundation-Models/paper/images/.
"""
import glob
import json
import shutil
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.decomposition import PCA

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from common import PROCESS_NAMES, load_events_fast

EVAL_PARQUET = Path("/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/data/eval_downstream.parquet")
RESULTS_DIR = Path("/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/results")
EMBED_DIR = RESULTS_DIR / "embeddings"
PROBE_DIR = RESULTS_DIR / "probes"
NPLM_DIR = RESULTS_DIR / "nplm"
VQVAE_DIR = RESULTS_DIR / "vqvae_fidelity"
FIG_DIR = RESULTS_DIR / "figures"
PAPER_IMG_DIR = REPO_ROOT / "paper" / "images"

MODEL_LABELS = {
    "supcon": "Supervised Contrastive (SupCon)",
    "supervised_e2e": "End-to-End Supervised",
    "ssl_kinematic": "Continuous SSL (Kinematic)",
    "ssl_kinematic_drop": "Continuous SSL (Kinematic + Track Drop)",
    "ssl_legacy": "Continuous SSL (Legacy Uncoupled Noise)",
    "ssl_tokens": "Tokenized SSL (VQ-VAE K=8192)",
    "ae_transformer": "Transformer Summary Bottleneck (Pre-LN)",
    "ae_transformer_postln": "Transformer Summary Bottleneck (Post-LN)",
    "ae_mlp": "MLP Summary Autoencoder",
    "vqvae_encoder": "VQ-VAE Pre-Quantization Encoder",
    "vqvae_code": "VQ-VAE Code Projection",
    "random_init": "Randomly Initialized Frozen Transformer",
}

MODEL_COLORS = {
    "supcon": "#1f77b4",
    "supervised_e2e": "#003f5c",
    "ssl_kinematic": "#2ca02c",
    "ssl_kinematic_drop": "#17becf",
    "ssl_legacy": "#98df8a",
    "ssl_tokens": "#ff7f0e",
    "ae_transformer": "#9467bd",
    "ae_transformer_postln": "#c5b0d5",
    "ae_mlp": "#d62728",
    "random_init": "#7f7f7f",
}


def save_both(fig, stem: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_IMG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png"]:
        out = FIG_DIR / f"{stem}.{ext}"
        fig.savefig(out, dpi=220, bbox_inches="tight")
        shutil.copy2(out, PAPER_IMG_DIR / f"{stem}.{ext}")
    print(f"[make_paper_figures] Saved {stem}.pdf and {stem}.png")


def plot_hp_v1_vs_v2_displaced():
    """
    Figure 1: Physics validation of the fixed Higgs Portal v2 (displaced LLP vertices)
    vs legacy Higgs Portal v1 (ddsim status-22 prompt-vertex bug), Hidden Valley, and SM.
    """
    eval_data = load_events_fast(EVAL_PARQUET)
    raw6 = eval_data["raw6"]
    mask = eval_data["mask"]
    lengths = eval_data["lengths"]
    labels = eval_data["labels"]

    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))

    channel_specs = [
        ([0, 1, 2], "SM Combined ($t\\bar{t}$, ggF, $hh$)", "black", "-", 1.8),
        ([4], "Hidden Valley (semi-visible jets)", "#2ca02c", "--", 1.8),
        ([5], "Higgs Portal v1 (legacy prompt bug)", "#d62728", ":", 2.0),
        ([3], "Higgs Portal v2 (fixed displaced LLP)", "#1f77b4", "-", 2.2),
    ]

    for lbl_list, cname, color, ls, lw in channel_specs:
        ev_sel = np.where(np.isin(labels, lbl_list))[0]
        if len(ev_sel) == 0:
            continue
        take = rng.choice(ev_sel, size=min(15_000, len(ev_sel)), replace=False)
        m_sub = mask[take]
        r6_sub = raw6[take]
        d0_trks = r6_sub[:, :, 0][m_sub]
        z0_trks = r6_sub[:, :, 1][m_sub]
        n_trks = lengths[take]

        axes[0].hist(d0_trks, bins=80, range=(-5.5, 5.5), density=True, histtype="step",
                     color=color, ls=ls, lw=lw, label=cname)
        axes[1].hist(np.abs(d0_trks), bins=60, range=(0.0, 6.0), density=True, histtype="step",
                     color=color, ls=ls, lw=lw, label=cname)
        axes[2].hist(n_trks, bins=50, range=(0, 160), density=True, histtype="step",
                     color=color, ls=ls, lw=lw, label=cname)

    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"Track $\mathrm{symlog}(d_0 / \mathrm{mm})$", fontsize=11)
    axes[0].set_ylabel("Track Probability Density", fontsize=11)
    axes[0].set_title(r"(a) Transverse Impact Parameter $\mathrm{symlog}(d_0)$", fontsize=11)

    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"Track $|\mathrm{symlog}(d_0 / \mathrm{mm})|$", fontsize=11)
    axes[1].set_ylabel("Track Probability Density", fontsize=11)
    axes[1].set_title(r"(b) Displaced Track Tail ($|\mathrm{symlog}(d_0)|$)", fontsize=11)

    axes[2].set_xlabel(r"Reconstructed Track Multiplicity $N_{\mathrm{tracks}}$", fontsize=11)
    axes[2].set_ylabel("Event Probability Density", fontsize=11)
    axes[2].set_title(r"(c) Event Track Multiplicity Distribution", fontsize=11)
    axes[2].legend(frameon=False, fontsize=8.5, loc="upper right")

    fig.tight_layout()
    save_both(fig, "fig_hp_v1_vs_v2_displaced")
    plt.close(fig)


def plot_latent_pca_grid():
    """
    Figure 2: 2D PCA projections of D=64 held-out representations across model families.
    """
    models_to_plot = [
        ("supcon_d64", "(a) Supervised Contrastive (SupCon)"),
        ("ssl_kinematic_d64", "(b) Continuous SSL (Kinematic)"),
        ("ssl_tokens_d64", "(c) Tokenized SSL (VQ-VAE K=8192)"),
        ("ae_transformer_d64", "(d) Transformer Summary Bottleneck"),
        ("ae_mlp_d64", "(e) MLP Summary Autoencoder"),
        ("random_init_d64", "(f) Random Frozen Transformer"),
    ]
    available = [(m, t) for m, t in models_to_plot if (EMBED_DIR / f"{m}.npz").exists()]
    if not available:
        return

    ncols = 3
    nrows = int(np.ceil(len(available) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14.0, 4.3 * nrows))
    axes_flat = np.atleast_1d(axes).flat

    rng = np.random.default_rng(42)
    class_Order = [
        (0, "$t\\bar{t}$ (SM)", "#1f77b4", 0.35, 6),
        (1, "ggF $H$ (SM)", "#2ca02c", 0.35, 6),
        (2, "DiHiggs $hh$ (SM)", "#ff7f0e", 0.35, 6),
        (4, "Hidden Valley (BSM)", "#9467bd", 0.50, 8),
        (3, "Higgs Portal v2 (BSM)", "#d62728", 0.65, 10),
    ]

    for ax, (mname, title) in zip(axes_flat, available):
        data = np.load(EMBED_DIR / f"{mname}.npz")
        z = data["z"].astype(np.float32)
        labels = data["labels"]

        # Fit PCA on balanced SM subsample
        sm_idx = np.where(np.isin(labels, [0, 1, 2]))[0]
        sm_fit = rng.choice(sm_idx, size=min(15_000, len(sm_idx)), replace=False)
        z_mean = z[sm_fit].mean(axis=0, keepdims=True)
        z_std = np.maximum(z[sm_fit].std(axis=0, keepdims=True), 1e-6)
        pca = PCA(n_components=2, random_state=42)
        pca.fit((z[sm_fit] - z_mean) / z_std)
        var_exp = pca.explained_variance_ratio_ * 100.0

        for lbl, cname, color, alpha, sz in class_Order:
            idx_c = np.where(labels == lbl)[0]
            if len(idx_c) == 0:
                continue
            sub_c = rng.choice(idx_c, size=min(1500, len(idx_c)), replace=False)
            pts = pca.transform((z[sub_c] - z_mean) / z_std)
            q_lo, q_hi = np.percentile(pts, [1, 99], axis=0)
            keep = (pts[:, 0] >= q_lo[0]) & (pts[:, 0] <= q_hi[0]) & (pts[:, 1] >= q_lo[1]) & (pts[:, 1] <= q_hi[1])
            ax.scatter(pts[keep, 0], pts[keep, 1], s=sz, c=color, alpha=alpha, label=cname, rasterized=True)

        ax.set_title(title, fontsize=11)
        ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}% SM var.)", fontsize=9.5)
        ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}% SM var.)", fontsize=9.5)
        ax.grid(True, alpha=0.2)

    axes_flat[0].legend(frameon=True, fontsize=8.0, loc="upper right", markerscale=1.8)
    for ax in list(axes_flat)[len(available):]:
        ax.axis("off")

    fig.tight_layout()
    save_both(fig, "fig_latent_pca_grid")
    plt.close(fig)


def plot_probe_scaling_and_label_efficiency():
    """
    Figure 3: Downstream Linear Probe scaling across D in {8, 32, 64}
    Figure 4: Strictly budgeted label-efficiency curves for D=64 models.
    """
    csv_path = PROBE_DIR / "probe_summary.csv"
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)

    # Parse family and dim
    families = []
    for m in df["model"]:
        parts = m.rsplit("_d", 1)
        families.append(parts[0])
    df["family"] = families

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))
    metrics = [
        ("sm3_linear_auc", "(a) 3-Class SM Linear Probe Macro-AUC"),
        ("ood_hp_v2_auc", "(b) OOD Linear Probe AUC: SM vs Higgs Portal v2"),
        ("ood_hv_auc", "(c) OOD Linear Probe AUC: SM vs Hidden Valley"),
    ]
    plot_fams = [
        "supcon", "ssl_kinematic", "ssl_legacy", "ssl_tokens",
        "ae_transformer", "ae_transformer_postln", "ae_mlp", "random_init"
    ]

    for ax, (col, title) in zip(axes, metrics):
        for fam in plot_fams:
            sub = df[df["family"] == fam].sort_values("latent_dim")
            if sub.empty:
                continue
            ax.plot(
                sub["latent_dim"], sub[col],
                marker="o", lw=2.0,
                color=MODEL_COLORS.get(fam, "black"),
                ls="--" if ("postln" in fam or "legacy" in fam or "random" in fam) else "-",
                label=MODEL_LABELS.get(fam, fam),
            )
        # Also plot E2E supervised reference line if available
        e2e = df[df["family"] == "supervised_e2e"]
        if not e2e.empty and col in e2e:
            val = float(e2e[col].iloc[0])
            ax.axhline(val, color="#003f5c", ls=":", lw=1.6, label="Supervised E2E (D=64)")

        ax.set_xscale("log", base=2)
        ax.set_xticks([8, 32, 64])
        ax.set_xticklabels(["8", "32", "64"])
        ax.set_xlabel("Latent Embedding Dimension $D$", fontsize=11)
        ax.set_ylabel("Test ROC-AUC", fontsize=11)
        ax.set_title(title, fontsize=10.5)
        ax.grid(True, alpha=0.25)

    axes[0].legend(frameon=True, fontsize=7.8, loc="lower right")
    fig.tight_layout()
    save_both(fig, "fig_probe_scaling")
    plt.close(fig)

    # Figure 4: Label efficiency curves
    curve_path = PROBE_DIR / "label_efficiency_curves.csv"
    if curve_path.exists():
        dfc = pd.read_csv(curve_path)
        dfc["family"] = [m.rsplit("_d", 1)[0] for m in dfc["model"]]
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5), sharey=True)
        for ax, (ptype, ptitle) in zip(axes, [("linear", "(a) Affine Linear Probe"), ("mlp", "(b) 2-Layer Nonlinear MLP Probe")]):
            sub_p = dfc[dfc["probe_type"] == ptype]
            for fam in ["supcon", "supervised_e2e", "ssl_kinematic", "ssl_kinematic_drop", "ssl_tokens", "ae_transformer", "ae_mlp", "random_init"]:
                sub_f = sub_p[sub_p["family"] == fam]
                if sub_f.empty:
                    continue
                grp = sub_f.groupby("n_labeled")["test_auc"].agg(["mean", "std"]).reset_index()
                ax.plot(grp["n_labeled"], grp["mean"], marker="o", lw=2.0,
                        color=MODEL_COLORS.get(fam, "black"), label=MODEL_LABELS.get(fam, fam))
                ax.fill_between(grp["n_labeled"], grp["mean"] - grp["std"], grp["mean"] + grp["std"],
                                color=MODEL_COLORS.get(fam, "black"), alpha=0.15)
            ax.set_xscale("log")
            ax.set_xlabel("Total Labeled SM Events $N_{\\mathrm{labeled}}$ (80% Train / 20% Val)", fontsize=10.5)
            ax.set_ylabel("3-Class SM Test Macro-AUC", fontsize=10.5)
            ax.set_title(ptitle, fontsize=11)
            ax.grid(True, alpha=0.25)
        axes[0].legend(frameon=True, fontsize=8.0, loc="lower right")
        fig.tight_layout()
        save_both(fig, "fig_label_efficiency")
        plt.close(fig)


def plot_nplm_results():
    """
    Figure 5: NPLM Rate-vs-Shape Confounding & Null Calibration
    Figure 6: NPLM Anomaly Detection Significance vs Signal Injection Fraction
    """
    json_files = sorted(glob.glob(str(NPLM_DIR / "*" / "*" / "nplm_results.json")))
    if not json_files:
        return

    records = []
    raw_by_key = {}
    for jf in json_files:
        with open(jf) as f:
            d = json.load(f)
        m = d["model"]
        fam = m.rsplit("_d", 1)[0] if "_d" in m else m
        sig = d["signal_name"]
        mode = d["sample_mode"]
        raw_by_key[(mode, sig, m)] = d
        for frac_s, inj in d["injections"].items():
            records.append({
                "model": m,
                "family": fam,
                "latent_dim": d["latent_dim"],
                "signal": sig,
                "mode": mode,
                "frac": float(frac_s),
                "t_median": inj["t_median"],
                "t_q16": inj["t_q16"],
                "t_q84": inj["t_q84"],
                "t_rate_only": inj["t_rate_only"],
                "z_empirical": inj["z_empirical"],
                "z_asymptotic": inj["z_asymptotic"],
                "z_chi2_unscaled": inj["z_chi2_unscaled"],
                "snr": inj["snr"],
                "null_mean": d["null_chi2_gof"]["mean"],
                "null_std": d["null_chi2_gof"]["std"],
                "chi2_ok": d["null_chi2_gof"]["chi2_ok"],
            })
    df_nplm = pd.DataFrame(records)
    df_nplm.to_csv(NPLM_DIR / "nplm_summary_all.csv", index=False)
    print(f"[make_paper_figures] Saved {NPLM_DIR / 'nplm_summary_all.csv'} ({len(df_nplm)} rows)")

    # Plot NPLM significance vs injection fraction (shape mode, D=64)
    sub_shape64 = df_nplm[(df_nplm["mode"] == "shape") & (df_nplm["latent_dim"] == 64)]
    if not sub_shape64.empty:
        fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6))
        for ax, (sig_name, stitle) in zip(axes, [
            ("higgs_portal_v2", "(a) Higgs Portal v2 (Displaced LLP, Shape-Only NPLM)"),
            ("hidden_valley", "(b) Hidden Valley (Semi-Visible Jets, Shape-Only NPLM)"),
        ]):
            s_df = sub_shape64[sub_shape64["signal"] == sig_name]
            for fam in ["supcon", "ssl_kinematic", "ssl_kinematic_drop", "ssl_tokens", "ae_transformer", "ae_mlp", "random_init"]:
                f_df = s_df[s_df["family"] == fam].sort_values("frac")
                if f_df.empty:
                    continue
                ax.plot(f_df["frac"] * 100.0, f_df["z_asymptotic"], marker="o", lw=2.0,
                        color=MODEL_COLORS.get(fam, "black"), label=MODEL_LABELS.get(fam, fam))
            ax.axhline(3.0, color="gray", ls="--", lw=1.0, alpha=0.7, label="$3\\sigma$ Evidence")
            ax.axhline(5.0, color="black", ls=":", lw=1.2, alpha=0.8, label="$5\\sigma$ Discovery")
            ax.set_xlabel("Injected Signal Fraction $N_S / N_B$ [%]", fontsize=11)
            ax.set_ylabel("Median Asymptotic Significance $Z$ [$\\sigma$]", fontsize=11)
            ax.set_title(stitle, fontsize=11)
            ax.grid(True, alpha=0.25)
        axes[0].legend(frameon=True, fontsize=8.0, loc="upper left")
        fig.tight_layout()
        save_both(fig, "fig_nplm_significance_curves")
        plt.close(fig)

    # Figure 6: Higgs Portal v1 (buggy prompt) vs v2 (fixed displaced LLP) NPLM comparison at D=64
    sub_hp_comp = df_nplm[(df_nplm["mode"] == "shape") & (df_nplm["latent_dim"] == 64) & (df_nplm["signal"].isin(["higgs_portal_v1", "higgs_portal_v2"]))]
    if not sub_hp_comp.empty:
        fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.5), sharey=True)
        for ax, (sig_name, stitle) in zip(axes, [
            ("higgs_portal_v1", "(a) Legacy Higgs Portal v1 (Prompt-Vertex Bug)"),
            ("higgs_portal_v2", "(b) Fixed Higgs Portal v2 (Displaced LLP Vertices)"),
        ]):
            s_df = sub_hp_comp[sub_hp_comp["signal"] == sig_name]
            for fam in ["ae_transformer", "ae_mlp", "ssl_kinematic", "ssl_tokens", "supcon", "random_init"]:
                f_df = s_df[s_df["family"] == fam].sort_values("frac")
                if f_df.empty:
                    continue
                ax.plot(f_df["frac"] * 100.0, f_df["z_asymptotic"], marker="o", lw=2.0,
                        color=MODEL_COLORS.get(fam, "black"), label=MODEL_LABELS.get(fam, fam))
            ax.axhline(3.0, color="gray", ls="--", lw=1.0, alpha=0.7, label="$3\\sigma$ Evidence")
            ax.axhline(5.0, color="black", ls=":", lw=1.2, alpha=0.8, label="$5\\sigma$ Discovery")
            ax.set_xlabel("Injected Signal Fraction $N_S / N_B$ [%]", fontsize=11)
            ax.set_ylabel("Median Asymptotic Significance $Z$ [$\\sigma$]", fontsize=11)
            ax.set_title(stitle, fontsize=11)
            ax.grid(True, alpha=0.25)
        axes[0].legend(frameon=True, fontsize=8.0, loc="upper left")
        fig.tight_layout()
        save_both(fig, "fig_nplm_v1_vs_v2")
        plt.close(fig)

    # Figure 7: Null chi2 calibration & Additive vs Shape-Only confounding
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.5))
    ref_key = ("shape", "higgs_portal_v2", "ae_transformer_d64")
    if ref_key not in raw_by_key:
        ref_key = next(iter(raw_by_key.keys()))
    d_ref = raw_by_key[ref_key]
    t_null = np.array(d_ref["null_t"], dtype=np.float64)
    gof = d_ref["null_chi2_gof"]
    ax0 = axes[0]
    ax0.hist(t_null, bins=22, density=True, alpha=0.55, color="#1f77b4", label=f"SM Null Toys ($N={len(t_null)}$)")
    xs = np.linspace(max(0.0, t_null.min() * 0.85), t_null.max() * 1.15, 300)
    pdf_fit = stats.chi2.pdf(xs, df=gof["dof_mle"], loc=gof["loc_mle"], scale=gof["scale_mle"])
    ax0.plot(xs, pdf_fit, color="black", lw=2.0, label=f"Scaled $\\chi^2$ Fit ($\\mathrm{{dof}}={gof['dof_mle']:.0f},\\ p_{{\\mathrm{{KS}}}}={gof['ks_pvalue']:.2f}$)")
    for frac_s, col in [("0.005", "#2ca02c"), ("0.01", "#ff7f0e")]:
        if frac_s in d_ref["injections"]:
            ts = np.array(d_ref["injections"][frac_s]["t_values"])
            ax0.hist(ts, bins=18, density=True, alpha=0.45, color=col, label=f"{float(frac_s)*100:.1f}% Higgs Portal v2 ($H_1$)")
    ax0.set_xlabel("NPLM Test Statistic $t$", fontsize=11)
    ax0.set_ylabel("Probability Density", fontsize=11)
    ax0.set_title(f"(a) Null $\\chi^2$ Calibration & Signal Shift ({d_ref['model']})", fontsize=10.5)
    ax0.legend(frameon=True, fontsize=8.0, loc="upper right")
    ax0.grid(True, alpha=0.25)

    ax1 = axes[1]
    sub_add = df_nplm[(df_nplm["signal"] == "higgs_portal_v2") & (df_nplm["latent_dim"] == 64)]
    for fam in ["ae_transformer", "ssl_kinematic", "ssl_tokens", "supcon", "random_init"]:
        for mode, ls, m_lbl in [("shape", "-", "Shape-Only"), ("additive", "--", "Additive (Rate+Shape)")]:
            f_df = sub_add[(sub_add["family"] == fam) & (sub_add["mode"] == mode)].sort_values("frac")
            if f_df.empty:
                continue
            ax1.plot(f_df["frac"] * 100.0, f_df["t_median"] - f_df["null_mean"], marker="o" if mode == "shape" else "s",
                     lw=1.8, ls=ls, color=MODEL_COLORS.get(fam, "black"),
                     label=f"{MODEL_LABELS.get(fam, fam)} [{m_lbl}]" if fam in ["ae_transformer", "random_init"] else None)
    # Plot pure Poisson rate contribution 2 * ((N_B + N_S) log(1 + N_S/N_B) - N_S)
    fracs = np.array([0.005, 0.01, 0.025, 0.05])
    n_b = 100_000.0
    n_s = fracs * n_b
    t_rate = 2.0 * ((n_b + n_s) * np.log(1.0 + n_s / n_b) - n_s)
    ax1.plot(fracs * 100.0, t_rate, color="crimson", ls=":", lw=2.2, label="Pure Rate Excess $t_{\\mathrm{rate}} \\approx N_S^2 / N_B$")
    ax1.set_yscale("log")
    ax1.set_xlabel("Injected Signal Fraction $N_S / N_B$ [%]", fontsize=11)
    ax1.set_ylabel("Excess Test Statistic $\\Delta t = t_{\\mathrm{med}} - \\langle t_{\\mathrm{null}}\\rangle$", fontsize=11)
    ax1.set_title("(b) Rate-vs-Shape Deconfounding in Extended NPLM", fontsize=10.5)
    ax1.legend(frameon=True, fontsize=7.8, loc="upper left")
    ax1.grid(True, alpha=0.25)

    fig.tight_layout()
    save_both(fig, "fig_nplm_calibration_and_rate")
    plt.close(fig)


def main():
    if (VQVAE_DIR / "vqvae_fidelity_panel.pdf").exists():
        PAPER_IMG_DIR.mkdir(parents=True, exist_ok=True)
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        for ext in ["pdf", "png"]:
            shutil.copy2(VQVAE_DIR / f"vqvae_fidelity_panel.{ext}", PAPER_IMG_DIR / f"fig_vqvae_fidelity.{ext}")
            shutil.copy2(VQVAE_DIR / f"vqvae_fidelity_panel.{ext}", FIG_DIR / f"fig_vqvae_fidelity.{ext}")
    plot_hp_v1_vs_v2_displaced()
    plot_latent_pca_grid()
    plot_probe_scaling_and_label_efficiency()
    plot_nplm_results()


if __name__ == "__main__":
    main()
