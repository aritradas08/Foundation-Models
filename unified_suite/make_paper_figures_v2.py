#!/usr/bin/env python3
"""
Generate all Phase 2 publication-quality figures from results/*_v2/
and copy them to both results/figures_v2/ and Foundation-Models/paper/images/.
"""
import glob
import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams["font.family"] = "DejaVu Sans"
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.decomposition import PCA

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from common import PROCESS_NAMES, load_events_fast

DATA_DIR = Path("/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/data")
EVAL_PARQUET = DATA_DIR / "eval_downstream_1050k.parquet"
HP_2X2_DIR = DATA_DIR / "hp_2x2_control"
RESULTS_DIR = Path("/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/results")

EMBED_DIR = RESULTS_DIR / "embeddings_v2"
PROBE_DIR = RESULTS_DIR / "probes_v2"
NPLM_DIR = RESULTS_DIR / "nplm_v2"
VQVAE_V2_DIR = RESULTS_DIR / "vqvae_fidelity_v2"
VQVAE_V1_DIR = RESULTS_DIR / "vqvae_fidelity"
CTRL_DIR = RESULTS_DIR / "controls_v2"
FIG_DIR = RESULTS_DIR / "figures_v2"
PAPER_IMG_DIR = REPO_ROOT / "paper" / "images"

MODEL_LABELS = {
    "supcon": "Supervised Contrastive (SupCon)",
    "supervised_e2e": "End-to-End Supervised (900k SM)",
    "supervised_scratch": "End-to-End Supervised (From Scratch)",
    "ssl_kinematic": "Continuous SSL (Kinematic, Frozen)",
    "ssl_kinematic_ft": "Continuous SSL + LP-FT (Unfrozen)",
    "ssl_kinematic_drop": "Continuous SSL (Kinematic + Drop)",
    "ssl_legacy": "Continuous SSL (Legacy Shortcut)",
    "ssl_tokens": "Tokenized SSL (VQ-VAE K=8192, Frozen)",
    "ssl_tokens_ft": "Tokenized SSL + LP-FT (Unfrozen)",
    "ae_transformer": "Transformer Summary Bottleneck (Pre-LN)",
    "ae_transformer_postln": "Transformer Summary Bottleneck (Post-LN)",
    "ae_mlp": "MLP Summary Autoencoder",
    "vqvae_encoder": "VQ-VAE Pre-VQ Encoder (D=128)",
    "vqvae_code": "VQ-VAE Pre-VQ Bottleneck (D=8)",
    "vqvae_quant": "VQ-VAE Post-VQ Codebook (D=8)",
    "raw_summary": "Direct 28-D Track Summary (No NN)",
    "pca_summary": "PCA-8 of 28-D Summary (No NN)",
    "phys_handcrafted": "6-D Handcrafted Physics Obs. (No NN)",
    "mult_only": "Track Multiplicity Only (D=1, No NN)",
    "random_init": "Random Frozen Set Transformer",
}

MODEL_COLORS = {
    "supcon": "#1f77b4",
    "supervised_e2e": "#003f5c",
    "supervised_scratch": "#e377c2",
    "ssl_kinematic": "#2ca02c",
    "ssl_kinematic_ft": "#1b661b",
    "ssl_kinematic_drop": "#17becf",
    "ssl_legacy": "#98df8a",
    "ssl_tokens": "#ff7f0e",
    "ssl_tokens_ft": "#c45100",
    "ae_transformer": "#9467bd",
    "ae_transformer_postln": "#c5b0d5",
    "ae_mlp": "#d62728",
    "vqvae_code": "#8c564b",
    "vqvae_quant": "#bcbd22",
    "raw_summary": "#111111",
    "pca_summary": "#555555",
    "phys_handcrafted": "#bc5090",
    "mult_only": "#aaaaaa",
    "random_init": "#7f7f7f",
}


def save_both(fig, stem: str, pdf_dpi: int = 220):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_IMG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png"]:
        out = FIG_DIR / f"{stem}.{ext}"
        dpi_use = pdf_dpi if ext == "pdf" else 220
        fig.savefig(out, dpi=dpi_use, bbox_inches="tight")
        shutil.copy2(out, PAPER_IMG_DIR / f"{stem}.{ext}")
    print(f"[figures_v2] Saved {stem}.pdf and {stem}.png")


def parse_family(model_name: str) -> str:
    if "_d" in model_name:
        return model_name.rsplit("_d", 1)[0]
    return model_name


def plot_hp_v1_vs_v2_displaced():
    """
    Figure 1: Physics validation of Higgs Portal v2 (3.0 T & matched 2.0 T seeding)
    vs legacy Higgs Portal v1 (ddsim status-22 prompt-vertex bug), Hidden Valley, and SM.
    """
    eval_data = load_events_fast(EVAL_PARQUET)
    raw6 = eval_data["raw6"]
    mask = eval_data["mask"]
    lengths = eval_data["lengths"]
    labels = eval_data["labels"]

    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.2))

    channel_specs = [
        ([0, 1, 2], "SM Combined ($t\\bar{t}$, ggF, $HH$; 2T seed)", "black", "-", 1.8),
        ([4], "Hidden Valley (dark shower; 2T seed)", "#2ca02c", "--", 1.8),
        ([5], "Higgs Portal v1 (prompt bug; 2T seed)", "#d62728", ":", 2.0),
        ([3], "Higgs Portal v2 (displaced; 3T seed)", "#1f77b4", "-", 2.1),
    ]

    for lbl_list, cname, color, ls, lw in channel_specs:
        ev_sel = np.where(np.isin(labels, lbl_list))[0]
        take = rng.choice(ev_sel, size=min(20_000, len(ev_sel)), replace=False)
        m_sub = mask[take]
        d0_trks = raw6[take][:, :, 0][m_sub]
        n_trks = lengths[take]

        axes[0].hist(d0_trks, bins=80, range=(-5.5, 5.5), density=True, histtype="step",
                     color=color, ls=ls, lw=lw, label=cname)
        axes[1].hist(np.abs(d0_trks), bins=60, range=(0.0, 6.0), density=True, histtype="step",
                     color=color, ls=ls, lw=lw, label=cname)
        axes[2].hist(n_trks, bins=50, range=(0, 160), density=True, histtype="step",
                     color=color, ls=ls, lw=lw, label=cname)

    # Overlay matched-seeding v2_displaced_2T control if available!
    p_v2_2t = HP_2X2_DIR / "v2_displaced_2T_runs0_9.parquet"
    if p_v2_2t.exists():
        d2t = load_events_fast(p_v2_2t)
        d0_2t = d2t["raw6"][:, :, 0][d2t["mask"]]
        axes[0].hist(d0_2t, bins=80, range=(-5.5, 5.5), density=True, histtype="step",
                     color="#ff7f0e", ls="-.", lw=1.9, label="Higgs Portal v2 (displaced; matched 2T seed)")
        axes[1].hist(np.abs(d0_2t), bins=60, range=(0.0, 6.0), density=True, histtype="step",
                     color="#ff7f0e", ls="-.", lw=1.9, label="Higgs Portal v2 (displaced; matched 2T seed)")
        axes[2].hist(d2t["lengths"], bins=50, range=(0, 160), density=True, histtype="step",
                     color="#ff7f0e", ls="-.", lw=1.9, label="Higgs Portal v2 (displaced; matched 2T seed)")

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
    axes[2].legend(frameon=False, fontsize=7.8, loc="upper right")

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
        ("raw_summary_d28", "(f) Direct 28-D Track Summary (No NN)"),
    ]
    available = [(m, t) for m, t in models_to_plot if (EMBED_DIR / f"{m}.npz").exists()]
    if not available:
        return

    ncols = 3
    nrows = int(np.ceil(len(available) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14.0, 4.3 * nrows))
    axes_flat = np.atleast_1d(axes).flat

    rng = np.random.default_rng(42)
    class_order = [
        (0, "$t\\bar{t}$ (SM)", "#1f77b4", 0.35, 6),
        (1, "ggF $H$ (SM)", "#2ca02c", 0.35, 6),
        (2, "DiHiggs $HH$ (SM)", "#ff7f0e", 0.35, 6),
        (4, "Hidden Valley (BSM)", "#9467bd", 0.50, 8),
        (3, "Higgs Portal v2 (BSM)", "#d62728", 0.65, 10),
    ]

    for ax, (mname, title) in zip(axes_flat, available):
        data = np.load(EMBED_DIR / f"{mname}.npz")
        z = data["z"].astype(np.float32)
        labels = data["labels"]

        sm_idx = np.where(np.isin(labels, [0, 1, 2]))[0]
        sm_fit = rng.choice(sm_idx, size=min(20_000, len(sm_idx)), replace=False)
        z_mean = z[sm_fit].mean(axis=0, keepdims=True)
        z_std = np.maximum(z[sm_fit].std(axis=0, keepdims=True), 1e-6)
        pca = PCA(n_components=2, random_state=42)
        pca.fit((z[sm_fit] - z_mean) / z_std)
        var_exp = pca.explained_variance_ratio_ * 100.0

        for lbl, cname, color, alpha, sz in class_order:
            idx_c = np.where(labels == lbl)[0]
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
    save_both(fig, "fig_latent_pca_grid", pdf_dpi=110)
    plt.close(fig)


def plot_probe_scaling_and_label_efficiency():
    """
    Figure 3: Downstream Linear Probe scaling across D in {8, 32, 64} + Zero-NN baselines
    Figure 4: Budgeted label-efficiency curves including End-to-End Supervised From Scratch!
    """
    csv_path = PROBE_DIR / "probe_summary.csv"
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    df["family"] = [parse_family(m) for m in df["model"]]

    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.5))
    metrics = [
        ("sm3_linear_auc", "(a) 3-Class SM Linear Probe Macro-AUC"),
        ("ood_hp_v2_auc", "(b) Transfer Linear Probe AUC: SM vs Higgs Portal v2"),
        ("ood_hv_auc", "(c) Transfer Linear Probe AUC: SM vs Hidden Valley"),
    ]
    plot_fams = [
        "supcon", "ssl_kinematic", "ssl_tokens", "ssl_legacy",
        "ae_transformer", "ae_mlp", "random_init",
    ]

    for ax, (col, title) in zip(axes, metrics):
        for fam in plot_fams:
            sub = df[(df["family"] == fam) & (df["latent_dim"].isin([8, 32, 64]))].sort_values("latent_dim")
            if sub.empty:
                continue
            ax.plot(
                sub["latent_dim"], sub[col],
                marker="o", lw=2.0,
                color=MODEL_COLORS.get(fam, "black"),
                ls="--" if ("legacy" in fam or "random" in fam) else "-",
                label=MODEL_LABELS.get(fam, fam),
            )
        for ref_m, ref_lbl, ref_col, ref_ls in [
            ("supervised_e2e_d64", "Supervised E2E (D=64)", "#003f5c", ":"),
            ("raw_summary_d28", "Direct 28-D Summary (No NN)", "#111111", "-."),
            ("phys_handcrafted_d6", "6-D Physics Obs. (No NN)", "#bc5090", ":"),
        ]:
            rm = df[df["model"] == ref_m]
            if not rm.empty:
                ax.axhline(float(rm[col].iloc[0]), color=ref_col, ls=ref_ls, lw=1.5, label=ref_lbl)

        ax.set_xscale("log", base=2)
        ax.set_xticks([8, 32, 64])
        ax.set_xticklabels(["8", "32", "64"])
        ax.set_xlabel("Latent Embedding Dimension $D$", fontsize=11)
        ax.set_ylabel("Paired Test ROC-AUC", fontsize=11)
        ax.set_title(title, fontsize=10.5)
        ax.grid(True, alpha=0.25)

    axes[0].legend(frameon=True, fontsize=7.3, loc="lower right")
    fig.tight_layout()
    save_both(fig, "fig_probe_scaling")
    plt.close(fig)

    # Figure 4: Label efficiency curves including supervised_scratch_d64 and Phase 3 LP-FT curves!
    curve_path = PROBE_DIR / "label_efficiency_curves.csv"
    if curve_path.exists():
        dfc = pd.read_csv(curve_path)
        ft_dir = RESULTS_DIR / "phase3_hardening" / "finetune_curves"
        if ft_dir.exists():
            ft_dfs = [pd.read_csv(p) for p in sorted(ft_dir.glob("*_curves.csv"))]
            if ft_dfs:
                df_ft = pd.concat(ft_dfs, ignore_index=True)
                df_ft_lin = df_ft.copy()
                df_ft_lin["probe_type"] = "linear"
                dfc = pd.concat([dfc, df_ft_lin], ignore_index=True)
        dfc["family"] = [parse_family(m) for m in dfc["model"]]
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.7), sharey=True)
        fams_curve = [
            "supcon", "ssl_kinematic_ft", "ssl_tokens_ft",
            "ssl_kinematic", "ssl_kinematic_drop", "ssl_tokens",
            "ae_transformer", "raw_summary", "supervised_scratch", "random_init",
        ]
        for ax, (ptype, ptitle) in zip(axes, [
            ("linear", "(a) Frozen Linear Probes, LP-FT Fine-Tuning & Scratch"),
            ("mlp", "(b) Frozen 2-Layer MLP Probes & Scratch Baseline"),
        ]):
            sub_p = dfc[dfc["probe_type"] == ptype]
            for fam in fams_curve:
                sub_f = sub_p[sub_p["family"] == fam]
                if sub_f.empty:
                    continue
                grp = sub_f.groupby("n_labeled")["test_auc"].agg(["mean", "std"]).reset_index()
                if fam in ("supervised_scratch", "random_init"):
                    ls, mk, lw = "--", ("s" if fam == "supervised_scratch" else "o"), (2.1 if fam == "supervised_scratch" else 1.8)
                elif fam.endswith("_ft"):
                    ls, mk, lw = "-", "^", 2.2
                elif fam == "raw_summary":
                    ls, mk, lw = "-.", "o", 1.8
                else:
                    ls, mk, lw = "-", "o", 1.8
                ax.plot(grp["n_labeled"], grp["mean"], marker=mk, lw=lw,
                        ls=ls, color=MODEL_COLORS.get(fam, "black"), label=MODEL_LABELS.get(fam, fam))
                ax.fill_between(grp["n_labeled"], grp["mean"] - grp["std"], grp["mean"] + grp["std"],
                                color=MODEL_COLORS.get(fam, "black"), alpha=0.13)
            ax.set_xscale("log")
            ax.set_xlabel("Total Labelled SM Events $N_{\\mathrm{labelled}}$ (80% Train / 20% Val)", fontsize=10.5)
            ax.set_ylabel("3-Class SM Test Macro ROC-AUC", fontsize=10.5)
            ax.set_title(ptitle, fontsize=10.8)
            ax.grid(True, alpha=0.25)
        axes[0].legend(frameon=True, fontsize=7.0, loc="lower right")
        fig.tight_layout()
        save_both(fig, "fig_label_efficiency")
        plt.close(fig)


def plot_nplm_highstats_and_rate():
    """
    Figure 5: 10,000-Null Empirical Calibration + Rate-vs-Shape Deconfounding
    Figure 6: Empirically Calibrated Discovery Power (alpha = 1e-2) + SNR [q16, q84] Curves
    Figure 7: Legacy HP v1 vs Corrected HP v2 Shape-Only NPLM
    """
    hs_csv = NPLM_DIR / "highstats_calibrated" / "highstats_power_10k_nulls.csv"
    sh_csv = NPLM_DIR / "nplm_summary_shape.csv"
    ad_csv = NPLM_DIR / "nplm_summary_additive.csv"

    if hs_csv.exists():
        df_hs = pd.read_csv(hs_csv)
        df_hs["family"] = [parse_family(m) for m in df_hs["model"]]

        # 2x2 Panel: Top row = Empirical Power at alpha=1e-2 (with 95% Clopper-Pearson CIs);
        #            Bottom row = Distribution-Free SNR with [q16, q84] bands!
        fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.2), sharex=True)
        plot_models = [
            ("supcon_d64", "SupCon (D=64)", "#1f77b4", "-", "o"),
            ("supervised_e2e_d64", "Supervised E2E (D=64)", "#003f5c", "--", "s"),
            ("ae_mlp_d32", "MLP Summary AE (D=32)", "#d62728", "-", "o"),
            ("ae_transformer_d32", "Transformer Summary AE (D=32)", "#9467bd", "-", "^"),
            ("vqvae_code_d8", "VQ-VAE Pre-VQ Pool (D=8)", "#8c564b", "-", "D"),
            ("ssl_tokens_d8", "Tokenized SSL (D=8)", "#ff7f0e", "-", "o"),
            ("ssl_kinematic_d32", "Continuous SSL Kinematic (D=32)", "#2ca02c", "-", "o"),
            ("raw_summary_d28", "Direct 28-D Summary (No NN)", "#111111", "-.", "s"),
            ("phys_handcrafted_d6", "6-D Physics Obs. (No NN)", "#bc5090", ":", "v"),
            ("random_init_d64", "Random Set Transformer (D=64)", "#7f7f7f", "--", "x"),
        ]

        for col_idx, (sig_name, sig_title) in enumerate([
            ("higgs_portal_v2", "Higgs Portal v2 (Displaced LLP)"),
            ("hidden_valley", "Hidden Valley (Dark Shower)"),
        ]):
            ax_pow = axes[0, col_idx]
            ax_snr = axes[1, col_idx]
            sub_s = df_hs[df_hs["signal"] == sig_name]

            for mname, mlbl, col, ls, mk in plot_models:
                mdf = sub_s[sub_s["model"] == mname].sort_values("frac")
                if mdf.empty:
                    continue
                x = mdf["frac"].values * 100.0
                y_pow = mdf["power_1e2"].values
                y_lo = mdf["power_1e2_ci_lo"].values
                y_hi = mdf["power_1e2_ci_hi"].values
                ax_pow.plot(x, y_pow, marker=mk, lw=1.9, ls=ls, color=col, label=mlbl)
                ax_pow.fill_between(x, y_lo, y_hi, color=col, alpha=0.12)

                ax_snr.plot(x, mdf["snr_median"].values, marker=mk, lw=1.9, ls=ls, color=col, label=mlbl)
                ax_snr.fill_between(x, mdf["snr_q16"].values, mdf["snr_q84"].values, color=col, alpha=0.12)

            ax_pow.axhline(0.01, color="gray", ls=":", lw=1.0)
            ax_pow.set_ylim(-0.03, 1.04)
            ax_pow.set_ylabel("Empirical Power at $\\alpha = 10^{-2}$", fontsize=10.5)
            ax_pow.set_title(f"({'a' if col_idx==0 else 'b'}) {sig_title}: Calibrated Power ($10^4$ Nulls)", fontsize=10.8)
            ax_pow.grid(True, alpha=0.25)

            ax_snr.set_xlabel("Injected Signal Fraction $N_S / N_B$ [%]", fontsize=10.5)
            ax_snr.set_ylabel("Test Statistic Shift $\\mathrm{SNR}$ ($[q_{16}, q_{84}]$)", fontsize=10.5)
            ax_snr.set_title(f"({'c' if col_idx==0 else 'd'}) {sig_title}: Distribution-Free $\\mathrm{{SNR}}$", fontsize=10.8)
            ax_snr.grid(True, alpha=0.25)

        axes[0, 0].legend(frameon=True, fontsize=7.4, loc="lower right", ncol=2)
        fig.tight_layout()
        save_both(fig, "fig_nplm_significance_curves")
        plt.close(fig)

    # Calibration + Rate-vs-Shape figure
    cal_npz = NPLM_DIR / "highstats_calibrated" / "ae_transformer_d32_highstats.npz"
    if cal_npz.exists() and sh_csv.exists() and ad_csv.exists():
        fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.5))
        d_cal = np.load(cal_npz)
        t_cal = d_cal["t_cal"]
        t_val = d_cal["t_val"]
        ax0 = axes[0]
        ax0.hist(t_cal, bins=45, density=True, alpha=0.50, color="#1f77b4", label=f"Calibration Nulls ($N={len(t_cal):,d}$)")
        ax0.hist(t_val, bins=45, density=True, histtype="step", lw=1.8, color="black", label=f"Independent Val Nulls ($N={len(t_val):,d}$)")
        thr_1e2 = float(np.sort(t_cal)[int(np.ceil((len(t_cal) + 1) * 0.99)) - 1])
        thr_1e3 = float(np.sort(t_cal)[int(np.ceil((len(t_cal) + 1) * 0.999)) - 1])
        ax0.axvline(thr_1e2, color="#ff7f0e", ls="--", lw=1.6, label=f"$\\alpha=10^{{-2}}$ ($t={thr_1e2:.1f}$)")
        ax0.axvline(thr_1e3, color="#d62728", ls=":", lw=1.6, label=f"$\\alpha=10^{{-3}}$ ($t={thr_1e3:.1f}$)")
        for frac_k, col_k, lbl_k in [("higgs_portal_v2_f0.005", "#2ca02c", "0.5% HP v2 ($H_1$)"), ("higgs_portal_v2_f0.01", "#9467bd", "1.0% HP v2 ($H_1$)")]:
            if frac_k in d_cal.files:
                ax0.hist(d_cal[frac_k], bins=24, density=True, alpha=0.40, color=col_k, label=lbl_k)
        ax0.set_xlabel("NPLM Test Statistic $t$", fontsize=11)
        ax0.set_ylabel("Probability Density", fontsize=11)
        ax0.set_title("(a) $10{,}000$-Toy Empirical Null Calibration (ae_transformer_d32)", fontsize=10.5)
        ax0.legend(frameon=True, fontsize=7.6, loc="upper right")
        ax0.grid(True, alpha=0.25)

        ax1 = axes[1]
        df_sh = pd.read_csv(sh_csv)
        df_ad = pd.read_csv(ad_csv)
        for mname, col, lbl in [
            ("ae_transformer_d64", "#9467bd", "Transformer Summary AE (D=64)"),
            ("raw_summary_d28", "#111111", "Direct 28-D Summary (No NN)"),
            ("ssl_kinematic_d64", "#2ca02c", "Continuous SSL Kinematic (D=64)"),
            ("supcon_d64", "#1f77b4", "SupCon (D=64)"),
            ("random_init_d64", "#7f7f7f", "Random Transformer (D=64)"),
        ]:
            s1 = df_sh[(df_sh["model"] == mname) & (df_sh["signal"] == "higgs_portal_v2")].sort_values("frac")
            a1 = df_ad[(df_ad["model"] == mname) & (df_ad["signal"] == "higgs_portal_v2")].sort_values("frac")
            if not s1.empty:
                ax1.plot(s1["frac"] * 100.0, np.maximum(s1["t_median"] - s1["null_mean"], 1.0),
                         marker="o", lw=1.9, ls="-", color=col, label=f"{lbl} [Shape]")
            if not a1.empty:
                ax1.plot(a1["frac"] * 100.0, np.maximum(a1["t_median"] - a1["null_mean"], 1.0),
                         marker="s", lw=1.6, ls="--", color=col)
        fracs = np.array([0.005, 0.01, 0.025, 0.05])
        n_b = 100_000.0
        n_s = fracs * n_b
        t_rate = 2.0 * ((n_b + n_s) * np.log(1.0 + n_s / n_b) - n_s)
        ax1.plot(fracs * 100.0, t_rate, color="crimson", ls=":", lw=2.3, label="Pure Rate Excess $t_{\\mathrm{rate}} \\approx N_S^2/N_B$")
        ax1.set_yscale("log")
        ax1.set_xlabel("Injected Signal Fraction $N_S / N_B$ [%]", fontsize=11)
        ax1.set_ylabel("Excess Test Statistic $\\Delta t = t_{\\mathrm{med}} - \\mu_{\\mathrm{null}}$", fontsize=11)
        ax1.set_title("(b) Shape-Only (solid) vs Additive Rate+Shape (dashed)", fontsize=10.5)
        ax1.legend(frameon=True, fontsize=7.4, loc="upper left")
        ax1.grid(True, alpha=0.25)

        fig.tight_layout()
        save_both(fig, "fig_nplm_calibration_and_rate")
        plt.close(fig)

    # Legacy v1 vs Corrected v2 figure
    if sh_csv.exists():
        df_sh = pd.read_csv(sh_csv)
        fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.5), sharey=True)
        models_v = [
            ("ae_mlp_d32", "MLP Summary AE (D=32)", "#d62728", "-"),
            ("ae_transformer_d32", "Transformer Summary AE (D=32)", "#9467bd", "-"),
            ("raw_summary_d28", "Direct 28-D Summary (No NN)", "#111111", "-."),
            ("vqvae_code_d8", "VQ-VAE Pre-VQ Pool (D=8)", "#8c564b", "-"),
            ("ssl_tokens_d8", "Tokenized SSL (D=8)", "#ff7f0e", "-"),
            ("ssl_kinematic_d32", "Continuous SSL Kinematic (D=32)", "#2ca02c", "-"),
            ("supcon_d64", "SupCon (D=64)", "#1f77b4", "--"),
        ]
        for ax, (sig_name, stitle) in zip(axes, [
            ("higgs_portal_v1", "(a) Legacy Higgs Portal v1 (Prompt-Vertex Bug)"),
            ("higgs_portal_v2", "(b) Corrected Higgs Portal v2 (Displaced LLP)"),
        ]):
            sub = df_sh[df_sh["signal"] == sig_name]
            for mname, mlbl, col, ls in models_v:
                mdf = sub[sub["model"] == mname].sort_values("frac")
                if mdf.empty:
                    continue
                ax.plot(mdf["frac"] * 100.0, mdf["snr"], marker="o", lw=2.0, ls=ls, color=col, label=mlbl)
            ax.axhline(3.0, color="gray", ls=":", lw=1.1)
            ax.set_xlabel("Injected Signal Fraction $N_S / N_B$ [%]", fontsize=11)
            ax.set_ylabel("Median Test Statistic Shift $\\mathrm{SNR}$", fontsize=11)
            ax.set_title(stitle, fontsize=11)
            ax.grid(True, alpha=0.25)
        axes[0].legend(frameon=True, fontsize=7.8, loc="upper left")
        fig.tight_layout()
        save_both(fig, "fig_nplm_v1_vs_v2")
        plt.close(fig)


def plot_physics_controls_2x2_and_geometry():
    """
    NEW Figure 8 (fig_physics_controls_2x2.pdf):
      (a) Truth-to-Reco tracking efficiency vs production radius r_perp across all 4 2x2 arms
      (b) 2x2 Factorial + N_tracks-Matched + (d0, z0)=0 Masking Decomposition at 1% injection
      (c) Isotropic StandardScaler vs Regularized ZCA Covariance Whitening on R
    """
    eff_json = CTRL_DIR / "hp_2x2_truth_efficiency.json"
    c2x2_csv = CTRL_DIR / "hp_2x2_factorial_results.csv"
    mask_csv = CTRL_DIR / "impact_param_zeroed_ablation.csv"
    geom_csv = NPLM_DIR / "geometry_ablation" / "whitening_and_bandwidth_ablation.csv"

    if not (eff_json.exists() and c2x2_csv.exists() and mask_csv.exists()):
        return

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8))

    # Panel (a): Truth tracking efficiency vs r_perp
    with open(eff_json) as f:
        eff_data = json.load(f)
    ax0 = axes[0]
    arm_styles = [
        ("v1_prompt_2T", "v1 Prompt + 2.0T Seed", "#d62728", ":", "o"),
        ("v1_prompt_3T", "v1 Prompt + 3.0T Seed", "#ff9896", "--", "s"),
        ("v2_displaced_2T", "v2 Displaced + 2.0T Seed (SM-matched)", "#ff7f0e", "-", "o"),
        ("v2_displaced_3T", "v2 Displaced + 3.0T Seed", "#1f77b4", "-.", "^"),
    ]
    bin_keys = ["[0,1)", "[1,30)", "[30,70)", "[70,150)", "[150,500)", "[500,100000)"]
    bin_labels = ["<1", "1-30", "30-70", "70-150", "150-500", ">500"]
    x_pos = np.arange(len(bin_labels))
    for aname, albl, col, ls, mk in arm_styles:
        if aname not in eff_data:
            continue
        bdict = eff_data[aname]["eff_vs_r_perp_all_charged_pt1GeV"]
        ys = [bdict[bk]["eff"] if (bk in bdict and bdict[bk]["n_truth"] >= 20) else np.nan for bk in bin_keys]
        ax0.plot(x_pos, ys, marker=mk, lw=2.0, ls=ls, color=col, label=albl)
    ax0.set_xticks(x_pos)
    ax0.set_xticklabels(bin_labels, rotation=22, fontsize=8.5)
    ax0.set_ylim(-0.03, 0.72)
    ax0.set_xlabel("Truth Production Radius $r_{xy} = \\sqrt{v_x^2 + v_y^2}$ [mm]", fontsize=10.2)
    ax0.set_ylabel("Charged Track Finding Efficiency ($p_T > 1$ GeV)", fontsize=10.2)
    ax0.set_title("(a) ACTS Inside-Out Efficiency vs Decay Radius", fontsize=10.5)
    ax0.legend(frameon=True, fontsize=7.4, loc="upper right")
    ax0.grid(True, alpha=0.25)

    # Panel (b): 2x2 + MultMatched + (d0,z0)=0 decomposition at 1% signal injection
    df_2x2 = pd.read_csv(c2x2_csv)
    df_msk = pd.read_csv(mask_csv)
    ax1 = axes[1]
    rep_m = [
        ("raw_summary_d28", "Direct 28D"),
        ("ae_mlp_d32", "MLP AE"),
        ("ae_transformer_d32", "Transf. AE"),
        ("vqvae_code_d8", "VQ Code"),
        ("ssl_tokens_d8", "SSL Tok-8"),
        ("ssl_kinematic_d32", "SSL Kin-32"),
        ("supcon_d64", "SupCon-64"),
    ]
    rep_m = [(m, l) for m, l in rep_m if m in set(df_2x2["model"])]
    xm = np.arange(len(rep_m))
    w = 0.18

    def get_vals(df_src, arm_n, mm_flag):
        vals = []
        for m, _ in rep_m:
            r = df_src[(df_src["model"] == m) & (df_src["arm"] == arm_n) & (df_src["mult_matched"] == mm_flag) & (np.isclose(df_src["frac"], 0.01))]
            vals.append(float(r["snr_median"].iloc[0]) if not r.empty else 0.0)
        return vals

    ax1.bar(xm - 1.5 * w, get_vals(df_2x2, "v1_prompt_3T", False), width=w, color="#d62728", alpha=0.8, label="v1 Prompt + 3T (B-field only)")
    ax1.bar(xm - 0.5 * w, get_vals(df_2x2, "v2_displaced_2T", False), width=w, color="#1f77b4", alpha=0.9, label="v2 Displaced + 2T (SM-matched)")
    ax1.bar(xm + 0.5 * w, get_vals(df_2x2, "v2_displaced_2T", True), width=w, color="#2ca02c", alpha=0.85, label="v2 Displaced + 2T ($N_{\\mathrm{trk}}$-matched)")
    ax1.bar(xm + 1.5 * w, get_vals(df_msk, "v2_displaced_2T_d0z0_zeroed", False), width=w, color="#ff7f0e", alpha=0.85, label="v2 Displaced + 2T ($d_0,z_0=0$)")
    ax1.set_xticks(xm)
    ax1.set_xticklabels([l for _, l in rep_m], rotation=20, ha="right", fontsize=8.5)
    ax1.set_ylabel("Shape-Only NPLM $\\mathrm{SNR}$ at $1\\%$ Injection", fontsize=10.2)
    ax1.set_title("(b) Displaced Vertex vs B-Field & Multiplicity Controls", fontsize=10.5)
    ax1.legend(frameon=True, fontsize=7.2, loc="upper right")
    ax1.grid(True, axis="y", alpha=0.25)

    # Panel (c): StandardScaler vs ZCA Whitening on R
    ax2 = axes[2]
    if geom_csv.exists():
        df_g = pd.read_csv(geom_csv)
        geom_candidates = [
            ("raw_summary_d28", "Direct 28D"),
            ("ae_mlp_d32", "MLP AE"),
            ("ae_transformer_d32", "Transf. AE"),
            ("vqvae_code_d8", "VQ Code"),
            ("ssl_tokens_d32", "SSL Tok-32"),
            ("ssl_kinematic_d64", "SSL Kin-64"),
            ("supcon_d64", "SupCon-64"),
        ]
        gm = [m for m, _ in geom_candidates if m in set(df_g["model"])]
        gml = [l for m, l in geom_candidates if m in set(df_g["model"])]
        xg = np.arange(len(gm))
        wg = 0.34
        v_std, v_zca = [], []
        for m in gm:
            r_s = df_g[(df_g["model"] == m) & (df_g["transform"] == "standard") & (np.isclose(df_g["sigma_quantile"], 0.90)) & (df_g["signal"] == "higgs_portal_v2") & (np.isclose(df_g["frac"], 0.01))]
            r_z = df_g[(df_g["model"] == m) & (df_g["transform"] == "zca") & (np.isclose(df_g["sigma_quantile"], 0.90)) & (df_g["signal"] == "higgs_portal_v2") & (np.isclose(df_g["frac"], 0.01))]
            v_std.append(float(r_s["snr_median"].iloc[0]) if not r_s.empty else 0.0)
            v_zca.append(float(r_z["snr_median"].iloc[0]) if not r_z.empty else 0.0)
        ax2.bar(xg - wg / 2, v_std, width=wg, color="#1f77b4", alpha=0.85, label="Diagonal StandardScaler on $R$")
        ax2.bar(xg + wg / 2, v_zca, width=wg, color="#9467bd", alpha=0.85, label="Full ZCA Covariance Whitening on $R$")
        ax2.set_xticks(xg)
        ax2.set_xticklabels(gml, rotation=20, ha="right", fontsize=8.5)
        ax2.set_ylabel("HP v2 Shape-Only $\\mathrm{SNR}$ at $1\\%$ Injection", fontsize=10.2)
        ax2.set_title("(c) Isotropic vs ZCA-Whitened Kernel Geometry", fontsize=10.5)
        ax2.legend(frameon=True, fontsize=7.6, loc="upper right")
        ax2.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    save_both(fig, "fig_physics_controls_2x2")
    plt.close(fig)


def main():
    if (VQVAE_V1_DIR / "vqvae_fidelity_panel.pdf").exists():
        PAPER_IMG_DIR.mkdir(parents=True, exist_ok=True)
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        for ext in ["pdf", "png"]:
            shutil.copy2(VQVAE_V1_DIR / f"vqvae_fidelity_panel.{ext}", PAPER_IMG_DIR / f"fig_vqvae_fidelity.{ext}")
            shutil.copy2(VQVAE_V1_DIR / f"vqvae_fidelity_panel.{ext}", FIG_DIR / f"fig_vqvae_fidelity.{ext}")
    plot_hp_v1_vs_v2_displaced()
    plot_latent_pca_grid()
    plot_probe_scaling_and_label_efficiency()
    plot_nplm_highstats_and_rate()
    plot_physics_controls_2x2_and_geometry()


if __name__ == "__main__":
    main()
