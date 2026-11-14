#!/usr/bin/env python3
"""
Workstream 1 (Paired Probes + True Supervised-From-Scratch Low-Label Baseline):
  1. Evaluates ALL models in results/embeddings_v2/ on the exact same frozen 1,050,000-event
     eval_downstream_1050k split (verifying eval_uid identity across every model), eliminating
     the mixed 2k-vs-20k HP_v2 test-set discrepancy.
  2. Computes paired test-set bootstrap standard errors (B=100 resamples) on accuracy and ROC-AUC.
  3. Trains supervised_scratch_d64: an end-to-end 8-layer/8-head Pre-LN GELU SetTransformer
     initialized from random weights and trained strictly on the budgeted N_labeled in {100, 1e3, 1e4, 1e5}
     subsets (80% train / 20% val, track scaler fit strictly on the 80% train subset, zero 900k pretraining!)
     across 5 seeds, evaluated on the exact same held-out sm_te test split.
"""
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path
from phase2_nplm_calibrated import gpfs_lock

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, label_binarize

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from common import SetTransformerEncoder, fit_track_scaler, load_events_fast, raw6_to_feat7_np
from evaluate_probes import AffineLinearProbe, NonlinearMLPProbe, make_stratified_split

DATA_DIR = Path("/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/data")
CANON_EVAL_PARQUET = DATA_DIR / "eval_downstream_1050k.parquet"
RESULTS_BASE = Path("/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/results")
EMBED_V2_DIR = RESULTS_BASE / "embeddings_v2"
PROBE_V2_DIR = RESULTS_BASE / "probes_v2"


def bootstrap_se(y_true: np.ndarray, probs: np.ndarray, preds: np.ndarray, n_classes: int, n_boot: int = 100, seed: int = 123) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    accs = []
    aucs = []
    if n_classes > 2:
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        accs.append(float((preds[idx] == y_true[idx]).mean()))
        if n_classes == 2:
            aucs.append(float(roc_auc_score(y_true[idx], probs[idx, 1])))
        else:
            aucs.append(float(roc_auc_score(y_bin[idx], probs[idx], average="macro", multi_class="ovr")))
    return float(np.std(accs, ddof=1)), float(np.std(aucs, ddof=1))


def train_probe_v2(
    z_train: np.ndarray,
    y_train: np.ndarray,
    z_val: np.ndarray,
    y_val: np.ndarray,
    z_test: np.ndarray,
    y_test: np.ndarray,
    n_classes: int,
    device: torch.device,
    probe_type: str = "linear",
    epochs: int = 40,
    batch_size: int = 2048,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
    compute_boot: bool = True,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    scaler = StandardScaler().fit(z_train)
    x_tr = torch.from_numpy(scaler.transform(z_train).astype(np.float32)).to(device)
    y_tr = torch.from_numpy(y_train.astype(np.int64)).to(device)
    x_va = torch.from_numpy(scaler.transform(z_val).astype(np.float32)).to(device)
    y_va = torch.from_numpy(y_val.astype(np.int64)).to(device)
    x_te = torch.from_numpy(scaler.transform(z_test).astype(np.float32)).to(device)

    in_dim = z_train.shape[1]
    model = AffineLinearProbe(in_dim, n_classes).to(device) if probe_type == "linear" else NonlinearMLPProbe(in_dim, n_classes).to(device)

    counts = np.bincount(y_train, minlength=n_classes).astype(np.float32)
    weights = counts.sum() / (n_classes * np.maximum(counts, 1.0))
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device))

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    eff_bs = min(batch_size, max(32, len(x_tr)))
    n_batches = max(1, (len(x_tr) + eff_bs - 1) // eff_bs)

    best_val_loss = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(x_tr), device=device)
        for b in range(n_batches):
            idx = perm[b * eff_bs : (b + 1) * eff_bs]
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(x_tr[idx]), y_tr[idx])
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(x_va), y_va).item())
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits_te = model(x_te)
        probs_te = torch.softmax(logits_te, dim=-1).cpu().numpy()
        preds_te = logits_te.argmax(dim=-1).cpu().numpy()

    acc = float(accuracy_score(y_test, preds_te))
    if n_classes == 2:
        macro_auc = float(roc_auc_score(y_test, probs_te[:, 1]))
        per_class_auc = {"binary": macro_auc}
    else:
        y_bin = label_binarize(y_test, classes=list(range(n_classes)))
        macro_auc = float(roc_auc_score(y_bin, probs_te, average="macro", multi_class="ovr"))
        per_class_auc = {str(c): float(roc_auc_score(y_bin[:, c], probs_te[:, c])) for c in range(n_classes)}

    acc_se, auc_se = bootstrap_se(y_test, probs_te, preds_te, n_classes) if compute_boot else (np.nan, np.nan)
    cm = confusion_matrix(y_test, preds_te, labels=list(range(n_classes))).tolist()
    return {
        "accuracy": acc,
        "accuracy_se": acc_se,
        "macro_auc": macro_auc,
        "macro_auc_se": auc_se,
        "per_class_auc": per_class_auc,
        "confusion_matrix": cm,
        "n_test": int(len(y_test)),
    }


def run_paired_probes(device: torch.device, models_filter: list[str] | None = None):
    PROBE_V2_DIR.mkdir(parents=True, exist_ok=True)
    npz_files = sorted(glob.glob(str(EMBED_V2_DIR / "*.npz")))
    if models_filter:
        npz_files = [p for p in npz_files if Path(p).stem in models_filter]
    assert len(npz_files) > 0, f"No .npz files in {EMBED_V2_DIR}"

    ref = np.load(npz_files[0])
    ref_uid = ref["eval_uid"]
    ref_labels = ref["labels"]
    assert len(ref_labels) == 1_050_000, f"Expected 1,050,000 events, got {len(ref_labels)}"

    sm_indices = np.where(np.isin(ref_labels, [0, 1, 2]))[0]
    sm_labels = ref_labels[sm_indices]
    sm_tr, sm_va, sm_te = make_stratified_split(sm_labels, 0.70, 0.10, seed=42)

    c5_indices = np.where(np.isin(ref_labels, [0, 1, 2, 3, 4]))[0]
    c5_labels = ref_labels[c5_indices]
    c5_tr, c5_va, c5_te = make_stratified_split(c5_labels, 0.70, 0.10, seed=42)

    rng_bin = np.random.default_rng(42)
    sm_sub = rng_bin.choice(sm_indices, size=60_000, replace=False)
    binary_tasks = {}
    for bsm_lbl, bsm_name in [(3, "sm_vs_higgs_portal_v2"), (4, "sm_vs_hidden_valley"), (5, "sm_vs_higgs_portal_v1")]:
        bsm_idx = np.where(ref_labels == bsm_lbl)[0]
        assert len(bsm_idx) == 100_000, f"Expected 100,000 for label {bsm_lbl}, got {len(bsm_idx)}"
        idx_all = np.concatenate([sm_sub, bsm_idx])
        y_bin = np.concatenate([np.zeros(len(sm_sub), dtype=np.int64), np.ones(len(bsm_idx), dtype=np.int64)])
        b_tr, b_va, b_te = make_stratified_split(y_bin, 0.70, 0.10, seed=42 + bsm_lbl)
        binary_tasks[bsm_name] = (idx_all, y_bin, b_tr, b_va, b_te)

    summary_rows = []
    curve_rows = []

    curve_models = {
        "supcon_d64", "supervised_e2e_d64", "ssl_kinematic_d64", "ssl_kinematic_drop_d64",
        "ssl_tokens_d64", "ssl_legacy_d64", "ae_transformer_d64", "ae_mlp_d64",
        "random_init_d64", "raw_summary_d28", "phys_handcrafted_d6",
    }

    for fpath in npz_files:
        mname = Path(fpath).stem
        det_json = PROBE_V2_DIR / f"{mname}_probe_detail.json"
        if det_json.exists() and (mname not in curve_models):
            with open(det_json) as f:
                det = json.load(f)
            prov = det.get("provenance", {})
            e2e = det.get("sm3_e2e_head")
            summary_rows.append({
                "model": mname,
                "latent_dim": det["latent_dim"],
                "erank_sm": prov.get("erank_sm", np.nan),
                "sm3_linear_acc": det["sm3_linear"]["accuracy"],
                "sm3_linear_acc_se": det["sm3_linear"]["accuracy_se"],
                "sm3_linear_auc": det["sm3_linear"]["macro_auc"],
                "sm3_linear_auc_se": det["sm3_linear"]["macro_auc_se"],
                "sm3_mlp_acc": det["sm3_mlp"]["accuracy"],
                "sm3_mlp_auc": det["sm3_mlp"]["macro_auc"],
                "sm3_e2e_head_acc": e2e["accuracy"] if e2e else np.nan,
                "sm3_e2e_head_auc": e2e["macro_auc"] if e2e else np.nan,
                "c5_linear_acc": det["c5_linear"]["accuracy"],
                "c5_linear_acc_se": det["c5_linear"]["accuracy_se"],
                "c5_linear_auc": det["c5_linear"]["macro_auc"],
                "ood_hp_v2_auc": det["binary_ood_linear"]["sm_vs_higgs_portal_v2"]["macro_auc"],
                "ood_hp_v2_auc_se": det["binary_ood_linear"]["sm_vs_higgs_portal_v2"]["macro_auc_se"],
                "ood_hv_auc": det["binary_ood_linear"]["sm_vs_hidden_valley"]["macro_auc"],
                "ood_hv_auc_se": det["binary_ood_linear"]["sm_vs_hidden_valley"]["macro_auc_se"],
                "ood_hp_v1_auc": det["binary_ood_linear"]["sm_vs_higgs_portal_v1"]["macro_auc"],
                "ood_hp_v1_auc_se": det["binary_ood_linear"]["sm_vs_higgs_portal_v1"]["macro_auc_se"],
            })
            sum_csv = PROBE_V2_DIR / "probe_summary.csv"
            with gpfs_lock(sum_csv):
                df_sum = pd.DataFrame(summary_rows)
                if sum_csv.exists():
                    old = pd.read_csv(sum_csv)
                    old = old[~old["model"].isin(df_sum["model"])]
                    df_sum = pd.concat([old, df_sum], ignore_index=True).sort_values("model")
                df_sum.to_csv(sum_csv, index=False)
            print(f"[probe_v2] Reused cached {det_json.name}", flush=True)
            continue
        t0 = time.time()
        data = np.load(fpath)
        assert np.array_equal(data["eval_uid"], ref_uid), f"eval_uid mismatch in {mname}!"
        assert np.array_equal(data["labels"], ref_labels), f"labels mismatch in {mname}!"
        z = data["z"].astype(np.float32)
        D = z.shape[1]

        z_sm = z[sm_indices]
        res_sm3_lin = train_probe_v2(z_sm[sm_tr], sm_labels[sm_tr], z_sm[sm_va], sm_labels[sm_va], z_sm[sm_te], sm_labels[sm_te], 3, device, "linear")
        res_sm3_mlp = train_probe_v2(z_sm[sm_tr], sm_labels[sm_tr], z_sm[sm_va], sm_labels[sm_va], z_sm[sm_te], sm_labels[sm_te], 3, device, "mlp")

        z_c5 = z[c5_indices]
        res_c5_lin = train_probe_v2(z_c5[c5_tr], c5_labels[c5_tr], z_c5[c5_va], c5_labels[c5_va], z_c5[c5_te], c5_labels[c5_te], 5, device, "linear")

        bin_results = {}
        for bname, (idx_all, y_bin, b_tr, b_va, b_te) in binary_tasks.items():
            z_b = z[idx_all]
            bin_results[bname] = train_probe_v2(z_b[b_tr], y_bin[b_tr], z_b[b_va], y_bin[b_va], z_b[b_te], y_bin[b_te], 2, device, "linear", batch_size=1024)

        e2e_head_res = None
        if "sm3_logits" in data.files:
            logits_te = data["sm3_logits"][sm_indices][sm_te].astype(np.float64)
            logits_te -= logits_te.max(axis=-1, keepdims=True)
            probs_e2e = np.exp(logits_te) / np.exp(logits_te).sum(axis=-1, keepdims=True)
            preds_e2e = probs_e2e.argmax(axis=-1)
            y_bin_e2e = label_binarize(sm_labels[sm_te], classes=[0, 1, 2])
            e2e_head_res = {
                "accuracy": float(accuracy_score(sm_labels[sm_te], preds_e2e)),
                "macro_auc": float(roc_auc_score(y_bin_e2e, probs_e2e, average="macro", multi_class="ovr")),
            }

        prov = json.loads(str(data["provenance_json"])) if "provenance_json" in data.files else {}
        row = {
            "model": mname,
            "latent_dim": D,
            "erank_sm": prov.get("erank_sm", np.nan),
            "sm3_linear_acc": res_sm3_lin["accuracy"],
            "sm3_linear_acc_se": res_sm3_lin["accuracy_se"],
            "sm3_linear_auc": res_sm3_lin["macro_auc"],
            "sm3_linear_auc_se": res_sm3_lin["macro_auc_se"],
            "sm3_mlp_acc": res_sm3_mlp["accuracy"],
            "sm3_mlp_auc": res_sm3_mlp["macro_auc"],
            "sm3_e2e_head_acc": e2e_head_res["accuracy"] if e2e_head_res else np.nan,
            "sm3_e2e_head_auc": e2e_head_res["macro_auc"] if e2e_head_res else np.nan,
            "c5_linear_acc": res_c5_lin["accuracy"],
            "c5_linear_acc_se": res_c5_lin["accuracy_se"],
            "c5_linear_auc": res_c5_lin["macro_auc"],
            "ood_hp_v2_auc": bin_results["sm_vs_higgs_portal_v2"]["macro_auc"],
            "ood_hp_v2_auc_se": bin_results["sm_vs_higgs_portal_v2"]["macro_auc_se"],
            "ood_hv_auc": bin_results["sm_vs_hidden_valley"]["macro_auc"],
            "ood_hv_auc_se": bin_results["sm_vs_hidden_valley"]["macro_auc_se"],
            "ood_hp_v1_auc": bin_results["sm_vs_higgs_portal_v1"]["macro_auc"],
            "ood_hp_v1_auc_se": bin_results["sm_vs_higgs_portal_v1"]["macro_auc_se"],
        }
        summary_rows.append(row)
        print(
            f"[probe_v2] {mname:26s} (D={D:3d}) | SM3={row['sm3_linear_acc']:.4f} (MLP {row['sm3_mlp_acc']:.4f}) | "
            f"C5={row['c5_linear_acc']:.4f} | HP_v2={row['ood_hp_v2_auc']:.4f} | HV={row['ood_hv_auc']:.4f} | HP_v1={row['ood_hp_v1_auc']:.4f} ({time.time()-t0:.1f}s)",
            flush=True,
        )

        detail = {
            "model": mname,
            "latent_dim": D,
            "provenance": prov,
            "sm3_linear": res_sm3_lin,
            "sm3_mlp": res_sm3_mlp,
            "sm3_e2e_head": e2e_head_res,
            "c5_linear": res_c5_lin,
            "binary_ood_linear": bin_results,
        }
        with open(PROBE_V2_DIR / f"{mname}_probe_detail.json", "w") as f:
            json.dump(detail, f, indent=2)

        if mname in curve_models:
            pool_idx = np.concatenate([sm_tr, sm_va])
            for n_labeled in [100, 1000, 10000, 100000]:
                for seed in range(5):
                    sss_b = StratifiedShuffleSplit(n_splits=1, train_size=n_labeled, random_state=1000 * seed + n_labeled)
                    sub_rel, _ = next(sss_b.split(np.zeros(len(pool_idx)), sm_labels[pool_idx]))
                    sub_idx = pool_idx[sub_rel]

                    sss_tv = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=2000 * seed + n_labeled)
                    tr_sub, va_sub = next(sss_tv.split(np.zeros(len(sub_idx)), sm_labels[sub_idx]))
                    idx_tr_b = sub_idx[tr_sub]
                    idx_va_b = sub_idx[va_sub]

                    eff_bs_b = min(512, max(32, len(idx_tr_b)))
                    n_batches_b = max(1, (len(idx_tr_b) + eff_bs_b - 1) // eff_bs_b)
                    epochs_b = max(40, min(200, int(np.ceil(250 / n_batches_b))))
                    for ptype in ["linear", "mlp"]:
                        r_c = train_probe_v2(
                            z_sm[idx_tr_b], sm_labels[idx_tr_b],
                            z_sm[idx_va_b], sm_labels[idx_va_b],
                            z_sm[sm_te], sm_labels[sm_te],
                            n_classes=3, device=device, probe_type=ptype,
                            epochs=epochs_b, batch_size=eff_bs_b, seed=seed, compute_boot=False,
                        )
                        curve_rows.append({
                            "model": mname,
                            "probe_type": ptype,
                            "n_labeled": n_labeled,
                            "seed": seed,
                            "test_acc": r_c["accuracy"],
                            "test_auc": r_c["macro_auc"],
                        })

        sum_csv = PROBE_V2_DIR / "probe_summary.csv"
        with gpfs_lock(sum_csv):
            df_sum = pd.DataFrame(summary_rows)
            if sum_csv.exists():
                old = pd.read_csv(sum_csv)
                old = old[~old["model"].isin(df_sum["model"])]
                df_sum = pd.concat([old, df_sum], ignore_index=True).sort_values("model")
            df_sum.to_csv(sum_csv, index=False)

        if curve_rows:
            c_csv = PROBE_V2_DIR / "label_efficiency_curves.csv"
            with gpfs_lock(c_csv):
                df_c = pd.DataFrame(curve_rows)
                if c_csv.exists():
                    old_c = pd.read_csv(c_csv)
                    keys_new = set(zip(df_c["model"], df_c["probe_type"], df_c["n_labeled"], df_c["seed"]))
                    keep = [
                        (r["model"], r["probe_type"], int(r["n_labeled"]), int(r["seed"])) not in keys_new
                        for _, r in old_c.iterrows()
                    ]
                    df_c = pd.concat([old_c[keep], df_c], ignore_index=True)
                df_c.to_csv(c_csv, index=False)
    print(f"[probe_v2] Completed saving probe_summary.csv and label_efficiency_curves.csv")


def run_supervised_from_scratch_curves(device: torch.device):
    """
    Trains the exact 8-layer/8-head Pre-LN GELU SetTransformer + linear 3-class head
    END-TO-END FROM RANDOM INITIALIZATION strictly on the budgeted N_labeled in {100, 1e3, 1e4, 1e5}
    SM subsets (80% train / 20% val; track scaler fit ONLY on the 80% train tracks; zero 900k pretraining!),
    evaluated on the exact same held-out sm_te test split (N=150,000 SM events).
    """
    PROBE_V2_DIR.mkdir(parents=True, exist_ok=True)
    eval_data = load_events_fast(CANON_EVAL_PARQUET)
    labels = eval_data["labels"]
    sm_indices = np.where(np.isin(labels, [0, 1, 2]))[0]

    raw6_sm = eval_data["raw6"][sm_indices, :128, :]
    mask_sm = eval_data["mask"][sm_indices, :128]
    lengths_sm = mask_sm.sum(axis=1).astype(np.int32)
    sm_labels = labels[sm_indices]
    del eval_data

    feat7_sm = raw6_to_feat7_np(raw6_sm, mask_sm)
    del raw6_sm

    sm_tr, sm_va, sm_te = make_stratified_split(sm_labels, 0.70, 0.10, seed=42)
    pool_idx = np.concatenate([sm_tr, sm_va])

    out_scratch = PROBE_V2_DIR / "supervised_scratch_curves.csv"
    rows = []
    done_pairs = set()
    if out_scratch.exists():
        old_sc = pd.read_csv(out_scratch)
        rows = old_sc.to_dict("records")
        done_pairs = set(zip(old_sc["n_labeled"].astype(int), old_sc["seed"].astype(int)))
        print(f"[supervised_scratch_d64] Loaded {len(done_pairs)} completed (n_labeled, seed) pairs from {out_scratch}")

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

            # Fit track scaler strictly on the budgeted 80% labeled training tracks!
            sc = fit_track_scaler(feat7_sm[idx_tr], mask_sm[idx_tr])
            mean = sc.mean_.astype(np.float32).reshape(1, 1, 7)
            scale = np.maximum(sc.scale_.astype(np.float32), 1e-6).reshape(1, 1, 7)

            x_tr = ((feat7_sm[idx_tr] - mean) / scale)
            x_tr[~mask_sm[idx_tr]] = 0.0
            x_va = ((feat7_sm[idx_va] - mean) / scale)
            x_va[~mask_sm[idx_va]] = 0.0

            x_tr_g = torch.from_numpy(x_tr).to(device)
            m_tr_g = torch.from_numpy(mask_sm[idx_tr]).to(device)
            l_tr_g = torch.from_numpy(lengths_sm[idx_tr]).to(device)
            y_tr_g = torch.from_numpy(sm_labels[idx_tr]).to(device)

            x_va_g = torch.from_numpy(x_va).to(device)
            m_va_g = torch.from_numpy(mask_sm[idx_va]).to(device)
            y_va_g = torch.from_numpy(sm_labels[idx_va]).to(device)

            enc = SetTransformerEncoder(latent_dim=64, norm_first=True, activation="gelu", use_out_norm=True).to(device)
            head = nn.Linear(64, 3).to(device)
            params = list(enc.parameters()) + list(head.parameters())
            opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

            best_va_loss = float("inf")
            best_enc = {k: v.detach().clone() for k, v in enc.state_dict().items()}
            best_head = {k: v.detach().clone() for k, v in head.state_dict().items()}

            n_tr = len(idx_tr)
            n_batches = max(1, (n_tr + bs - 1) // bs)
            for _ in range(epochs):
                enc.train()
                head.train()
                perm = torch.randperm(n_tr, device=device)
                for b in range(n_batches):
                    bi = perm[b * bs : (b + 1) * bs]
                    mt = max(int(l_tr_g[bi].max().item()), 1)
                    opt.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                        logits = head(enc(x_tr_g[bi, :mt, :], m_tr_g[bi, :mt]))
                        loss = F.cross_entropy(logits, y_tr_g[bi])
                    loss.backward()
                    nn.utils.clip_grad_norm_(params, 1.0)
                    opt.step()
                sched.step()

                enc.eval()
                head.eval()
                va_losses = []
                with torch.no_grad():
                    for vs in range(0, len(idx_va), 1024):
                        ve = min(vs + 1024, len(idx_va))
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                            v_log = head(enc(x_va_g[vs:ve], m_va_g[vs:ve]))
                            va_losses.append(float(F.cross_entropy(v_log, y_va_g[vs:ve], reduction="sum").item()))
                va_loss = sum(va_losses) / len(idx_va)
                if va_loss < best_va_loss:
                    best_va_loss = va_loss
                    best_enc = {k: v.detach().clone() for k, v in enc.state_dict().items()}
                    best_head = {k: v.detach().clone() for k, v in head.state_dict().items()}

            enc.load_state_dict(best_enc)
            head.load_state_dict(best_head)
            enc.eval()
            head.eval()

            probs_list = []
            with torch.no_grad():
                for ts in range(0, len(sm_te), 2048):
                    te_idx = sm_te[ts : ts + 2048]
                    mb = mask_sm[te_idx]
                    mt = max(int(mb.sum(axis=1).max()), 1)
                    mb = mb[:, :mt]
                    xb = (feat7_sm[te_idx, :mt, :] - mean) / scale
                    xb[~mb] = 0.0
                    xt = torch.from_numpy(xb).to(device)
                    mt_g = torch.from_numpy(mb).to(device)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                        lg = head(enc(xt, mt_g)).float()
                    probs_list.append(torch.softmax(lg, dim=-1).cpu().numpy())

            probs_te = np.concatenate(probs_list, axis=0)
            preds_te = probs_te.argmax(axis=-1)
            y_te = sm_labels[sm_te]
            acc = float(accuracy_score(y_te, preds_te))
            y_bin = label_binarize(y_te, classes=[0, 1, 2])
            auc = float(roc_auc_score(y_bin, probs_te, average="macro", multi_class="ovr"))
            print(f"[supervised_scratch_d64] N_labeled={n_labeled:6d} seed={seed}: acc={acc:.4f}, auc={auc:.4f} ({time.time()-t0:.1f}s)", flush=True)

            for ptype in ["e2e_scratch", "linear", "mlp"]:
                rows.append({
                    "model": "supervised_scratch_d64",
                    "probe_type": ptype,
                    "n_labeled": n_labeled,
                    "seed": seed,
                    "test_acc": acc,
                    "test_auc": auc,
                })
            pd.DataFrame(rows).to_csv(out_scratch, index=False)

            del x_tr_g, m_tr_g, x_va_g, m_va_g
            torch.cuda.empty_cache()

    df_s = pd.DataFrame(rows)
    out_scratch = PROBE_V2_DIR / "supervised_scratch_curves.csv"
    df_s.to_csv(out_scratch, index=False)

    c_csv = PROBE_V2_DIR / "label_efficiency_curves.csv"
    with gpfs_lock(c_csv):
        if c_csv.exists():
            old_c = pd.read_csv(c_csv)
            old_c = old_c[old_c["model"] != "supervised_scratch_d64"]
            df_s = pd.concat([old_c, df_s], ignore_index=True)
        df_s.to_csv(c_csv, index=False)
    print(f"[supervised_scratch_d64] Merged into {c_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=str, required=True, choices=["probes", "scratch_curves", "both"])
    parser.add_argument("--models", type=str, nargs="*", default=None)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    if args.step in ("probes", "both"):
        run_paired_probes(device, args.models)
    if args.step in ("scratch_curves", "both"):
        run_supervised_from_scratch_curves(device)


if __name__ == "__main__":
    main()
