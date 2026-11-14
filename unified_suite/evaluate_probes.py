#!/usr/bin/env python3
"""
Validation of Learned Representations on the strictly disjoint eval_downstream split:
  1. 3-Class SM Linear Probe (ttbar vs ggf vs dihiggs): Accuracy, Macro ROC-AUC, Per-Class AUC
  2. 5-Class SM + BSM Linear Probe (ttbar, ggf, dihiggs, higgs_portal_v2, hidden_valley)
  3. Binary OOD Linear Probes (SM vs higgs_portal_v2, SM vs hidden_valley, SM vs higgs_portal_v1)
  4. Sample-Efficiency / Label-Budget Curves across N_labeled in {100, 1000, 10000, 100000}
     with strictly budgeted validation (80% train / 20% val of N_labeled, no 135k val leakage)
     for both True Linear Probe (nn.Linear) and Nonlinear MLP Probe.
"""
import argparse
import glob
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, label_binarize

RESULTS_BASE = Path("/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/results")
EMBED_DIR = RESULTS_BASE / "embeddings"
PROBE_OUT_DIR = RESULTS_BASE / "probes"


class AffineLinearProbe(nn.Module):
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class NonlinearMLPProbe(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_torch_probe(
    z_train: np.ndarray,
    y_train: np.ndarray,
    z_val: np.ndarray,
    y_val: np.ndarray,
    z_test: np.ndarray,
    y_test: np.ndarray,
    n_classes: int,
    device: torch.device,
    probe_type: str = "linear",
    epochs: int = 50,
    batch_size: int = 1024,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
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
    if probe_type == "linear":
        model = AffineLinearProbe(in_dim, n_classes).to(device)
    else:
        model = NonlinearMLPProbe(in_dim, n_classes).to(device)

    # Class-balanced cross entropy for imbalanced tasks (e.g. 5-class when HP_v2 has 10k vs 250k SM)
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
        per_class_auc = {
            str(c): float(roc_auc_score(y_bin[:, c], probs_te[:, c]))
            for c in range(n_classes)
        }
    cm = confusion_matrix(y_test, preds_te, labels=list(range(n_classes))).tolist()
    return {
        "accuracy": acc,
        "macro_auc": macro_auc,
        "per_class_auc": per_class_auc,
        "confusion_matrix": cm,
    }


def make_stratified_split(labels: np.ndarray, train_frac: float = 0.70, val_frac: float = 0.10, seed: int = 42):
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=(1.0 - train_frac), random_state=seed)
    train_idx, temp_idx = next(sss1.split(np.zeros(len(labels)), labels))
    rel_val = val_frac / (1.0 - train_frac)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=(1.0 - rel_val), random_state=seed + 1)
    val_rel, test_rel = next(sss2.split(np.zeros(len(temp_idx)), labels[temp_idx]))
    return train_idx, temp_idx[val_rel], temp_idx[test_rel]


def evaluate_all_models(device: torch.device, models_filter: list[str] | None = None):
    PROBE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    npz_files = sorted(glob.glob(str(EMBED_DIR / "*.npz")))
    if models_filter:
        npz_files = [p for p in npz_files if Path(p).stem in models_filter]
    if not npz_files:
        raise FileNotFoundError(f"No embedding files found in {EMBED_DIR}")

    ref = np.load(npz_files[0])
    ref_event_ids = ref["event_ids"]
    ref_labels = ref["labels"]
    print(f"[evaluate_probes] Loaded reference {Path(npz_files[0]).name}: N={len(ref_labels)} events")

    # Precompute shared indices for 100% paired evaluation across all models
    sm_mask = np.isin(ref_labels, [0, 1, 2])
    sm_indices = np.where(sm_mask)[0]
    sm_labels = ref_labels[sm_indices]
    sm_tr, sm_va, sm_te = make_stratified_split(sm_labels, 0.70, 0.10, seed=42)

    # 5-class SM + BSM (0: ttbar, 1: ggf, 2: dihiggs, 3: higgs_portal_v2, 4: hidden_valley)
    c5_mask = np.isin(ref_labels, [0, 1, 2, 3, 4])
    c5_indices = np.where(c5_mask)[0]
    c5_labels = ref_labels[c5_indices]
    c5_tr, c5_va, c5_te = make_stratified_split(c5_labels, 0.70, 0.10, seed=42)

    # Binary OOD probes: balanced SM subsample (50k) vs BSM channel
    rng_bin = np.random.default_rng(42)
    sm_sub = rng_bin.choice(sm_indices, size=min(60_000, len(sm_indices)), replace=False)
    binary_tasks = {}
    for bsm_lbl, bsm_name in [(3, "sm_vs_higgs_portal_v2"), (4, "sm_vs_hidden_valley"), (5, "sm_vs_higgs_portal_v1")]:
        bsm_idx = np.where(ref_labels == bsm_lbl)[0]
        if len(bsm_idx) == 0:
            continue
        idx_all = np.concatenate([sm_sub, bsm_idx])
        y_bin = np.concatenate([np.zeros(len(sm_sub), dtype=np.int64), np.ones(len(bsm_idx), dtype=np.int64)])
        b_tr, b_va, b_te = make_stratified_split(y_bin, 0.70, 0.10, seed=42 + bsm_lbl)
        binary_tasks[bsm_name] = (idx_all, y_bin, b_tr, b_va, b_te)

    summary_rows = []
    curve_rows = []

    for fpath in npz_files:
        mname = Path(fpath).stem
        data = np.load(fpath)
        assert np.array_equal(data["event_ids"], ref_event_ids), f"event_ids mismatch in {mname}!"
        assert np.array_equal(data["labels"], ref_labels), f"labels mismatch in {mname}!"
        z = data["z"].astype(np.float32)
        D = z.shape[1]
        print(f"\n=== Evaluating {mname} (D={D}) ===")

        # 1. 3-Class SM Linear Probe
        z_sm = z[sm_indices]
        res_sm3_lin = train_torch_probe(
            z_sm[sm_tr], sm_labels[sm_tr], z_sm[sm_va], sm_labels[sm_va], z_sm[sm_te], sm_labels[sm_te],
            n_classes=3, device=device, probe_type="linear", epochs=40, batch_size=2048
        )
        res_sm3_mlp = train_torch_probe(
            z_sm[sm_tr], sm_labels[sm_tr], z_sm[sm_va], sm_labels[sm_va], z_sm[sm_te], sm_labels[sm_te],
            n_classes=3, device=device, probe_type="mlp", epochs=40, batch_size=2048
        )

        # 2. 5-Class SM + BSM Linear Probe
        z_c5 = z[c5_indices]
        res_c5_lin = train_torch_probe(
            z_c5[c5_tr], c5_labels[c5_tr], z_c5[c5_va], c5_labels[c5_va], z_c5[c5_te], c5_labels[c5_te],
            n_classes=5, device=device, probe_type="linear", epochs=40, batch_size=2048
        )

        # 3. Binary OOD Linear Probes (SM vs HP_v2, SM vs HV, SM vs HP_v1)
        bin_results = {}
        for bname, (idx_all, y_bin, b_tr, b_va, b_te) in binary_tasks.items():
            z_b = z[idx_all]
            r_b = train_torch_probe(
                z_b[b_tr], y_bin[b_tr], z_b[b_va], y_bin[b_va], z_b[b_te], y_bin[b_te],
                n_classes=2, device=device, probe_type="linear", epochs=40, batch_size=1024
            )
            bin_results[bname] = r_b

        e2e_head_res = None
        if "sm3_logits" in data.files:
            logits_sm_te = data["sm3_logits"][sm_indices][sm_te].astype(np.float64)
            logits_sm_te = logits_sm_te - logits_sm_te.max(axis=-1, keepdims=True)
            probs_e2e = np.exp(logits_sm_te) / np.exp(logits_sm_te).sum(axis=-1, keepdims=True)
            preds_e2e = probs_e2e.argmax(axis=-1)
            y_bin_e2e = label_binarize(sm_labels[sm_te], classes=[0, 1, 2])
            e2e_head_res = {
                "accuracy": float(accuracy_score(sm_labels[sm_te], preds_e2e)),
                "macro_auc": float(roc_auc_score(y_bin_e2e, probs_e2e, average="macro", multi_class="ovr")),
            }

        row = {
            "model": mname,
            "latent_dim": D,
            "sm3_linear_acc": res_sm3_lin["accuracy"],
            "sm3_linear_auc": res_sm3_lin["macro_auc"],
            "sm3_mlp_acc": res_sm3_mlp["accuracy"],
            "sm3_mlp_auc": res_sm3_mlp["macro_auc"],
            "sm3_e2e_head_acc": e2e_head_res["accuracy"] if e2e_head_res else np.nan,
            "sm3_e2e_head_auc": e2e_head_res["macro_auc"] if e2e_head_res else np.nan,
            "c5_linear_acc": res_c5_lin["accuracy"],
            "c5_linear_auc": res_c5_lin["macro_auc"],
            "ood_hp_v2_auc": bin_results.get("sm_vs_higgs_portal_v2", {}).get("macro_auc", np.nan),
            "ood_hv_auc": bin_results.get("sm_vs_hidden_valley", {}).get("macro_auc", np.nan),
            "ood_hp_v1_auc": bin_results.get("sm_vs_higgs_portal_v1", {}).get("macro_auc", np.nan),
        }
        summary_rows.append(row)
        print(
            f"  SM-3 Linear: acc={row['sm3_linear_acc']:.4f}, auc={row['sm3_linear_auc']:.4f} | "
            f"5-Class Linear: acc={row['c5_linear_acc']:.4f}, auc={row['c5_linear_auc']:.4f} | "
            f"OOD AUC (HP_v2={row['ood_hp_v2_auc']:.4f}, HV={row['ood_hv_auc']:.4f}, HP_v1={row['ood_hp_v1_auc']:.4f})"
        )

        # Save detailed per-model JSON
        detail = {
            "model": mname,
            "latent_dim": D,
            "sm3_linear": res_sm3_lin,
            "sm3_mlp": res_sm3_mlp,
            "sm3_e2e_head": e2e_head_res,
            "c5_linear": res_c5_lin,
            "binary_ood_linear": bin_results,
        }
        with open(PROBE_OUT_DIR / f"{mname}_probe_detail.json", "w") as f:
            json.dump(detail, f, indent=2)

        # 4. Label-budget curve for D=64 models (strictly budgeted 80% train / 20% val of total N_labeled)
        if D == 64:
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
                        r_c = train_torch_probe(
                            z_sm[idx_tr_b], sm_labels[idx_tr_b],
                            z_sm[idx_va_b], sm_labels[idx_va_b],
                            z_sm[sm_te], sm_labels[sm_te],
                            n_classes=3, device=device, probe_type=ptype,
                            epochs=epochs_b, batch_size=eff_bs_b, seed=seed
                        )
                        curve_rows.append({
                            "model": mname,
                            "probe_type": ptype,
                            "n_labeled": n_labeled,
                            "seed": seed,
                            "test_acc": r_c["accuracy"],
                            "test_auc": r_c["macro_auc"],
                        })

    df_sum = pd.DataFrame(summary_rows)
    sum_csv = PROBE_OUT_DIR / "probe_summary.csv"
    if sum_csv.exists():
        old_sum = pd.read_csv(sum_csv)
        old_sum = old_sum[~old_sum["model"].isin(df_sum["model"])]
        df_sum = pd.concat([old_sum, df_sum], ignore_index=True).sort_values("model")
    df_sum.to_csv(sum_csv, index=False)
    print(f"\nSaved summary table to {sum_csv} ({len(df_sum)} models)")

    if curve_rows:
        df_curves = pd.DataFrame(curve_rows)
        curve_csv = PROBE_OUT_DIR / "label_efficiency_curves.csv"
        if curve_csv.exists():
            old_c = pd.read_csv(curve_csv)
            old_c = old_c[~old_c["model"].isin(df_curves["model"])]
            df_curves = pd.concat([old_c, df_curves], ignore_index=True)
        df_curves.to_csv(curve_csv, index=False)
        print(f"Saved label-efficiency curves to {curve_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, nargs="*", default=None)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    dev = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    evaluate_all_models(dev, args.models)
