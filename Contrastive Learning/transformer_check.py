import os
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH  = "/global/cfs/cdirs/m4958/usr/aritra08/Anomaly_Detection/data/combined_pu0_1500k.parquet"
SAVE_DIR   = "/global/cfs/cdirs/m4958/usr/aritra08/contrastive_learning/checks/300k_events_5class_v4"
PLOT_DIR   = "/global/cfs/cdirs/m4958/usr/aritra08/contrastive_learning/checks/300k_events_5class_v4/plots"

RAW_COLS = ["d0", "z0", "theta", "p", "eta", "phi", "pt"]
RAW_IDX = {name: i for i, name in enumerate(RAW_COLS)}

TRACK_FEAT_NAMES = ["d0", "z0", "p", "pt", "tx", "ty", "tz"]
N_FEATURES = len(TRACK_FEAT_NAMES)
N_CLASSES  = 5

def symlog(x):
    return np.sign(x) * np.log1p(np.abs(x))


def engineer_tracks(arr_raw: np.ndarray) -> np.ndarray:
    d0 = symlog(arr_raw[:, RAW_IDX["d0"]])
    z0 = symlog(arr_raw[:, RAW_IDX["z0"]])
    p = arr_raw[:, RAW_IDX["p"]]
    pt = arr_raw[:, RAW_IDX["pt"]]
    phi = arr_raw[:, RAW_IDX["phi"]]
    theta = arr_raw[:, RAW_IDX["theta"]]
    tx = np.cos(phi) * np.sin(theta)
    ty = np.sin(phi) * np.sin(theta)
    tz = np.cos(theta)
    return np.column_stack([d0, z0, p, pt, tx, ty, tz]).astype(np.float32)

PROCESS_NAMES = ["ttbar", "ggf", "dihiggs", "higgs_portal", "hidden_valley"]
COLORS        = ["Blue", "Green", "Red", "Black", "Purple"]

MODEL_CONFIG = {
    "n_features": N_FEATURES,
    "d_model": 64,
    "nhead": 8,
    "num_layers": 8,
    "dim_feedforward": 256,
    "dropout": 0.1,
    "n_classes": N_CLASSES,
}

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

class TrackDataset(Dataset):
    def __init__(self, df, scaler=None, fit_scaler=False):
        self.labels = df["label"].values.astype(np.int64)

        if fit_scaler:
            all_tracks = np.concatenate(
                [engineer_tracks(np.column_stack([df[c].iloc[i] for c in RAW_COLS]).astype(np.float32))
                 for i in range(len(df))],
                axis=0
            ).astype(np.float32)
            scaler = StandardScaler()
            scaler.fit(all_tracks)
            print(f"Scaler fit done on {len(all_tracks):,} tracks (Training data only)")

        self.scaler = scaler

        self.tracks = []
        for i in range(len(df)):
            arr_raw = np.column_stack(
                [df[c].iloc[i] for c in RAW_COLS]
            ).astype(np.float32)
            arr = engineer_tracks(arr_raw)
            if self.scaler is not None:
                arr = self.scaler.transform(arr)
            self.tracks.append(arr)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.tracks[idx], self.labels[idx]


def collate_fn(batch):
    tracks, labels = zip(*batch)
    max_len = max(t.shape[0] for t in tracks)

    padded = np.zeros((len(tracks), max_len, N_FEATURES), dtype=np.float32)
    mask   = np.ones((len(tracks), max_len), dtype=bool)

    for i, t in enumerate(tracks):
        n = t.shape[0]
        padded[i, :n, :] = t
        mask[i, :n] = False    # False = real track, True = padding

    return (
        torch.tensor(padded),
        torch.tensor(mask),
        torch.tensor(labels, dtype=torch.long),
    )


class TrackTransformer(nn.Module):
    def __init__(
        self,
        n_features = N_FEATURES,
        d_model = 128,
        nhead = 8,
        num_layers = 4,
        dim_feedforward = 256,
        dropout = 0.1,
        n_classes = N_CLASSES,
    ):
        super().__init__()

        # input embedding
        self.input_proj = nn.Linear(n_features, d_model)

        # encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = d_model,
            nhead = nhead,
            dim_feedforward = dim_feedforward,
            dropout = dropout,
            batch_first = True,
            norm_first = True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x, padding_mask):
        x = self.input_proj(x)                              # (batch, seq_len, d_model)
        x = self.encoder(x, src_key_padding_mask=padding_mask)

        # masked mean pooling
        real_mask = (~padding_mask).unsqueeze(-1).float()   # (batch, seq_len, 1)
        x = (x * real_mask).sum(dim=1) / real_mask.sum(dim=1)   # (batch, d_model)

        return self.head(x)                                 # (batch, n_classes)


def accuracy(logits, labels):
    return (logits.argmax(dim=-1) == labels).float().mean().item()

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for x, mask, y in tqdm(loader, desc="  train", leave=False):
        x, mask, y = x.to(device), mask.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x, mask)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(y)
        total_acc  += accuracy(logits, y) * len(y)
        n += len(y)
    return total_loss/n, total_acc/n


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for x, mask, y in tqdm(loader, desc="  eval ", leave=False):
        x, mask, y = x.to(device), mask.to(device), y.to(device)
        logits = model(x, mask)
        loss = criterion(logits, y)
        total_loss += loss.item() * len(y)
        total_acc  += accuracy(logits, y) * len(y)
        n += len(y)
    return total_loss/n, total_acc/n



@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for x, mask, y in tqdm(loader, desc="  infer", leave=False):
        x, mask = x.to(device), mask.to(device)
        all_logits.append(model(x, mask).cpu().numpy())
        all_labels.append(y.numpy())
    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    preds = logits.argmax(axis=-1)
    return probs, preds, labels


def plot_confusion_matrix(labels, preds):
    cm = confusion_matrix(labels, preds, normalize="true")
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=PROCESS_NAMES, yticklabels=PROCESS_NAMES, ax=ax
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix", fontweight="bold")
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "confusion_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f" The plot has been saved at {path}")


def plot_roc_curves(labels, probs):
    labels_onehot = np.eye(N_CLASSES)[labels]
    fig, ax = plt.subplots(figsize=(9, 7))

    for i, (name, color) in enumerate(zip(PROCESS_NAMES, COLORS)):
        fpr, tpr, _ = roc_curve(labels_onehot[:, i], probs[:, i])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{name}  (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves", fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.6)
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "roc_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"The plot has been saved at {path}")


def plot_training_curves(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Training History", fontweight="bold")
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Train", color="Blue")
    ax1.plot(epochs, history["val_loss"],   label="Val",   color="Red")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Loss"); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], label="Train", color="Blue")
    ax2.plot(epochs, history["val_acc"],   label="Val",   color="Red")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy"); ax2.legend(); ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "training_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_evaluation(model, test_loader, device):
    print("\nEvaluating on test set...")
    probs, preds, labels = get_predictions(model, test_loader, device)

    acc = (preds == labels).mean()
    print(f"\nTest accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=PROCESS_NAMES, digits=4))

    plot_confusion_matrix(labels, preds)
    plot_roc_curves(labels, probs)
    print("\n ALl the plots have been saved.")

def load_splits(batch_size):
    df = pd.read_parquet(DATA_PATH)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Total events: {len(df):,}")

    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42,
                                         stratify=df["label"])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42,
                                         stratify=temp_df["label"])

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")

    train_ds = TrackDataset(train_df, fit_scaler=True)
    val_ds = TrackDataset(val_df,   scaler=train_ds.scaler)
    test_ds = TrackDataset(test_df,  scaler=train_ds.scaler)

    # save scaler
    scaler_path = os.path.join(SAVE_DIR, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(train_ds.scaler, f)
    print(f" Scaler has beeen saved to {scaler_path}")

    loader_kwargs = dict(collate_fn=collate_fn, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader, train_ds.scaler


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, test_loader, _ = load_splits(args.batch_size)

    # building the model
    model = TrackTransformer(**MODEL_CONFIG).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_path    = os.path.join(SAVE_DIR, "best_model.pt")
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc":   []}

    print(f"\nTraining for {args.epochs} epochs")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer,
                                            criterion, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        marker = " *" if val_acc > best_val_acc else ""
        print(f"Epoch {epoch:3d}/{args.epochs}; train loss={train_loss:.4f} acc={train_acc:.4f}; val loss={val_loss:.4f} acc={val_acc:.4f}{marker}")
      
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "model_config": MODEL_CONFIG,
                "optimizer": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
            }, best_path)

    print(f"\nBest val accuracy: {best_val_acc:.4f}")
   
    plot_training_curves(history)

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    run_evaluation(model, test_loader, device)


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    scaler_path = os.path.join(SAVE_DIR, "scaler.pkl")

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    model = TrackTransformer(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    df = pd.read_parquet(DATA_PATH)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    _, temp_df = train_test_split(df, test_size=0.30, random_state=42,
                                  stratify=df["label"])
    _, test_df = train_test_split(temp_df, test_size=0.50, random_state=42,
                                  stratify=temp_df["label"])
    test_df = test_df.reset_index(drop=True)
    print(f" Test events: {len(test_df):,}")

    test_ds = TrackDataset(test_df, scaler=scaler)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate_fn,
                             num_workers=4, pin_memory=True)

    run_evaluation(model, test_loader, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TrackTransformer classifier")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--checkpoint", type=str, default=os.path.join(SAVE_DIR, "best_model.pt"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    else:
        evaluate(args)
