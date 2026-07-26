import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler

DATA_PATH = "/global/cfs/cdirs/m4958/usr/aritra08/Anomaly_Detection/data/combined_pu0_1500k.parquet"
SAVE_DIR = "/global/cfs/cdirs/m4958/usr/aritra08/autoencoders/mlp/results/sm_only_v3_run2"
CKPT_DIR = os.path.join(SAVE_DIR, "checkpoints")

RAW_COLS = ["d0", "z0", "theta", "p", "eta", "phi", "pt"]
RAW_IDX = {name: i for i, name in enumerate(RAW_COLS)}
TRACK_FEAT_NAMES = ["d0", "z0", "p", "pt", "tx", "ty", "tz"]
N_TRACK_FEATS = 7

SM_LABELS = [0, 1, 2]
BSM_LABELS = [3, 4]
ALL_PROCESS_NAMES = ["ttbar", "ggf", "dihiggs", "higgs_portal", "hidden_valley"]

TARGET_DIM = N_TRACK_FEATS * 4   # 28

LATENT_DIM = 64     
ENCODER_HIDDEN_1 = 256    
ENCODER_HIDDEN_2 = 128
DECODER_HIDDEN_1 = 128    
DECODER_HIDDEN_2 = 256

BATCH_SIZE = 512
N_EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-6

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def symlog(x: np.ndarray) -> np.ndarray:
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


def compute_event_summary(tokens_scaled: np.ndarray) -> np.ndarray:
    return np.concatenate([
        tokens_scaled.mean(axis=0),
        tokens_scaled.std(axis=0),
        tokens_scaled.max(axis=0),
        tokens_scaled.min(axis=0),
    ]).astype(np.float32)


def fit_scaler_on_sm(df: pd.DataFrame) -> StandardScaler:
    df_sm = df[df["label"].isin(SM_LABELS)].reset_index(drop=True)
    all_tracks = []
    for i in range(len(df_sm)):
        arr_raw = np.column_stack(
            [df_sm[c].iloc[i] for c in RAW_COLS]
        ).astype(np.float32)
        all_tracks.append(engineer_tracks(arr_raw))
    all_tracks = np.concatenate(all_tracks, axis=0)
    scaler = StandardScaler()
    scaler.fit(all_tracks)
    print(f"Track scaler fit on SM-only tracks: {all_tracks.shape}")
    return scaler


def fit_target_scaler_on_sm(df: pd.DataFrame, track_scaler: StandardScaler) -> StandardScaler:
    df_sm = df[df["label"].isin(SM_LABELS)].reset_index(drop=True)
    all_summaries = []
    for i in range(len(df_sm)):
        arr_raw = np.column_stack(
            [df_sm[c].iloc[i] for c in RAW_COLS]
        ).astype(np.float32)
        tokens = engineer_tracks(arr_raw)
        tokens_scaled = track_scaler.transform(tokens).astype(np.float32)
        all_summaries.append(compute_event_summary(tokens_scaled))
    all_summaries = np.stack(all_summaries, axis=0)
    target_scaler = StandardScaler()
    target_scaler.fit(all_summaries)
    print(f"Target scaler fit on SM-only summaries: {all_summaries.shape}")
    return target_scaler

class AEMLPEventDataset(Dataset):
    def __init__(self, df: pd.DataFrame, track_scaler: StandardScaler, target_scaler: StandardScaler, include_labels=None):
        if include_labels is not None:
            df = df[df["label"].isin(include_labels)].reset_index(drop=True)

        self.labels = df["label"].values.astype(np.int64)
        self.summaries = []

        for i in range(len(df)):
            arr_raw = np.column_stack(
                [df[c].iloc[i] for c in RAW_COLS]
            ).astype(np.float32)
            tokens = engineer_tracks(arr_raw)
            tokens_scaled = track_scaler.transform(tokens).astype(np.float32)
            summary_raw = compute_event_summary(tokens_scaled)
            summary_scaled = target_scaler.transform(
                summary_raw.reshape(1, -1)
            ).squeeze(0).astype(np.float32)
            self.summaries.append(summary_scaled)

        self.summaries = np.stack(self.summaries, axis=0)   # (N_events, 28)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.summaries[idx], self.labels[idx]

class MLPEncoder(nn.Module):
    def __init__(self, in_dim=TARGET_DIM,
                 hidden1=ENCODER_HIDDEN_1,
                 hidden2=ENCODER_HIDDEN_2,
                 latent_dim=LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1), nn.ReLU(inplace=True),
            nn.Linear(hidden1, hidden2), nn.ReLU(inplace=True),
            nn.Linear(hidden2, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AEDecoder(nn.Module):
    def __init__(self, in_dim=LATENT_DIM, hidden1=DECODER_HIDDEN_1, hidden2=DECODER_HIDDEN_2, out_dim=TARGET_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1), nn.ReLU(inplace=True),
            nn.Linear(hidden1, hidden2), nn.ReLU(inplace=True),
            nn.Linear(hidden2, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

def train_ae():
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    assert set(SM_LABELS).isdisjoint(set(BSM_LABELS))

    print(f"\nLoading data from {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)

    track_scaler = fit_scaler_on_sm(df)
    target_scaler = fit_target_scaler_on_sm(df, track_scaler)

    dataset = AEMLPEventDataset(df, track_scaler, target_scaler,
                                include_labels=SM_LABELS)

    class_counts = np.bincount(dataset.labels)
    weights = torch.from_numpy(
        (1.0 / class_counts[dataset.labels]).astype(np.float32)
    )
    sampler = WeightedRandomSampler(weights, num_samples=len(weights),
                                    replacement=True)

    loader = DataLoader(
        dataset,
        batch_size = BATCH_SIZE,
        sampler = sampler,
        drop_last = True,
        num_workers = 4,
    )
    print(f" Training events (SM only): {len(dataset)}")
    for label_val in SM_LABELS:
        n = (dataset.labels == label_val).sum()
        print(f"label={label_val} ({ALL_PROCESS_NAMES[label_val]}): {n} events")

    encoder = MLPEncoder().to(DEVICE)
    decoder = AEDecoder().to(DEVICE)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS
    )

    history = {"epoch": [], "loss": []}

    for epoch in range(1, N_EPOCHS + 1):
        encoder.train()
        decoder.train()
        epoch_loss = 0.0
        n_batches  = 0

        for summaries, labels in loader:
            summaries = summaries.to(DEVICE)   # (B, 28) 
            z = encoder(summaries)          # (B, LATENT_DIM)
            recon = decoder(z)                  # (B, 28)
            loss = F.mse_loss(recon, summaries)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(decoder.parameters()),
                max_norm=1.0,
            )
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / n_batches
        history["epoch"].append(epoch)
        history["loss"].append(avg_loss)
        print(f"Epoch {epoch:3d}/{N_EPOCHS}; loss = {avg_loss:.6f}; lr = {scheduler.get_last_lr()[0]:.2e} ")

    torch.save(encoder.state_dict(),
               os.path.join(CKPT_DIR, "encoder_ae_mlp.pt"))
    torch.save(decoder.state_dict(),
               os.path.join(CKPT_DIR, "decoder_ae_mlp.pt"))

    with open(os.path.join(CKPT_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(track_scaler, f)
    with open(os.path.join(CKPT_DIR, "target_scaler.pkl"), "wb") as f:
        pickle.dump(target_scaler, f)

    pd.DataFrame(history).to_csv(
        os.path.join(SAVE_DIR, "training_history.csv"), index=False
    )

    config = {
        "mode": "ae", "encoder": "mlp", "latent_dim": LATENT_DIM,
        "target_dim": TARGET_DIM,
        "target_stats": ["mean", "std", "max", "min"],
    }
    
    with open(os.path.join(SAVE_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nPretraining done")
    return encoder, decoder, track_scaler, target_scaler, history


if __name__ == "__main__":
    train_ae()
