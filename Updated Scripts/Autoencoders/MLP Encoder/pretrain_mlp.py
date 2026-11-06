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

data_path = "/global/cfs/cdirs/m4958/usr/aritra08/Anomaly_Detection/data/combined_pu0_1500k.parquet"
save_dir = "/global/cfs/cdirs/m4958/usr/aritra08/autoencoders/mlp/results/sm_only_v3_run2"
ckpt_dir = os.path.join(save_dir, "checkpoints")

raw_cols = ["d0", "z0", "theta", "p", "eta", "phi", "pt"]
raw_idx = {name: i for i, name in enumerate(raw_cols)}
track_feat_names = ["d0", "z0", "p", "pt", "tx", "ty", "tz"]
n_track_feats = 7

sm_labels = [0, 1, 2]
bsm_labels = [3, 4]
all_process_names = ["ttbar", "ggf", "dihiggs", "higgs_portal", "hidden_valley"]

target_dim = n_track_feats * 4

latent_dim = 64
encoder_hidden_1 = 256
encoder_hidden_2 = 128
decoder_hidden_1 = 128
decoder_hidden_2 = 256

batch_size = 512
n_epochs = 50
lr = 1e-3
weight_decay = 1e-6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def symlog(x):
    return np.sign(x) * np.log1p(np.abs(x))


def engineer_tracks(arr_raw):
    d0 = symlog(arr_raw[:, raw_idx["d0"]])
    z0 = symlog(arr_raw[:, raw_idx["z0"]])
    p = arr_raw[:, raw_idx["p"]]
    pt = arr_raw[:, raw_idx["pt"]]
    phi = arr_raw[:, raw_idx["phi"]]
    theta = arr_raw[:, raw_idx["theta"]]

    tx = np.cos(phi) * np.sin(theta)
    ty = np.sin(phi) * np.sin(theta)
    tz = np.cos(theta)
    return np.column_stack([d0, z0, p, pt, tx, ty, tz]).astype(np.float32)


def event_to_array(df, i):
    return np.column_stack([df[c].iloc[i] for c in raw_cols]).astype(np.float32)


def compute_event_summary(tokens_scaled):
    return np.concatenate([
        tokens_scaled.mean(axis=0),
        tokens_scaled.std(axis=0),
        tokens_scaled.max(axis=0),
        tokens_scaled.min(axis=0),
    ]).astype(np.float32)


def fit_scaler_on_sm(df):
    df_sm = df[df["label"].isin(sm_labels)].reset_index(drop=True)
    all_tracks = np.concatenate(
        [engineer_tracks(event_to_array(df_sm, i)) for i in range(len(df_sm))], axis=0
    )
    scaler = StandardScaler()
    scaler.fit(all_tracks)
    return scaler


def fit_target_scaler_on_sm(df, track_scaler):
    df_sm = df[df["label"].isin(sm_labels)].reset_index(drop=True)
    summaries = []
    for i in range(len(df_sm)):
        tokens = engineer_tracks(event_to_array(df_sm, i))
        tokens_scaled = track_scaler.transform(tokens).astype(np.float32)
        summaries.append(compute_event_summary(tokens_scaled))

    target_scaler = StandardScaler()
    target_scaler.fit(np.stack(summaries, axis=0))
    return target_scaler


class AEMLPEventDataset(Dataset):
    def __init__(self, df, track_scaler, target_scaler, include_labels=None):
        if include_labels is not None:
            df = df[df["label"].isin(include_labels)].reset_index(drop=True)

        self.labels = df["label"].values.astype(np.int64)
        summaries = []

        for i in range(len(df)):
            tokens = engineer_tracks(event_to_array(df, i))
            tokens_scaled = track_scaler.transform(tokens).astype(np.float32)
            summary = compute_event_summary(tokens_scaled)
            summary_scaled = target_scaler.transform(summary.reshape(1, -1)).squeeze(0).astype(np.float32)
            summaries.append(summary_scaled)

        self.summaries = np.stack(summaries, axis=0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.summaries[idx], self.labels[idx]


class MLPEncoder(nn.Module):
    def __init__(self, in_dim=target_dim, hidden1=encoder_hidden_1,
                 hidden2=encoder_hidden_2, latent_dim=latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1), nn.ReLU(inplace=True),
            nn.Linear(hidden1, hidden2), nn.ReLU(inplace=True),
            nn.Linear(hidden2, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class AEDecoder(nn.Module):
    def __init__(self, in_dim=latent_dim, hidden1=decoder_hidden_1,
                 hidden2=decoder_hidden_2, out_dim=target_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1), nn.ReLU(inplace=True),
            nn.Linear(hidden1, hidden2), nn.ReLU(inplace=True),
            nn.Linear(hidden2, out_dim),
        )

    def forward(self, z):
        return self.net(z)


def train_ae():
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    assert set(sm_labels).isdisjoint(set(bsm_labels))

    df = pd.read_parquet(data_path)

    track_scaler = fit_scaler_on_sm(df)
    target_scaler = fit_target_scaler_on_sm(df, track_scaler)

    dataset = AEMLPEventDataset(df, track_scaler, target_scaler, include_labels=sm_labels)

    class_counts = np.bincount(dataset.labels)
    weights = torch.from_numpy((1.0 / class_counts[dataset.labels]).astype(np.float32))
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=True,
        num_workers=4,
    )

    encoder = MLPEncoder().to(device)
    decoder = AEDecoder().to(device)

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    history = {"epoch": [], "loss": []}

    for epoch in range(1, n_epochs + 1):
        encoder.train()
        decoder.train()
        epoch_loss = 0.0
        n_batches = 0

        for summaries, labels in loader:
            summaries = summaries.to(device)

            z = encoder(summaries)
            recon = decoder(z)
            loss = F.mse_loss(recon, summaries)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / n_batches
        history["epoch"].append(epoch)
        history["loss"].append(avg_loss)
        print(f"epoch {epoch}/{n_epochs}: loss {avg_loss:.6f}", flush=True)

    torch.save(encoder.state_dict(), os.path.join(ckpt_dir, "encoder_ae_mlp.pt"))
    torch.save(decoder.state_dict(), os.path.join(ckpt_dir, "decoder_ae_mlp.pt"))

    with open(os.path.join(ckpt_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(track_scaler, f)
    with open(os.path.join(ckpt_dir, "target_scaler.pkl"), "wb") as f:
        pickle.dump(target_scaler, f)

    pd.DataFrame(history).to_csv(os.path.join(save_dir, "training_history.csv"), index=False)

    config = {
        "mode": "ae",
        "encoder": "mlp",
        "latent_dim": latent_dim,
        "target_dim": target_dim,
        "target_stats": ["mean", "std", "max", "min"],
    }
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    return encoder, decoder, track_scaler, target_scaler, history


if __name__ == "__main__":
    train_ae()
