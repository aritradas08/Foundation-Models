import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pretrain import (
RAW_COLS, N_TRACK_FEATS, ALL_PROCESS_NAMES, SM_LABELS,
TARGET_DIM,
engineer_tracks, compute_event_summary,
MLPEncoder, AEDecoder,
)

ENCODER_CKPT = "encoder_ae_mlp.pt"
DECODER_CKPT = "decoder_ae_mlp.pt"

DATA_PATH = "/global/cfs/cdirs/m4958/usr/aritra08/Anomaly_Detection/data/combined_pu0_1500k.parquet"

CKPT_DIR = "/global/cfs/cdirs/m4958/usr/aritra08/autoencoders/mlp/results/sm_only_v3_run2/checkpoints"
OUT_DIR = "/global/cfs/cdirs/m4958/usr/aritra08/autoencoders/mlp/results/sm_only_v3_run2/embeddings"
OUT_PATH = os.path.join(OUT_DIR, "z_embeddings_all_events.npz")

INCLUDE_LABELS = None   
BATCH_SIZE = 512
N_PRINT_PER_LABEL = 3
N_PRINT_RANDOM = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_latent_dim_from_config(ckpt_dir: str) -> int:
    config_path = os.path.join(os.path.dirname(ckpt_dir), "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        latent_dim = config["latent_dim"]
        print(f"LATENT_DIM = {latent_dim}")
        return latent_dim

    ckpt_path = os.path.join(ckpt_dir, ENCODER_CKPT)
    state = torch.load(ckpt_path, map_location="cpu")
    latent_dim = state["net.4.weight"].shape[0]
    return latent_dim

class InferenceMLPDataset(torch.utils.data.Dataset):
    def __init__(self, df, track_scaler, target_scaler, include_labels=None):
        if include_labels is not None:
            df = df[df["label"].isin(include_labels)].reset_index(drop=True)

        self.labels = df["label"].values.astype(np.int64)
        all_summaries = []
        self.n_tracks = []

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
            all_summaries.append(summary_scaled)
            self.n_tracks.append(tokens_scaled.shape[0])

        self.summaries = np.stack(all_summaries, axis=0)   # (N_events, 28)
        self.n_tracks = np.array(self.n_tracks, dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.summaries[idx], self.labels[idx], self.n_tracks[idx]

def produce_embeddings():
    os.makedirs(OUT_DIR, exist_ok=True)

    LATENT_DIM = load_latent_dim_from_config(CKPT_DIR)

    with open(os.path.join(CKPT_DIR, "scaler.pkl"), "rb") as f:
        track_scaler = pickle.load(f)

    with open(os.path.join(CKPT_DIR, "target_scaler.pkl"), "rb") as f:
        target_scaler = pickle.load(f)

    encoder = MLPEncoder(latent_dim=LATENT_DIM).to(DEVICE)
    encoder.load_state_dict(torch.load(
        os.path.join(CKPT_DIR, ENCODER_CKPT), map_location=DEVICE
    ))
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    decoder = AEDecoder(in_dim=LATENT_DIM).to(DEVICE)
    decoder.load_state_dict(torch.load(
        os.path.join(CKPT_DIR, DECODER_CKPT), map_location=DEVICE
    ))
    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad = False

    print(f"Loading data from {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)

    dataset = InferenceMLPDataset(
        df, track_scaler, target_scaler,
        include_labels=INCLUDE_LABELS,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size = BATCH_SIZE,
        shuffle = False,   # preserve row order
        num_workers = 4,
    )
    print(f" Events to embed: {len(dataset)}")

    all_z, all_labels, all_n_tracks = [], [], []
    all_recon_errors, all_best_cond = [], []

    with torch.no_grad():
        for summaries, labels, n_tracks in loader:
            summaries = summaries.to(DEVICE)   # (B, 28)

            z = encoder(summaries)   # (B, LATENT_DIM) 

            recon = decoder(z)
            recon_err = F.mse_loss(recon, summaries,
                               reduction="none").mean(dim=1)
            best_cond = torch.full((z.shape[0],), -1, dtype=torch.long)

            
            all_z.append(z.cpu().numpy())
            all_labels.append(labels.numpy())
            all_n_tracks.append(n_tracks.numpy())
            all_recon_errors.append(recon_err.cpu().numpy())
            all_best_cond.append(best_cond.numpy())

    z_all = np.concatenate(all_z, axis=0).astype(np.float32)
    labels_all = np.concatenate(all_labels, axis=0)
    n_tracks_all = np.concatenate(all_n_tracks, axis=0)
    recon_err_all = np.concatenate(all_recon_errors, axis=0).astype(np.float32)
    best_cond_all = np.concatenate(all_best_cond, axis=0)

    print(f"z shape: {z_all.shape}")

    np.savez(
        OUT_PATH,
        z = z_all,
        labels = labels_all,
        n_tracks = n_tracks_all,
        recon_errors = recon_err_all,
        best_condition = best_cond_all,
    )
    print(f"\nSaved embeddings to {OUT_PATH}")

    return z_all, labels_all, n_tracks_all, recon_err_all, best_cond_all


if __name__ == "__main__":
    produce_embeddings()
