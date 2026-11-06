import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from pretrain_mlp import raw_cols, engineer_tracks, compute_event_summary, MLPEncoder, AEDecoder


encoder_ckpt = "encoder_ae_mlp.pt"
decoder_ckpt = "decoder_ae_mlp.pt"

data_path = "/global/cfs/cdirs/m4958/usr/aritra08/Anomaly_Detection/data/combined_pu0_1500k.parquet"
ckpt_dir = "/global/cfs/cdirs/m4958/usr/aritra08/autoencoders/mlp/results/sm_only_v3_run2/checkpoints"
out_dir = "/global/cfs/cdirs/m4958/usr/aritra08/autoencoders/mlp/results/sm_only_v3_run2/embeddings"
out_path = os.path.join(out_dir, "z_embeddings_all_events.npz")

include_labels = None

batch_size = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_latent_dim(ckpt_dir):
    config_path = os.path.join(os.path.dirname(ckpt_dir), "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            return json.load(f)["latent_dim"]

    state = torch.load(os.path.join(ckpt_dir, encoder_ckpt), map_location="cpu")
    return state["net.4.weight"].shape[0]


class InferenceMLPDataset(Dataset):
    def __init__(self, df, track_scaler, target_scaler, include_labels=None):
        if include_labels is not None:
            df = df[df["label"].isin(include_labels)].reset_index(drop=True)

        self.labels = df["label"].values.astype(np.int64)
        summaries = []
        n_tracks = []

        for i in range(len(df)):
            arr_raw = np.column_stack([df[c].iloc[i] for c in raw_cols]).astype(np.float32)
            tokens = engineer_tracks(arr_raw)
            tokens_scaled = track_scaler.transform(tokens).astype(np.float32)
            summary = compute_event_summary(tokens_scaled)
            summary_scaled = target_scaler.transform(summary.reshape(1, -1)).squeeze(0).astype(np.float32)
            summaries.append(summary_scaled)
            n_tracks.append(tokens_scaled.shape[0])

        self.summaries = np.stack(summaries, axis=0)
        self.n_tracks = np.array(n_tracks, dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.summaries[idx], self.labels[idx], self.n_tracks[idx]


def produce_embeddings():
    os.makedirs(out_dir, exist_ok=True)

    latent_dim = load_latent_dim(ckpt_dir)

    with open(os.path.join(ckpt_dir, "scaler.pkl"), "rb") as f:
        track_scaler = pickle.load(f)
    with open(os.path.join(ckpt_dir, "target_scaler.pkl"), "rb") as f:
        target_scaler = pickle.load(f)

    encoder = MLPEncoder(latent_dim=latent_dim).to(device)
    encoder.load_state_dict(torch.load(os.path.join(ckpt_dir, encoder_ckpt), map_location=device))
    decoder = AEDecoder(in_dim=latent_dim).to(device)
    decoder.load_state_dict(torch.load(os.path.join(ckpt_dir, decoder_ckpt), map_location=device))

    for model in (encoder, decoder):
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

    df = pd.read_parquet(data_path)
    dataset = InferenceMLPDataset(df, track_scaler, target_scaler, include_labels=include_labels)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False, 
        num_workers=4,
    )

    all_z, all_labels, all_n_tracks, all_recon_errors = [], [], [], []

    with torch.no_grad():
        for summaries, labels, n_tracks in loader:
            summaries = summaries.to(device)

            z = encoder(summaries)
            recon = decoder(z)
            recon_err = F.mse_loss(recon, summaries, reduction="none").mean(dim=1)

            all_z.append(z.cpu().numpy())
            all_labels.append(labels.numpy())
            all_n_tracks.append(n_tracks.numpy())
            all_recon_errors.append(recon_err.cpu().numpy())

    z_all = np.concatenate(all_z, axis=0).astype(np.float32)
    labels_all = np.concatenate(all_labels, axis=0)
    n_tracks_all = np.concatenate(all_n_tracks, axis=0)
    recon_err_all = np.concatenate(all_recon_errors, axis=0).astype(np.float32)

    print(f"z shape: {z_all.shape}, per-dim std: {z_all.std(axis=0)}")

    np.savez(
        out_path,
        z=z_all,
        labels=labels_all,
        n_tracks=n_tracks_all,
        recon_errors=recon_err_all,
    )
    print(f"saved to {out_path}")

    return z_all, labels_all, n_tracks_all, recon_err_all


if __name__ == "__main__":
    produce_embeddings()
