import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from pretrain_transformer import (
    raw_cols, n_track_feats, engineer_tracks, compute_event_summary,
    SetTransformerEncoder, AEDecoder,
)


encoder_ckpt = "encoder_ae_transformer.pt"
decoder_ckpt = "decoder_ae_transformer.pt"

data_path = "/global/cfs/cdirs/m4958/usr/aritra08/Anomaly_Detection/data/combined_pu0_1500k.parquet"
ckpt_dir = "/global/cfs/cdirs/m4958/usr/aritra08/autoencoders/transformer/results/sm_only_v4/checkpoints"
out_dir = "/global/cfs/cdirs/m4958/usr/aritra08/autoencoders/transformer/results/sm_only_v4/embeddings"
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
    return state["out_proj.weight"].shape[0]


class InferenceEventDataset(Dataset):
    def __init__(self, df, track_scaler, target_scaler, include_labels=None):
        if include_labels is not None:
            df = df[df["label"].isin(include_labels)].reset_index(drop=True)

        self.labels = df["label"].values.astype(np.int64)
        self.event_tracks = []
        self.summaries = []

        for i in range(len(df)):
            arr_raw = np.column_stack([df[c].iloc[i] for c in raw_cols]).astype(np.float32)
            tokens = engineer_tracks(arr_raw)
            tokens_scaled = track_scaler.transform(tokens).astype(np.float32)
            summary = compute_event_summary(tokens_scaled)
            summary_scaled = target_scaler.transform(summary.reshape(1, -1)).squeeze(0).astype(np.float32)
            self.event_tracks.append(tokens_scaled)
            self.summaries.append(summary_scaled)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        tracks = self.event_tracks[idx]
        return tracks, self.summaries[idx], self.labels[idx], tracks.shape[0]


def inference_collate_fn(batch):
    tracks_list, summaries_list, labels, n_tracks = zip(*batch)
    bs = len(tracks_list)
    max_len = max(t.shape[0] for t in tracks_list)

    padded = np.zeros((bs, max_len, n_track_feats), dtype=np.float32)
    mask = np.zeros((bs, max_len), dtype=bool)
    for i, t in enumerate(tracks_list):
        padded[i, :len(t)] = t
        mask[i, :len(t)] = True

    return (
        torch.from_numpy(padded),
        torch.from_numpy(mask),
        torch.from_numpy(np.stack(summaries_list, axis=0)),
        torch.from_numpy(np.array(labels, dtype=np.int64)),
        torch.from_numpy(np.array(n_tracks, dtype=np.int64)),
    )


def produce_embeddings():
    os.makedirs(out_dir, exist_ok=True)

    latent_dim = load_latent_dim(ckpt_dir)

    with open(os.path.join(ckpt_dir, "scaler.pkl"), "rb") as f:
        track_scaler = pickle.load(f)
    with open(os.path.join(ckpt_dir, "target_scaler.pkl"), "rb") as f:
        target_scaler = pickle.load(f)

    encoder = SetTransformerEncoder(latent_dim=latent_dim).to(device)
    encoder.load_state_dict(torch.load(os.path.join(ckpt_dir, encoder_ckpt), map_location=device))
    decoder = AEDecoder(in_dim=latent_dim).to(device)
    decoder.load_state_dict(torch.load(os.path.join(ckpt_dir, decoder_ckpt), map_location=device))

    for model in (encoder, decoder):
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

    df = pd.read_parquet(data_path)
    dataset = InferenceEventDataset(df, track_scaler, target_scaler, include_labels=include_labels)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  
        collate_fn=inference_collate_fn,
        num_workers=4,
    )

    all_z, all_labels, all_n_tracks, all_recon_errors = [], [], [], []

    with torch.no_grad():
        for tracks, mask, targets, labels, n_tracks in loader:
            tracks, mask, targets = tracks.to(device), mask.to(device), targets.to(device)

            z = encoder(tracks, mask)
            recon = decoder(z)
            recon_err = F.mse_loss(recon, targets, reduction="none").mean(dim=1)

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
