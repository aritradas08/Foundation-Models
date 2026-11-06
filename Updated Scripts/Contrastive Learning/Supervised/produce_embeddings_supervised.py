import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from pretrain_supervised import raw_cols, n_track_feats, engineer_tracks, SetTransformerEncoder


# paths
data_path = "/global/cfs/cdirs/m4958/usr/aritra08/Anomaly_Detection/data/combined_pu0_1500k.parquet"
ckpt_dir = "/global/cfs/cdirs/m4958/usr/aritra08/cont_learn/supervised/results/d64/checkpoints"
encoder_ckpt = os.path.join(ckpt_dir, "encoder_supervised_simclr.pt")
scaler_ckpt = os.path.join(ckpt_dir, "scaler.pkl")

out_dir = "/global/cfs/cdirs/m4958/usr/aritra08/cont_learn/supervised/results/d64/embeddings"
out_path = os.path.join(out_dir, "z_embeddings_all_events.npz")

# embed both SM and BSM events
include_labels = None

batch_size = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class InferenceEventDataset(Dataset):
    def __init__(self, df, scaler, include_labels=None):
        if include_labels is not None:
            df = df[df["label"].isin(include_labels)].reset_index(drop=True)

        self.labels = df["label"].values.astype(np.int64)
        self.event_tracks = []

        for i in range(len(df)):
            arr_raw = np.column_stack([df[c].iloc[i] for c in raw_cols]).astype(np.float32)
            tokens = engineer_tracks(arr_raw)
            self.event_tracks.append(scaler.transform(tokens).astype(np.float32))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.event_tracks[idx], self.labels[idx]


def inference_collate_fn(batch):
    # pad to the longest event in the batch, mask is true for real tracks
    tracks_list, labels = zip(*batch)
    bs = len(tracks_list)
    max_len = max(t.shape[0] for t in tracks_list)

    padded = np.zeros((bs, max_len, n_track_feats), dtype=np.float32)
    mask = np.zeros((bs, max_len), dtype=bool)
    n_tracks = np.zeros(bs, dtype=np.int64)

    for i, t in enumerate(tracks_list):
        padded[i, :len(t)] = t
        mask[i, :len(t)] = True
        n_tracks[i] = len(t)

    return (torch.from_numpy(padded), torch.from_numpy(mask),
            torch.tensor(labels, dtype=torch.long), torch.from_numpy(n_tracks))


def produce_embeddings():
    os.makedirs(out_dir, exist_ok=True)

    # useing scaler from training
    with open(scaler_ckpt, "rb") as f:
        scaler = pickle.load(f)

    # freezing the trained encoder
    encoder = SetTransformerEncoder().to(device)
    encoder.load_state_dict(torch.load(encoder_ckpt, map_location=device))
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    df = pd.read_parquet(data_path)
    dataset = InferenceEventDataset(df, scaler, include_labels=include_labels)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # keep order so z[i] matches row i
        collate_fn=inference_collate_fn,
        num_workers=4,
    )

    all_z, all_labels, all_n_tracks = [], [], []

    with torch.no_grad():
        for tracks, mask, labels, n_tracks in loader:
            z = encoder(tracks.to(device), mask.to(device))
            all_z.append(z.cpu().numpy())
            all_labels.append(labels.numpy())
            all_n_tracks.append(n_tracks.numpy())

    z_all = np.concatenate(all_z, axis=0)
    labels_all = np.concatenate(all_labels, axis=0)
    n_tracks_all = np.concatenate(all_n_tracks, axis=0)

    print(f"z shape: {z_all.shape}, per-dim std: {z_all.std(axis=0)}")

    np.savez(
        out_path,
        z=z_all.astype(np.float32),
        labels=labels_all,
        n_tracks=n_tracks_all,
    )
    print(f"saved to {out_path}")

    return z_all, labels_all, n_tracks_all


if __name__ == "__main__":
    produce_embeddings()
