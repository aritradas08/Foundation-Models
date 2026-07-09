# Creates the embeddings for the ssl contrastive learning case

import os
import pickle
import numpy as np
import pandas as pd
import torch


from pretrain_ssl_contrastive import (
    RAW_COLS, N_TRACK_FEATS, ALL_PROCESS_NAMES,
    engineer_tracks, SetTransformerEncoder,
)


DATA_PATH = "/global/cfs/cdirs/m4958/usr/aritra08/Anomaly_Detection/datasets/150k_events/combined_pu0_600k.parquet"
CKPT_DIR = "/global/cfs/cdirs/m4958/usr/aritra08/ssl_eucaif/results/ssl_contrastive/sm_only_v3/checkpoints"
ENCODER_CKPT = os.path.join(CKPT_DIR, "encoder_ssl_simclr.pt")
SCALER_CKPT  = os.path.join(CKPT_DIR, "scaler.pkl")
OUT_DIR = "/global/cfs/cdirs/m4958/usr/aritra08/ssl_eucaif/results/ssl_contrastive/sm_only_v3/embeddings"
OUT_PATH = os.path.join(OUT_DIR, "z_embeddings_all_events.npz")

INCLUDE_LABELS = None

BATCH_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_PRINT_PER_LABEL = 3
N_PRINT_RANDOM    = 5

class InferenceEventDataset(torch.utils.data.Dataset):
    def __init__(self, df, scaler, include_labels=None):
        if include_labels is not None:
            df = df[df["label"].isin(include_labels)].reset_index(drop=True)

        self.labels = df["label"].values.astype(np.int64)
        self.event_tracks = []

        for i in range(len(df)):
            arr_raw = np.column_stack(
                [df[c].iloc[i] for c in RAW_COLS]
            ).astype(np.float32)
            tokens = engineer_tracks(arr_raw)
            tokens = scaler.transform(tokens)
            self.event_tracks.append(tokens.astype(np.float32))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.event_tracks[idx], self.labels[idx]

# Pads to N_max and builds the attention mask
def inference_collate_fn(batch):
    tracks_list, labels = zip(*batch)
    batch_size = len(tracks_list)
    max_len = max(t.shape[0] for t in tracks_list)

    padded = np.zeros((batch_size, max_len, N_TRACK_FEATS), dtype=np.float32)
    mask   = np.zeros((batch_size, max_len), dtype=bool)
    n_tracks = np.zeros(batch_size, dtype=np.int64)

    for i, t in enumerate(tracks_list):
        n = t.shape[0]
        padded[i, :n] = t
        mask[i, :n] = True
        n_tracks[i] = n

    return (torch.from_numpy(padded), torch.from_numpy(mask),
            torch.tensor(labels, dtype=torch.long), torch.from_numpy(n_tracks))


#Loads frozen encoder, creates embeddings
def produce_embeddings():
    os.makedirs(OUT_DIR, exist_ok=True)

    #Loading the scaler values, same as the one used during (pre)training
    with open(SCALER_CKPT, "rb") as f:
        scaler = pickle.load(f)

    # rebuilds encoder architecture
    encoder = SetTransformerEncoder().to(DEVICE)   
    encoder.load_state_dict(torch.load(ENCODER_CKPT, map_location=DEVICE))
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False    # Doesn't modify the encoder weights,"freezing" it

    df = pd.read_parquet(DATA_PATH)

    dataset = InferenceEventDataset(df, scaler, include_labels=INCLUDE_LABELS)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,           
        collate_fn=inference_collate_fn,
        num_workers=4,
    )

    all_z = []
    all_labels = []
    all_n_tracks = []

    with torch.no_grad():
        for tracks, mask, labels, n_tracks in loader:
            tracks, mask = tracks.to(DEVICE), mask.to(DEVICE)
            z = encoder(tracks, mask)              
            all_z.append(z.cpu().numpy())
            all_labels.append(labels.numpy())
            all_n_tracks.append(n_tracks.numpy())

    z_all = np.concatenate(all_z, axis=0)
    labels_all = np.concatenate(all_labels, axis=0)
    n_tracks_all = np.concatenate(all_n_tracks, axis=0)

    print(f"z shape: {z_all.shape}")

    np.savez(
        OUT_PATH,
        z=z_all.astype(np.float32),
        labels=labels_all,
        n_tracks=n_tracks_all,
    )

    return z_all, labels_all, n_tracks_all


if __name__ == "__main__":
    produce_embeddings()
