#Pre-training script for self-supervised contrastive training

import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

DATA_PATH = "/global/cfs/cdirs/m4958/usr/aritra08/Anomaly_Detection/datasets/150k_events/combined_pu0_600k.parquet"
SAVE_DIR = "/global/cfs/cdirs/m4958/usr/aritra08/ssl_eucaif/results/ssl_contrastive/sm_only_v3"
CKPT_DIR = os.path.join(SAVE_DIR, "checkpoints")

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

RAW_COLS = ["d0", "z0", "theta", "p", "eta", "phi", "pt"]
RAW_IDX = {name: i for i, name in enumerate(RAW_COLS)}
TRACK_FEAT_NAMES = ["d0", "z0", "p", "pt", "tx", "ty", "tz"]
N_TRACK_FEATS = 7

SM_LABELS = [0, 1, 2] 
BSM_LABELS = [3]      
ALL_PROCESS_NAMES = ["ttbar", "ggf", "dihiggs", "higgs_portal"]


# Hyperparameters
LATENT_DIM = 64  
MODEL_DIM = 64  
N_HEADS = 4
N_LAYERS = 4
FFN_DIM = 256
DROPOUT = 0.025

PROJ_HIDDEN_DIM = 64

BATCH_SIZE = 512
N_EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-6
TEMPERATURE = 0.5

NOISE_STD = {    
    "d0": 0.1,
    "z0": 0.1,
    "p": 0.05,
    "pt": 0.05,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Symlog transformation
def symlog(x):
    return np.sign(x) * np.log1p(np.abs(x))

#Do feature engineering
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

#Used toaugment the events by doing random roations and adding gaussian noise
def augment_event(tokens: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    view = tokens.copy()
	tx_idx, ty_idx = TRACK_FEAT_NAMES.index("tx"), TRACK_FEAT_NAMES.index("ty")


	mean = scaler.mean_[[tx_idx, ty_idx]]
	std  = scaler.scale_[[tx_idx, ty_idx]]
	tx_ty_phys = view[:, [tx_idx, ty_idx]] * std + mean

	angle = np.random.uniform(0, 2 * np.pi)
	cos_a, sin_a = np.cos(angle), np.sin(angle)
	rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
	tx_ty_rot = tx_ty_phys @ rot.T

	view[:, [tx_idx, ty_idx]] = (tx_ty_rot - mean) / std

    for feat_name, noise_std in NOISE_STD.items():
        idx = TRACK_FEAT_NAMES.index(feat_name)
        view[:, idx] = view[:, idx] + np.random.normal(0.0, noise_std, size=view.shape[0])

    return view.astype(np.float32)


#Used to make 2 augmented views per event
class ContrastiveEventDataset(Dataset):
    def __init__(self, df, scaler, include_labels=None):
        if include_labels is not None:
            df = df[df["label"].isin(include_labels)].reset_index(drop=True)

        self.scaler = scaler
        self.event_tracks = []   # list of (N_tracks, 7) scaled arrays

        for i in range(len(df)):
            arr_raw = np.column_stack(
                [df[c].iloc[i] for c in RAW_COLS]
            ).astype(np.float32)
            tokens = engineer_tracks(arr_raw)
            tokens = scaler.transform(tokens)
            self.event_tracks.append(tokens.astype(np.float32))

    def __len__(self):
        return len(self.event_tracks)

    def __getitem__(self, idx):
        tokens = self.event_tracks[idx]
        view_1 = augment_event(tokens, self.scaler)
        view_2 = augment_event(tokens, self.scaler)
        return view_1, view_2

# Used to create batches 
def contrastive_collate_fn(batch):
    views_1, views_2 = zip(*batch)
    batch_size = len(views_1)
    max_len = max(max(v.shape[0] for v in views_1),
                   max(v.shape[0] for v in views_2))

    def pad_stack(views):
        padded = np.zeros((batch_size, max_len, N_TRACK_FEATS), dtype=np.float32)
        mask   = np.zeros((batch_size, max_len), dtype=bool)
        for i, v in enumerate(views):
            n = v.shape[0]
            padded[i, :n] = v
            mask[i, :n] = True
        return torch.from_numpy(padded), torch.from_numpy(mask)

    x1, mask1 = pad_stack(views_1)
    x2, mask2 = pad_stack(views_2)
    return x1, mask1, x2, mask2


# The transformer encoder part. Uses CLS token as the learned summary vector
class SetTransformerEncoder(nn.Module):
    def __init__(self, in_dim=N_TRACK_FEATS, model_dim=MODEL_DIM,
                 n_heads=N_HEADS, n_layers=N_LAYERS, ffn_dim=FFN_DIM,
                 latent_dim=LATENT_DIM, dropout=DROPOUT):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, model_dim)
        self.cls_token  = nn.Parameter(torch.randn(1, 1, model_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_proj = nn.Linear(model_dim, latent_dim)

    def forward(self, tracks, mask):
        B = tracks.shape[0]
        x = self.input_proj(tracks)                          

        cls = self.cls_token.expand(B, -1, -1)                
        x = torch.cat([cls, x], dim=1)                        

        cls_mask = torch.ones(B, 1, dtype=torch.bool, device=mask.device)
        full_mask = torch.cat([cls_mask, mask], dim=1)        

        key_padding_mask = ~full_mask

        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        cls_out = x[:, 0, :] # pooled CLS token, shaoe is (B, model_dim)

        z = self.out_proj(cls_out) # (B, latent_dim)
        return z


class ProjectionHead(nn.Module):
    def __init__(self, in_dim=LATENT_DIM, hidden_dim=PROJ_HIDDEN_DIM, out_dim=LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z):
        return self.net(z)


# Self-supervised sim-clr loss
def self_supervised_simclr_loss(p1: torch.Tensor, p2: torch.Tensor, temperature: float):
    B = p1.shape[0]
    device = p1.device

    p1 = F.normalize(p1, dim=1)
    p2 = F.normalize(p2, dim=1)

    z = torch.cat([p1, p2], dim=0)                             

    # creating the pairwise cosine similarity matrix
    sim = torch.matmul(z, z.T) / temperature     #Temperature is used to magnify the differences

    # masking out the self-similarity
    self_mask = torch.eye(2 * B, dtype=torch.bool, device=device)
    sim.masked_fill_(self_mask, float("-inf"))

    pos_idx = torch.cat([
        torch.arange(B, 2 * B, device=device),
        torch.arange(0, B, device=device),
    ])
    
    loss = F.cross_entropy(sim, pos_idx)
    return loss


#Training part
def fit_scaler_on_sm(df, include_labels=SM_LABELS):
    df_sm = df[df["label"].isin(include_labels)].reset_index(drop=True)

    all_tracks = []
    for i in range(len(df_sm)):
        arr_raw = np.column_stack(
            [df_sm[c].iloc[i] for c in RAW_COLS]
        ).astype(np.float32)
        tokens = engineer_tracks(arr_raw)
        all_tracks.append(tokens)
    all_tracks = np.concatenate(all_tracks, axis=0)

    scaler = StandardScaler()
    scaler.fit(all_tracks)
    print(f"Track scaler fit on tracks of SM processes")
    return scaler


def train_self_supervised_contrastive():
    df = pd.read_parquet(DATA_PATH)
    print(f"Data Loaded")

    scaler = fit_scaler_on_sm(df, include_labels=SM_LABELS)

    train_dataset = ContrastiveEventDataset(df, scaler, include_labels=SM_LABELS)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=contrastive_collate_fn,
        drop_last=True,     
        num_workers=4,
    )
    print(f"Training events: {len(train_dataset)} SM events")

    encoder = SetTransformerEncoder().to(DEVICE)
    proj_head = ProjectionHead().to(DEVICE)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(proj_head.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

    history = {"epoch": [], "loss": []}

    for epoch in range(1, N_EPOCHS + 1):
        encoder.train()
        proj_head.train()
        epoch_loss = 0.0
        n_batches = 0

        for x1, mask1, x2, mask2 in train_loader:
            x1, mask1 = x1.to(DEVICE), mask1.to(DEVICE)
            x2, mask2 = x2.to(DEVICE), mask2.to(DEVICE)

            z1 = encoder(x1, mask1)           
            z2 = encoder(x2, mask2)           

            p1 = proj_head(z1)                  
            p2 = proj_head(z2)

            loss = self_supervised_simclr_loss(p1, p2, TEMPERATURE)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / n_batches
        history["epoch"].append(epoch)
        history["loss"].append(avg_loss)
        print(f"Epoch {epoch}; loss = {avg_loss:.2f}; lr = {scheduler.get_last_lr()[0]:.2e}")

    torch.save(encoder.state_dict(), os.path.join(CKPT_DIR, "encoder_ssl_simclr.pt"))
    torch.save(proj_head.state_dict(), os.path.join(CKPT_DIR, "proj_head_ssl_simclr.pt"))

    with open(os.path.join(CKPT_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(os.path.join(SAVE_DIR, "training_history.csv"), index=False)

    print(f"Training done.")
    return encoder, proj_head, scaler, history


if __name__ == "__main__":
    train_self_supervised_contrastive()
