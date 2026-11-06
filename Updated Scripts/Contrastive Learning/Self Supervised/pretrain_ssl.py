# pre-training script for self-supervised contrastive training

import os
import math
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

data_path = "/global/cfs/cdirs/m4958/usr/aritra08/Anomaly_Detection/data/combined_pu0_1500k.parquet"
save_dir = "/global/cfs/cdirs/m4958/usr/aritra08/cont_learn/ssl/results/d2"
ckpt_dir = os.path.join(save_dir, "checkpoints")

os.makedirs(save_dir, exist_ok=True)
os.makedirs(ckpt_dir, exist_ok=True)

raw_cols = ["d0", "z0", "theta", "p", "eta", "phi", "pt"]
raw_idx = {name: i for i, name in enumerate(raw_cols)}
track_feat_names = ["d0", "z0", "p", "pt", "tx", "ty", "tz"]
n_track_feats = 7

sm_labels = [0, 1, 2]
bsm_labels = [3, 4]
all_process_names = ["ttbar", "ggf", "dihiggs", "higgs_portal", "hidden_valley"]

# hyperparameters
latent_dim = 2
model_dim = 64
n_heads = 8
n_layers = 8
ffn_dim = 256
dropout = 0.025
proj_hidden_dim = 64

batch_size = 512
n_epochs = 20
lr = 1e-3
weight_decay = 1e-6
temperature = 0.5

warmup_steps = 500
grad_clip = 1.0

noise_std = {
    "d0": 0.1,
    "z0": 0.1,
    "p": 0.05,
    "pt": 0.05,
}

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


def augment_event(tokens, scaler):
    # random rotation in the transverse plane + gaussian noise
    view = tokens.copy()
    tx_idx = track_feat_names.index("tx")
    ty_idx = track_feat_names.index("ty")

    mean = scaler.mean_[[tx_idx, ty_idx]]
    std = scaler.scale_[[tx_idx, ty_idx]]
    tx_ty = view[:, [tx_idx, ty_idx]] * std + mean

    angle = np.random.uniform(0, 2 * np.pi)
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    tx_ty = tx_ty @ rot.T

    view[:, [tx_idx, ty_idx]] = (tx_ty - mean) / std

    for name, sigma in noise_std.items():
        idx = track_feat_names.index(name)
        view[:, idx] += np.random.normal(0.0, sigma, size=view.shape[0])

    return view.astype(np.float32)


class ContrastiveEventDataset(Dataset):
    # returns two augmented views of each event
    def __init__(self, df, scaler, include_labels=None):
        if include_labels is not None:
            df = df[df["label"].isin(include_labels)].reset_index(drop=True)

        self.scaler = scaler
        self.event_tracks = []
        for i in range(len(df)):
            tokens = engineer_tracks(event_to_array(df, i))
            self.event_tracks.append(scaler.transform(tokens).astype(np.float32))

    def __len__(self):
        return len(self.event_tracks)

    def __getitem__(self, idx):
        tokens = self.event_tracks[idx]
        return augment_event(tokens, self.scaler), augment_event(tokens, self.scaler)


def contrastive_collate_fn(batch):
    views_1, views_2 = zip(*batch)
    bs = len(views_1)
    max_len = max(max(v.shape[0] for v in views_1), max(v.shape[0] for v in views_2))

    def pad_stack(views):
        padded = np.zeros((bs, max_len, n_track_feats), dtype=np.float32)
        mask = np.zeros((bs, max_len), dtype=bool)
        for i, v in enumerate(views):
            padded[i, :len(v)] = v
            mask[i, :len(v)] = True
        return torch.from_numpy(padded), torch.from_numpy(mask)

    x1, mask1 = pad_stack(views_1)
    x2, mask2 = pad_stack(views_2)
    return x1, mask1, x2, mask2


class SetTransformerEncoder(nn.Module):
    # pre-norm transformer, cls token is the summary vector
    def __init__(self, in_dim=n_track_feats, model_dim=model_dim, n_heads=n_heads,
                 n_layers=n_layers, ffn_dim=ffn_dim, latent_dim=latent_dim, dropout=dropout):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, model_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, model_dim) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(model_dim)
        self.out_proj = nn.Linear(model_dim, latent_dim)

    def forward(self, tracks, mask):
        b = tracks.shape[0]
        x = self.input_proj(tracks)

        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)

        cls_mask = torch.ones(b, 1, dtype=torch.bool, device=mask.device)
        full_mask = torch.cat([cls_mask, mask], dim=1)

        x = self.transformer(x, src_key_padding_mask=~full_mask)
        cls_out = self.out_norm(x[:, 0, :])
        return self.out_proj(cls_out)


class ProjectionHead(nn.Module):
    def __init__(self, in_dim=latent_dim, hidden_dim=proj_hidden_dim, out_dim=latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z):
        return self.net(z)


def simclr_loss(p1, p2, temperature):
    b = p1.shape[0]
    dev = p1.device

    p1 = F.normalize(p1, dim=1)
    p2 = F.normalize(p2, dim=1)
    z = torch.cat([p1, p2], dim=0)

    sim = torch.matmul(z, z.T) / temperature
    sim.masked_fill_(torch.eye(2 * b, dtype=torch.bool, device=dev), float("-inf"))

    pos_idx = torch.cat([torch.arange(b, 2 * b, device=dev), torch.arange(0, b, device=dev)])
    return F.cross_entropy(sim, pos_idx)


def fit_scaler_on_sm(df, include_labels=sm_labels):
    df_sm = df[df["label"].isin(include_labels)].reset_index(drop=True)
    all_tracks = np.concatenate(
        [engineer_tracks(event_to_array(df_sm, i)) for i in range(len(df_sm))], axis=0
    )
    scaler = StandardScaler()
    scaler.fit(all_tracks)
    return scaler


def train():
    df = pd.read_parquet(data_path)
    scaler = fit_scaler_on_sm(df)

    train_dataset = ContrastiveEventDataset(df, scaler, include_labels=sm_labels)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=contrastive_collate_fn,
        drop_last=True,
        num_workers=4,
    )

    encoder = SetTransformerEncoder().to(device)
    proj_head = ProjectionHead().to(device)

    params = list(encoder.parameters()) + list(proj_head.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    total_steps = n_epochs * len(train_loader)
    lr_fn = lambda s: min(1.0, (s + 1) / warmup_steps) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(total_steps, 1))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)

    history = {"epoch": [], "loss": [], "embed_std": []}

    for epoch in range(1, n_epochs + 1):
        encoder.train()
        proj_head.train()
        epoch_loss = 0.0
        n_batches = 0
        embed_std = None

        for x1, mask1, x2, mask2 in train_loader:
            x1, mask1 = x1.to(device), mask1.to(device)
            x2, mask2 = x2.to(device), mask2.to(device)

            z1 = encoder(x1, mask1)
            z2 = encoder(x2, mask2)

            if n_batches == 0:
                with torch.no_grad():
                    embed_std = z1.std(dim=0).mean().item()

            loss = simclr_loss(proj_head(z1), proj_head(z2), temperature)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, grad_clip)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        history["epoch"].append(epoch)
        history["loss"].append(avg_loss)
        history["embed_std"].append(embed_std)
        print(f"epoch {epoch}: loss {avg_loss:.4f}, embed_std {embed_std:.4f}", flush=True)

    torch.save(encoder.state_dict(), os.path.join(ckpt_dir, "encoder_ssl_simclr.pt"))
    torch.save(proj_head.state_dict(), os.path.join(ckpt_dir, "proj_head_ssl_simclr.pt"))

    with open(os.path.join(ckpt_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    pd.DataFrame(history).to_csv(os.path.join(save_dir, "training_history.csv"), index=False)

    config = dict(
        model_dim=model_dim, n_heads=n_heads, n_layers=n_layers, ffn_dim=ffn_dim,
        dropout=dropout, latent_dim=latent_dim, proj_hidden_dim=proj_hidden_dim,
        noise_std=noise_std, batch_size=batch_size, epochs=n_epochs, lr=lr,
        weight_decay=weight_decay, temperature=temperature,
        warmup_steps=warmup_steps, grad_clip=grad_clip,
    )
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    return encoder, proj_head, scaler, history


if __name__ == "__main__":
    train()
