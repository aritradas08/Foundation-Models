# pre-training script for supervised contrastive training

import os
import math
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.preprocessing import StandardScaler


data_path = "/global/cfs/cdirs/m4958/usr/aritra08/Anomaly_Detection/data/combined_pu0_1500k.parquet"
save_dir = "/global/cfs/cdirs/m4958/usr/aritra08/cont_learn/supervised/results/d64"
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

latent_dim = 64
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
temperature = 0.1

warmup_steps = 500
grad_clip = 1.0

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


class SupervisedEventDataset(Dataset):
    def __init__(self, df, scaler, include_labels=sm_labels):
        df = df[df["label"].isin(include_labels)].reset_index(drop=True)

        self.labels = df["label"].values.astype(np.int64)
        self.event_tracks = []
        for i in range(len(df)):
            tokens = engineer_tracks(event_to_array(df, i))
            self.event_tracks.append(scaler.transform(tokens).astype(np.float32))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.event_tracks[idx], self.labels[idx]


def supervised_collate_fn(batch):
    tracks_list, labels = zip(*batch)
    bs = len(tracks_list)
    max_len = max(t.shape[0] for t in tracks_list)

    padded = np.zeros((bs, max_len, n_track_feats), dtype=np.float32)
    mask = np.zeros((bs, max_len), dtype=bool)

    for i, t in enumerate(tracks_list):
        padded[i, :len(t)] = t
        mask[i, :len(t)] = True

    return (torch.from_numpy(padded), torch.from_numpy(mask),
            torch.tensor(labels, dtype=torch.long))


class BalancedClassSampler(Sampler):
    # every batch has the same number of events from each class/process
    def __init__(self, labels, batch_size, classes, drop_last=True):
        self.labels = np.asarray(labels)
        self.batch_size = batch_size
        self.classes = classes
        self.drop_last = drop_last
        self.class_indices = {c: np.where(self.labels == c)[0] for c in classes}
        self.n_per_class = batch_size // len(classes)
        self.n_batches = min(len(idx) for idx in self.class_indices.values()) // self.n_per_class

    def __iter__(self):
        shuffled = {c: np.random.permutation(idx) for c, idx in self.class_indices.items()}
        for b in range(self.n_batches):
            batch = []
            for c in self.classes:
                start = b * self.n_per_class
                batch.extend(shuffled[c][start:start + self.n_per_class].tolist())
            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.n_batches


class SetTransformerEncoder(nn.Module):
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


def supervised_simclr_loss(p, labels, temperature):
    b = p.shape[0]
    dev = p.device

    p = F.normalize(p, dim=1)
    sim = torch.matmul(p, p.T) / temperature

    sim_max, _ = sim.max(dim=1, keepdim=True)
    sim = sim - sim_max.detach()

    self_mask = torch.eye(b, dtype=torch.bool, device=dev)

    exp_sim = torch.exp(sim).masked_fill(self_mask, 0.0)
    denom = exp_sim.sum(dim=1, keepdim=True)
    log_prob = sim - torch.log(denom + 1e-12)

    labels = labels.view(-1, 1)
    positive_mask = (labels == labels.T) & (~self_mask)

    n_positives = positive_mask.sum(dim=1)
    valid = n_positives > 0

    sum_log_prob_pos = (log_prob * positive_mask.float()).sum(dim=1)
    mean_log_prob_pos = sum_log_prob_pos[valid] / n_positives[valid].float()

    return -mean_log_prob_pos.mean()


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

    assert set(sm_labels).isdisjoint(set(bsm_labels))

    scaler = fit_scaler_on_sm(df)

    train_dataset = SupervisedEventDataset(df, scaler, include_labels=sm_labels)
    sampler = BalancedClassSampler(train_dataset.labels, batch_size, sm_labels)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        collate_fn=supervised_collate_fn,
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

        for tracks, mask, labels in train_loader:
            tracks, mask, labels = tracks.to(device), mask.to(device), labels.to(device)

            z = encoder(tracks, mask)

            if n_batches == 0:
                with torch.no_grad():
                    embed_std = z.std(dim=0).mean().item()

            loss = supervised_simclr_loss(proj_head(z), labels, temperature)

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

    torch.save(encoder.state_dict(), os.path.join(ckpt_dir, "encoder_supervised_simclr.pt"))
    torch.save(proj_head.state_dict(), os.path.join(ckpt_dir, "proj_head_supervised_simclr.pt"))

    with open(os.path.join(ckpt_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    pd.DataFrame(history).to_csv(os.path.join(save_dir, "training_history.csv"), index=False)

    config = dict(
        model_dim=model_dim, n_heads=n_heads, n_layers=n_layers, ffn_dim=ffn_dim,
        dropout=dropout, latent_dim=latent_dim, proj_hidden_dim=proj_hidden_dim,
        batch_size=batch_size, epochs=n_epochs, lr=lr,
        weight_decay=weight_decay, temperature=temperature,
        warmup_steps=warmup_steps, grad_clip=grad_clip,
    )
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    return encoder, proj_head, scaler, history


if __name__ == "__main__":
    train()
