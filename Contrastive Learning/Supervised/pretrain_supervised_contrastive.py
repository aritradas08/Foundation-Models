#Pre-training script for supervised contrastive training

import os
import numpy as np
import pandas as pd
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


DATA_PATH = "/global/cfs/cdirs/m4958/usr/aritra08/Anomaly_Detection/datasets/150k_events/combined_pu0_600k.parquet"
SAVE_DIR = "/global/cfs/cdirs/m4958/usr/aritra08/supervised_cl_eucaif/results/supervised_contrastive/sm_only_v3"
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
TEMPERATURE = 0.1  

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def symlog(x):
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

class SupervisedEventDataset(Dataset):
    def __init__(self, df, scaler, include_labels=SM_LABELS):
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


def supervised_collate_fn(batch):
    tracks_list, labels = zip(*batch)
    batch_size = len(tracks_list)
    max_len = max(t.shape[0] for t in tracks_list)

    padded = np.zeros((batch_size, max_len, N_TRACK_FEATS), dtype=np.float32)
    mask = np.zeros((batch_size, max_len), dtype=bool)

    for i, t in enumerate(tracks_list):
        n = t.shape[0]
        padded[i, :n] = t
        mask[i, :n] = True

    return (torch.from_numpy(padded), torch.from_numpy(mask),
            torch.tensor(labels, dtype=torch.long))


class BalancedClassSampler(torch.utils.data.Sampler):
    def __init__(self, labels, batch_size, classes, drop_last=True):
        self.labels = np.asarray(labels)
        self.batch_size = batch_size
        self.classes = classes
        self.drop_last = drop_last
        self.class_indices = {c: np.where(self.labels == c)[0] for c in classes}
        self.n_per_class = batch_size // len(classes)  # even split across classes
        self.n_batches = min(len(idx) for idx in self.class_indices.values()) // self.n_per_class

    def __iter__(self):
        shuffled = {c: np.random.permutation(idx) for c, idx in self.class_indices.items()}
        for b in range(self.n_batches):
            batch = []
            for c in self.classes:
                start = b * self.n_per_class
                end = start + self.n_per_class
                batch.extend(shuffled[c][start:end].tolist())
            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.n_batches


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
        cls_out = x[:, 0, :]

        z = self.out_proj(cls_out)
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

#Supervised CLR loss
def supervised_simclr_loss(p: torch.Tensor, labels: torch.Tensor, temperature: float):
    B = p.shape[0]
    device = p.device

    p = F.normalize(p, dim=1)

    # full pairwise cosine similarity, magnified by temperature, like before
    sim = torch.matmul(p, p.T) / temperature                    

    sim_max, _ = sim.max(dim=1, keepdim=True)
    sim = sim - sim_max.detach()

    # masking out self-similarity 
    self_mask = torch.eye(B, dtype=torch.bool, device=device)

    exp_sim = torch.exp(sim)
    exp_sim = exp_sim.masked_fill(self_mask, 0.0)

    denom = exp_sim.sum(dim=1, keepdim=True)                     
    log_prob = sim - torch.log(denom + 1e-12)                    

    labels = labels.view(-1, 1)
    positive_mask = (labels == labels.T) & (~self_mask)          

    n_positives = positive_mask.sum(dim=1)                       
    valid = n_positives > 0

    sum_log_prob_pos = (log_prob * positive_mask.float()).sum(dim=1)
    mean_log_prob_pos = sum_log_prob_pos[valid] / n_positives[valid].float()

    loss = -mean_log_prob_pos.mean()
    return loss



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


#Training part
def train_supervised_contrastive():
    df = pd.read_parquet(DATA_PATH)

    assert set(SM_LABELS).isdisjoint(set(BSM_LABELS))

    scaler = fit_scaler_on_sm(df, include_labels=SM_LABELS)

    train_dataset = SupervisedEventDataset(df, scaler, include_labels=SM_LABELS)
    print(f"Training: {len(train_dataset)} SM events")
    for label_val in SM_LABELS:
        n = (train_dataset.labels == label_val).sum()
        print(f"label={label_val}=({ALL_PROCESS_NAMES[label_val]}): {n} events")

    sampler = BalancedClassSampler(train_dataset.labels, BATCH_SIZE, SM_LABELS)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        collate_fn=supervised_collate_fn,
        num_workers=4,
    )

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

        for tracks, mask, labels in train_loader:
            tracks, mask, labels = tracks.to(DEVICE), mask.to(DEVICE), labels.to(DEVICE)

            z = encoder(tracks, mask)          
            p = proj_head(z)                   

            loss = supervised_simclr_loss(p, labels, TEMPERATURE)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / n_batches
        history["epoch"].append(epoch)
        history["loss"].append(avg_loss)
        print(f"Epoch {epoch:3d}; loss = {avg_loss:.2f}; lr = {scheduler.get_last_lr()[0]:.2e}")

    torch.save(encoder.state_dict(), os.path.join(CKPT_DIR, "encoder_supervised_simclr.pt"))
    torch.save(proj_head.state_dict(), os.path.join(CKPT_DIR, "proj_head_supervised_simclr.pt"))
    
    with open(os.path.join(CKPT_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(os.path.join(SAVE_DIR, "training_history.csv"), index=False)

    return encoder, proj_head, scaler, history


if __name__ == "__main__":
    train_supervised_contrastive()
