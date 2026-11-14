#!/usr/bin/env python3
"""
Unified training and disjoint embedding extraction pipeline for ColliderML Foundation Models.
Supports:
  - cache_only: pre-populate /pscratch cache for pretrain_sm_900k and eval_downstream
  - extract_existing: extract disjoint eval_downstream embeddings for existing checkpoints + random_init
  - train_ssl_kinematic: train Continuous SSL Contrastive with kinematically consistent GPU augmentations
  - train_ae_transformer: train standardized Pre-LN GELU 8L/8H Transformer 28-D Summary Bottleneck
  - train_ssl_tokens: train 8L/8H Pre-LN GELU TokenSetTransformerEncoder on VQ-VAE (K=8192) tokens
  - train_supervised_e2e: train end-to-end supervised 8L/8H SetTransformer classifier on 3 SM classes
"""
import argparse
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(REPO_ROOT / "Tokenization-scripts" / "scripts"))

from common import (
    AEDecoder,
    MAX_TRACKS_DEFAULT,
    MLPEncoder,
    N_TRACK_FEATS,
    ProjectionHead,
    SM_LABELS,
    SetTransformerEncoder,
    augment_batch_kinematic_torch,
    compute_summaries28_np,
    fit_track_scaler,
    load_events_fast,
    raw6_to_feat7_np,
    raw6_to_vqvae6_np,
)

DATA_DIR = Path("/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/data")
PRETRAIN_PARQUET = DATA_DIR / "pretrain_sm_900k.parquet"
EVAL_PARQUET = DATA_DIR / "eval_downstream.parquet"
RESULTS_BASE = Path("/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/results")
EMBED_DIR = RESULTS_BASE / "embeddings"
CKPT_BASE = RESULTS_BASE / "checkpoints"


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def simclr_nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T) / temperature
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, -1e9)
    targets = torch.cat([torch.arange(B, 2 * B, device=z.device), torch.arange(0, B, device=z.device)])
    return F.cross_entropy(sim, targets)


@torch.no_grad()
def extract_set_transformer_embeddings(
    encoder: nn.Module,
    feat7_unscaled: np.ndarray,
    mask: np.ndarray,
    scaler: StandardScaler,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    encoder.eval()
    N, T, F_dim = feat7_unscaled.shape
    mean = scaler.mean_.astype(np.float32).reshape(1, 1, F_dim)
    scale = scaler.scale_.astype(np.float32).reshape(1, 1, F_dim)

    all_z = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        m_b = mask[start:end]
        max_t = max(int(m_b.sum(axis=1).max()), 1)
        m_b = m_b[:, :max_t]
        x_b = (feat7_unscaled[start:end, :max_t, :] - mean) / scale
        x_b[~m_b] = 0.0
        x_t = torch.from_numpy(x_b).to(device, non_blocking=True)
        m_t = torch.from_numpy(m_b).to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            z = encoder(x_t, m_t)
        all_z.append(z.float().cpu().numpy())
    return np.concatenate(all_z, axis=0)


def save_embedding_npz(
    name: str,
    z: np.ndarray,
    eval_data: dict[str, np.ndarray],
    extra: dict | None = None,
):
    EMBED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EMBED_DIR / f"{name}.npz"
    payload = {
        "z": z.astype(np.float32),
        "labels": eval_data["labels"].astype(np.int64),
        "event_ids": eval_data["event_ids"].astype(np.int64),
        "n_tracks": eval_data["lengths"].astype(np.int32),
    }
    if extra:
        payload.update(extra)
    np.savez(out_path, **payload)
    print(f"[save_embedding_npz] Saved {out_path} -> z.shape={z.shape}")


def run_extract_existing(device: torch.device):
    eval_data = load_events_fast(EVAL_PARQUET)
    feat7_eval = raw6_to_feat7_np(eval_data["raw6"], eval_data["mask"])
    mask_eval = eval_data["mask"]

    # 1. Supervised Contrastive (SupCon) d8, d32, d64
    for d in [8, 32, 64]:
        if (EMBED_DIR / f"supcon_d{d}.npz").exists():
            continue
        cdir = Path(f"/global/cfs/cdirs/m4958/usr/aritra08/cont_learn/supervised/results/d{d}/checkpoints")
        with open(cdir / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        enc = SetTransformerEncoder(latent_dim=d, norm_first=True, activation="gelu", use_out_norm=True).to(device)
        enc.load_state_dict(torch.load(cdir / "encoder_supervised_simclr.pt", map_location=device))
        z = extract_set_transformer_embeddings(enc, feat7_eval, mask_eval, scaler, device)
        save_embedding_npz(f"supcon_d{d}", z, eval_data)

    # 2. Continuous SSL (legacy independent p/pt noise) d8, d32, d64
    ssl_paths = {
        8: Path("/global/cfs/cdirs/m4958/usr/aritra08/cont_learn/ssl/results/d8/checkpoints"),
        32: Path("/global/cfs/cdirs/m4958/usr/aritra08/cont_learn/ssl/results/d32/checkpoints"),
        64: Path("/global/cfs/cdirs/m4958/usr/aritra08/token/vqvae_tracks/self_supervised/sm_only_v1/checkpoints"),
    }
    for d, cdir in ssl_paths.items():
        if (EMBED_DIR / f"ssl_legacy_d{d}.npz").exists():
            continue
        with open(cdir / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        enc = SetTransformerEncoder(latent_dim=d, norm_first=True, activation="gelu", use_out_norm=True).to(device)
        enc.load_state_dict(torch.load(cdir / "encoder_ssl_simclr.pt", map_location=device))
        z = extract_set_transformer_embeddings(enc, feat7_eval, mask_eval, scaler, device)
        save_embedding_npz(f"ssl_legacy_d{d}", z, eval_data)

    # 3. MLP 28-D Summary Autoencoder d8, d32, d64
    mlp_runs = {8: "sm_only_v1_run2", 32: "sm_only_v2_run2", 64: "sm_only_v3_run2"}
    for d, run_name in mlp_runs.items():
        if (EMBED_DIR / f"ae_mlp_d{d}.npz").exists():
            continue
        cdir = Path(f"/global/cfs/cdirs/m4958/usr/aritra08/autoencoders/mlp/results/{run_name}/checkpoints")
        with open(cdir / "scaler.pkl", "rb") as f:
            trk_scaler = pickle.load(f)
        with open(cdir / "target_scaler.pkl", "rb") as f:
            tgt_scaler = pickle.load(f)
        sum28 = compute_summaries28_np(feat7_eval, mask_eval, scaler=trk_scaler)
        sum28_s = tgt_scaler.transform(sum28).astype(np.float32)

        enc = MLPEncoder(latent_dim=d).to(device)
        enc.load_state_dict(torch.load(cdir / "encoder_ae_mlp.pt", map_location=device))
        enc.eval()
        with torch.no_grad():
            z = enc(torch.from_numpy(sum28_s).to(device)).cpu().numpy()
        save_embedding_npz(f"ae_mlp_d{d}", z, eval_data)

    # 4. Legacy Post-LN/ReLU Transformer 28-D Summary Bottleneck d8, d32, d64
    tr_runs = {8: "sm_only_v4", 32: "sm_only_v2", 64: "sm_only_v3"}
    for d, run_name in tr_runs.items():
        if (EMBED_DIR / f"ae_transformer_postln_d{d}.npz").exists():
            continue
        cdir = Path(f"/global/cfs/cdirs/m4958/usr/aritra08/autoencoders/transformer/results/{run_name}/checkpoints")
        with open(cdir / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        enc = SetTransformerEncoder(latent_dim=d, norm_first=False, activation="relu", use_out_norm=False).to(device)
        enc.load_state_dict(torch.load(cdir / "encoder_ae_transformer.pt", map_location=device))
        z = extract_set_transformer_embeddings(enc, feat7_eval, mask_eval, scaler, device)
        save_embedding_npz(f"ae_transformer_postln_d{d}", z, eval_data)

    # 5. Randomly Initialized Frozen 8L/8H Pre-LN GELU SetTransformer Baseline d8, d32, d64
    ref_scaler_path = Path("/global/cfs/cdirs/m4958/usr/aritra08/cont_learn/supervised/results/d64/checkpoints/scaler.pkl")
    with open(ref_scaler_path, "rb") as f:
        ref_scaler = pickle.load(f)
    for d in [8, 32, 64]:
        if (EMBED_DIR / f"random_init_d{d}.npz").exists():
            continue
        set_seed(1000 + d)
        enc = SetTransformerEncoder(latent_dim=d, norm_first=True, activation="gelu", use_out_norm=True).to(device)
        z = extract_set_transformer_embeddings(enc, feat7_eval, mask_eval, ref_scaler, device)
        save_embedding_npz(f"random_init_d{d}", z, eval_data)

    # 6. VQ-VAE Pre-Quantization Encoder (D=128 hidden & D=8 code) & Existing Tokenized SSL (D=64)
    del feat7_eval
    from vqvae_tracks import TrackVQVAE
    from pretrain_ssl_contrastive_tokens import TokenSetTransformerEncoder

    vq_dir = Path("/global/cfs/cdirs/m4958/usr/aritra08/token/vqvae_tracks/sm_only_v1")
    with open(vq_dir / "scaler.pkl", "rb") as f:
        vq_scaler = pickle.load(f)
    vq_ckpt = torch.load(vq_dir / "vqvae_best.pt", map_location=device)
    vqvae = TrackVQVAE(**vq_ckpt["cfg"]).to(device)
    vqvae.load_state_dict(vq_ckpt["state_dict"])
    vqvae.eval()

    vq6_eval, mask_vq_eval = raw6_to_vqvae6_np(eval_data["raw6"], mask_eval)
    vq_mean = vq_scaler.mean_.astype(np.float32).reshape(1, 1, 6)
    vq_scale = vq_scaler.scale_.astype(np.float32).reshape(1, 1, 6)

    tok_ckpt = Path("/global/cfs/cdirs/m4958/usr/aritra08/token/ssl_tokens/sm_only_v1/encoder_ssl_tokens.pt")
    tok_enc64 = TokenSetTransformerEncoder(codebook_size=8192, latent_dim=64).to(device)
    tok_enc64.load_state_dict(torch.load(tok_ckpt, map_location=device))
    tok_enc64.eval()

    N = len(eval_data["labels"])
    z_vq_enc128 = []
    z_vq_code8 = []
    z_tok64 = []
    tokens_eval = np.full((N, vq6_eval.shape[1]), -1, dtype=np.int16)

    with torch.no_grad():
        for start in range(0, N, 1024):
            end = min(start + 1024, N)
            m_b = mask_vq_eval[start:end]
            x_b = (vq6_eval[start:end] - vq_mean) / vq_scale
            x_b[~m_b] = 0.0
            x_t = torch.from_numpy(x_b).to(device, non_blocking=True)
            m_t = torch.from_numpy(m_b).to(device, non_blocking=True)

            h = vqvae.encoder(vqvae.enc_in(x_t), src_key_padding_mask=~m_t)
            h_norm = vqvae.enc_out[0](h)
            m_f = m_t.unsqueeze(-1).float()
            denom = m_f.sum(dim=1).clamp(min=1.0)
            pooled128 = (h_norm * m_f).sum(dim=1) / denom
            z_vq_enc128.append(pooled128.cpu().numpy())

            z_code = vqvae.enc_out[1](h_norm)
            pooled8 = (z_code * m_f).sum(dim=1) / denom
            z_vq_code8.append(pooled8.cpu().numpy())

            _, idx, _, _, _ = vqvae.vq(z_code, m_t)
            tokens_eval[start:end] = idx.cpu().numpy().astype(np.int16)

            z_t64 = tok_enc64(idx.clamp(min=0), m_t)
            z_tok64.append(z_t64.cpu().numpy())

    save_embedding_npz("vqvae_encoder_d128", np.concatenate(z_vq_enc128, axis=0), eval_data)
    save_embedding_npz("vqvae_code_d8", np.concatenate(z_vq_code8, axis=0), eval_data)
    save_embedding_npz("ssl_tokens_d64", np.concatenate(z_tok64, axis=0), eval_data)

    tok_cache = Path("/pscratch/sd/d/danieltm/fm_cache/eval_downstream_vq_tokens.npz")
    np.savez(tok_cache, tokens=tokens_eval, mask=mask_vq_eval)
    print(f"[run_extract_existing] Saved eval VQ tokens to {tok_cache}")


def train_ssl_kinematic(latent_dim: int, drop_prob: float = 0.0, epochs: int = 20, batch_size: int = 512, device: torch.device = torch.device("cuda")):
    tag = f"ssl_kinematic_drop_d{latent_dim}" if drop_prob > 0 else f"ssl_kinematic_d{latent_dim}"
    save_dir = CKPT_BASE / tag
    save_dir.mkdir(parents=True, exist_ok=True)
    set_seed(42 + latent_dim)

    pre_data = load_events_fast(PRETRAIN_PARQUET, include_labels=SM_LABELS)
    feat7_pre = raw6_to_feat7_np(pre_data["raw6"], pre_data["mask"])
    scaler = fit_track_scaler(feat7_pre, pre_data["mask"])
    del feat7_pre

    with open(save_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    mean_t = torch.from_numpy(scaler.mean_.astype(np.float32)).to(device)
    scale_t = torch.from_numpy(scaler.scale_.astype(np.float32)).to(device)

    encoder = SetTransformerEncoder(latent_dim=latent_dim, norm_first=True, activation="gelu", use_out_norm=True).to(device)
    proj_head = ProjectionHead(in_dim=latent_dim, hidden_dim=64, out_dim=latent_dim).to(device)
    params = list(encoder.parameters()) + list(proj_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-6)

    N = len(pre_data["labels"])
    steps_per_epoch = N // batch_size
    total_steps = epochs * steps_per_epoch
    warmup_steps = 500
    lr_lambda = lambda step: min(1.0, (step + 1) / warmup_steps) * 0.5 * (1.0 + math.cos(math.pi * max(0.0, step - warmup_steps) / max(1, total_steps - warmup_steps)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    raw6_gpu = torch.from_numpy(pre_data["raw6"]).to(device)
    mask_gpu = torch.from_numpy(pre_data["mask"]).to(device)
    lengths_gpu = torch.from_numpy(pre_data["lengths"]).to(device)
    del pre_data
    history = []

    print(f"[{tag}] Training for {epochs} epochs ({steps_per_epoch} steps/epoch) on {device}...", flush=True)
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        encoder.train()
        proj_head.train()
        perm = torch.randperm(N, device=device)
        epoch_loss = 0.0
        embed_std = 0.0

        for step in range(steps_per_epoch):
            idx = perm[step * batch_size : (step + 1) * batch_size]
            max_t = min(max(int(lengths_gpu[idx].max().item()), 1), 128)
            r6_b = raw6_gpu[idx, :max_t, :]
            m_b = mask_gpu[idx, :max_t]

            x1, m1 = augment_batch_kinematic_torch(r6_b, m_b, mean_t, scale_t, drop_prob=drop_prob)
            x2, m2 = augment_batch_kinematic_torch(r6_b, m_b, mean_t, scale_t, drop_prob=drop_prob)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                z1 = encoder(x1, m1)
                z2 = encoder(x2, m2)
                loss = simclr_nt_xent_loss(proj_head(z1), proj_head(z2), temperature=0.5)

            if step == 0:
                embed_std = float(z1.detach().float().std(dim=0).mean().item())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += float(loss.item())

        avg_loss = epoch_loss / steps_per_epoch
        dt = time.time() - t0
        history.append({"epoch": epoch, "loss": avg_loss, "embed_std": embed_std, "sec": dt})
        print(f"[{tag}] Epoch {epoch:02d}/{epochs}: loss={avg_loss:.4f}, embed_std={embed_std:.4f} ({dt:.1f}s)", flush=True)

    del raw6_gpu, mask_gpu, lengths_gpu
    torch.cuda.empty_cache()

    torch.save(encoder.state_dict(), save_dir / "encoder.pt")
    torch.save(proj_head.state_dict(), save_dir / "proj_head.pt")
    pd.DataFrame(history).to_csv(save_dir / "training_history.csv", index=False)

    eval_data = load_events_fast(EVAL_PARQUET)
    feat7_eval = raw6_to_feat7_np(eval_data["raw6"], eval_data["mask"])
    z_eval = extract_set_transformer_embeddings(encoder, feat7_eval, eval_data["mask"], scaler, device)
    save_embedding_npz(tag, z_eval, eval_data)


def train_ae_transformer(latent_dim: int, epochs: int = 30, batch_size: int = 512, device: torch.device = torch.device("cuda")):
    tag = f"ae_transformer_d{latent_dim}"
    save_dir = CKPT_BASE / tag
    save_dir.mkdir(parents=True, exist_ok=True)
    set_seed(142 + latent_dim)

    pre_data = load_events_fast(PRETRAIN_PARQUET, include_labels=SM_LABELS)
    feat7_pre = raw6_to_feat7_np(pre_data["raw6"], pre_data["mask"])
    del pre_data["raw6"]
    mask_pre = pre_data["mask"]
    trk_scaler = fit_track_scaler(feat7_pre, mask_pre)

    mean = trk_scaler.mean_.astype(np.float32).reshape(1, 1, 7)
    scale = trk_scaler.scale_.astype(np.float32).reshape(1, 1, 7)
    feat7_pre -= mean
    feat7_pre /= scale
    feat7_pre[~mask_pre] = 0.0
    feat7_s = feat7_pre

    sum28 = compute_summaries28_np(feat7_s, mask_pre)
    tgt_scaler = StandardScaler().fit(sum28)
    sum28_s = tgt_scaler.transform(sum28).astype(np.float32)

    with open(save_dir / "scaler.pkl", "wb") as f:
        pickle.dump(trk_scaler, f)
    with open(save_dir / "target_scaler.pkl", "wb") as f:
        pickle.dump(tgt_scaler, f)

    encoder = SetTransformerEncoder(latent_dim=latent_dim, norm_first=True, activation="gelu", use_out_norm=True).to(device)
    decoder = AEDecoder(latent_dim=latent_dim, out_dim=28).to(device)
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    N = len(pre_data["labels"])
    steps_per_epoch = N // batch_size
    history = []

    feat7_gpu = torch.from_numpy(feat7_s).to(device)
    mask_gpu = torch.from_numpy(mask_pre).to(device)
    lengths_gpu = torch.from_numpy(pre_data["lengths"]).to(device)
    sum28_gpu = torch.from_numpy(sum28_s).to(device)
    del feat7_s, mask_pre, sum28_s, pre_data

    print(f"[{tag}] Training Pre-LN GELU Transformer Summary Bottleneck for {epochs} epochs on {device}...", flush=True)
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        encoder.train()
        decoder.train()
        perm = torch.randperm(N, device=device)
        epoch_loss = 0.0

        for step in range(steps_per_epoch):
            idx = perm[step * batch_size : (step + 1) * batch_size]
            max_t = min(max(int(lengths_gpu[idx].max().item()), 1), 128)
            x_b = feat7_gpu[idx, :max_t, :]
            m_b = mask_gpu[idx, :max_t]
            y_b = sum28_gpu[idx]

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                z = encoder(x_b, m_b)
                recon = decoder(z)
                loss = F.mse_loss(recon, y_b)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            epoch_loss += float(loss.item())

        scheduler.step()
        avg_loss = epoch_loss / steps_per_epoch
        dt = time.time() - t0
        history.append({"epoch": epoch, "loss": avg_loss, "sec": dt})
        print(f"[{tag}] Epoch {epoch:02d}/{epochs}: mse_loss={avg_loss:.5f} ({dt:.1f}s)", flush=True)

    del feat7_gpu, mask_gpu, lengths_gpu, sum28_gpu
    torch.cuda.empty_cache()

    torch.save(encoder.state_dict(), save_dir / "encoder.pt")
    torch.save(decoder.state_dict(), save_dir / "decoder.pt")
    pd.DataFrame(history).to_csv(save_dir / "training_history.csv", index=False)

    eval_data = load_events_fast(EVAL_PARQUET)
    feat7_eval = raw6_to_feat7_np(eval_data["raw6"], eval_data["mask"])
    z_eval = extract_set_transformer_embeddings(encoder, feat7_eval, eval_data["mask"], trk_scaler, device)
    save_embedding_npz(tag, z_eval, eval_data)


def train_ssl_tokens(latent_dim: int, epochs: int = 15, batch_size: int = 512, device: torch.device = torch.device("cuda")):
    from vqvae_tracks import TrackVQVAE
    from pretrain_ssl_contrastive_tokens import TokenSetTransformerEncoder

    tag = f"ssl_tokens_d{latent_dim}"
    if (EMBED_DIR / f"{tag}.npz").exists():
        print(f"[{tag}] {EMBED_DIR / f'{tag}.npz'} already exists, skipping.")
        return
    save_dir = CKPT_BASE / tag
    save_dir.mkdir(parents=True, exist_ok=True)
    set_seed(242 + latent_dim)

    vq_dir = Path("/global/cfs/cdirs/m4958/usr/aritra08/token/vqvae_tracks/sm_only_v1")
    with open(vq_dir / "scaler.pkl", "rb") as f:
        vq_scaler = pickle.load(f)
    vq_ckpt = torch.load(vq_dir / "vqvae_best.pt", map_location=device)
    vqvae = TrackVQVAE(**vq_ckpt["cfg"]).to(device)
    vqvae.load_state_dict(vq_ckpt["state_dict"])
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad = False

    mean_t = torch.from_numpy(vq_scaler.mean_.astype(np.float32)).to(device).view(1, 1, 6)
    scale_t = torch.from_numpy(vq_scaler.scale_.astype(np.float32)).to(device).view(1, 1, 6)
    noise_vec = torch.tensor([0.05, 0.05, 0.03, 0.0, 0.0, 0.0], dtype=torch.float32, device=device).view(1, 1, 6)

    pre_data = load_events_fast(PRETRAIN_PARQUET, include_labels=SM_LABELS)
    vq6_pre, mask_pre = raw6_to_vqvae6_np(pre_data["raw6"], pre_data["mask"])
    del pre_data["raw6"]

    encoder = TokenSetTransformerEncoder(codebook_size=8192, latent_dim=latent_dim).to(device)
    proj_head = ProjectionHead(in_dim=latent_dim, hidden_dim=64, out_dim=latent_dim).to(device)
    params = list(encoder.parameters()) + list(proj_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-6)

    N = len(pre_data["labels"])
    steps_per_epoch = N // batch_size
    total_steps = epochs * steps_per_epoch
    warmup_steps = 500
    lr_lambda = lambda step: min(1.0, (step + 1) / warmup_steps) * 0.5 * (1.0 + math.cos(math.pi * max(0.0, step - warmup_steps) / max(1, total_steps - warmup_steps)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def make_tokenized_view(x_b: torch.Tensor, m_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x_b.shape
        drop_prob = 0.10 + 0.10 * torch.rand((B, 1), device=device)
        keep_r = torch.rand((B, T), device=device) >= drop_prob
        keep_r[:, 0] = True
        m_aug = m_b & keep_r
        x_aug = x_b + torch.randn_like(x_b) * noise_vec
        x_scaled = ((x_aug - mean_t) / scale_t).masked_fill(~m_aug.unsqueeze(-1), 0.0)
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                z_e = vqvae.encode(x_scaled, m_aug)
            _, idx, _, _, _ = vqvae.vq(z_e.float(), m_aug)
        return idx.clamp(min=0), m_aug

    vq6_gpu = torch.from_numpy(vq6_pre).to(device)
    mask_gpu = torch.from_numpy(mask_pre).to(device)
    lengths_vq_gpu = mask_gpu.sum(dim=1)
    del vq6_pre, mask_pre, pre_data

    history = []
    print(f"[{tag}] Training TokenSetTransformerEncoder for {epochs} epochs on {device}...", flush=True)
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        encoder.train()
        proj_head.train()
        perm = torch.randperm(N, device=device)
        epoch_loss = 0.0
        embed_std = 0.0

        for step in range(steps_per_epoch):
            idx = perm[step * batch_size : (step + 1) * batch_size]
            max_t = max(int(lengths_vq_gpu[idx].max().item()), 1)
            x_b = vq6_gpu[idx, :max_t, :]
            m_b = mask_gpu[idx, :max_t]

            tok1, m1 = make_tokenized_view(x_b, m_b)
            tok2, m2 = make_tokenized_view(x_b, m_b)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                z1 = encoder(tok1, m1)
                z2 = encoder(tok2, m2)
                loss = simclr_nt_xent_loss(proj_head(z1), proj_head(z2), temperature=0.5)

            if step == 0:
                embed_std = float(z1.detach().float().std(dim=0).mean().item())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += float(loss.item())

        avg_loss = epoch_loss / steps_per_epoch
        dt = time.time() - t0
        history.append({"epoch": epoch, "loss": avg_loss, "embed_std": embed_std, "sec": dt})
        print(f"[{tag}] Epoch {epoch:02d}/{epochs}: loss={avg_loss:.4f}, embed_std={embed_std:.4f} ({dt:.1f}s)", flush=True)

    del vq6_gpu, mask_gpu, lengths_vq_gpu
    torch.cuda.empty_cache()

    torch.save(encoder.state_dict(), save_dir / "encoder.pt")
    torch.save(proj_head.state_dict(), save_dir / "proj_head.pt")
    pd.DataFrame(history).to_csv(save_dir / "training_history.csv", index=False)

    # Extract eval_downstream embeddings (using cached VQ tokens if available)
    eval_data = load_events_fast(EVAL_PARQUET)
    tok_cache = Path("/pscratch/sd/d/danieltm/fm_cache/eval_downstream_vq_tokens.npz")
    encoder.eval()
    z_eval_list = []
    N_eval = len(eval_data["labels"])
    if tok_cache.exists():
        tc = np.load(tok_cache)
        tokens_eval = tc["tokens"]
        mask_eval = tc["mask"]
        if len(tokens_eval) == N_eval:
            with torch.no_grad():
                for start in range(0, N_eval, 2048):
                    end = min(start + 2048, N_eval)
                    t_b = torch.from_numpy(tokens_eval[start:end].astype(np.int64)).clamp(min=0).to(device)
                    m_b = torch.from_numpy(mask_eval[start:end]).to(device)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                        z = encoder(t_b, m_b)
                    z_eval_list.append(z.float().cpu().numpy())
            save_embedding_npz(tag, np.concatenate(z_eval_list, axis=0), eval_data)
            return

    vq6_eval, mask_eval = raw6_to_vqvae6_np(eval_data["raw6"], eval_data["mask"])
    with torch.no_grad():
        for start in range(0, N_eval, 1024):
            end = min(start + 1024, N_eval)
            x_b = torch.from_numpy(vq6_eval[start:end]).to(device, non_blocking=True)
            m_b = torch.from_numpy(mask_eval[start:end]).to(device, non_blocking=True)
            x_scaled = ((x_b - mean_t) / scale_t).masked_fill(~m_b.unsqueeze(-1), 0.0)
            z_e = vqvae.encode(x_scaled, m_b)
            _, idx, _, _, _ = vqvae.vq(z_e, m_b)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                z = encoder(idx.clamp(min=0), m_b)
            z_eval_list.append(z.float().cpu().numpy())
    save_embedding_npz(tag, np.concatenate(z_eval_list, axis=0), eval_data)


def train_supervised_e2e(latent_dim: int = 64, epochs: int = 20, batch_size: int = 512, device: torch.device = torch.device("cuda")):
    tag = f"supervised_e2e_d{latent_dim}"
    save_dir = CKPT_BASE / tag
    save_dir.mkdir(parents=True, exist_ok=True)
    set_seed(342 + latent_dim)

    pre_data = load_events_fast(PRETRAIN_PARQUET, include_labels=SM_LABELS)
    feat7_pre = raw6_to_feat7_np(pre_data["raw6"], pre_data["mask"])
    del pre_data["raw6"]
    mask_pre = pre_data["mask"]
    labels_pre = pre_data["labels"]
    scaler = fit_track_scaler(feat7_pre, mask_pre)

    mean = scaler.mean_.astype(np.float32).reshape(1, 1, 7)
    scale = scaler.scale_.astype(np.float32).reshape(1, 1, 7)
    feat7_pre -= mean
    feat7_pre /= scale
    feat7_pre[~mask_pre] = 0.0
    feat7_s = feat7_pre

    encoder = SetTransformerEncoder(latent_dim=latent_dim, norm_first=True, activation="gelu", use_out_norm=True).to(device)
    head = nn.Linear(latent_dim, 3).to(device)
    params = list(encoder.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    N = len(labels_pre)
    steps_per_epoch = N // batch_size
    history = []

    feat7_gpu = torch.from_numpy(feat7_s).to(device)
    mask_gpu = torch.from_numpy(mask_pre).to(device)
    lengths_gpu = torch.from_numpy(pre_data["lengths"]).to(device)
    labels_gpu = torch.from_numpy(labels_pre).to(device)
    del feat7_s, mask_pre, labels_pre, pre_data

    print(f"[{tag}] Training End-to-End Supervised 3-Class SM Classifier for {epochs} epochs on {device}...", flush=True)
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        encoder.train()
        head.train()
        perm = torch.randperm(N, device=device)
        epoch_loss = 0.0
        correct = 0
        total = 0

        for step in range(steps_per_epoch):
            idx = perm[step * batch_size : (step + 1) * batch_size]
            max_t = min(max(int(lengths_gpu[idx].max().item()), 1), 128)
            x_b = feat7_gpu[idx, :max_t, :]
            m_b = mask_gpu[idx, :max_t]
            y_b = labels_gpu[idx]

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                z = encoder(x_b, m_b)
                logits = head(z)
                loss = F.cross_entropy(logits, y_b)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            epoch_loss += float(loss.item())
            correct += int((logits.argmax(dim=-1) == y_b).sum().item())
            total += len(y_b)

        scheduler.step()
        avg_loss = epoch_loss / steps_per_epoch
        acc = correct / total
        dt = time.time() - t0
        history.append({"epoch": epoch, "loss": avg_loss, "train_acc": acc, "sec": dt})
        print(f"[{tag}] Epoch {epoch:02d}/{epochs}: loss={avg_loss:.4f}, train_acc={acc:.4f} ({dt:.1f}s)", flush=True)

    del feat7_gpu, mask_gpu, lengths_gpu, labels_gpu
    torch.cuda.empty_cache()

    torch.save(encoder.state_dict(), save_dir / "encoder.pt")
    torch.save(head.state_dict(), save_dir / "head.pt")
    with open(save_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    pd.DataFrame(history).to_csv(save_dir / "training_history.csv", index=False)

    eval_data = load_events_fast(EVAL_PARQUET)
    feat7_eval = raw6_to_feat7_np(eval_data["raw6"], eval_data["mask"])
    z_eval = extract_set_transformer_embeddings(encoder, feat7_eval, eval_data["mask"], scaler, device)
    head.eval()
    with torch.no_grad():
        logits_eval = head(torch.from_numpy(z_eval).to(device)).cpu().numpy()
    save_embedding_npz(tag, z_eval, eval_data, extra={"sm3_logits": logits_eval.astype(np.float32)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=[
        "cache_only",
        "extract_existing",
        "train_ssl_kinematic",
        "train_ssl_kinematic_drop",
        "train_ae_transformer",
        "train_ssl_tokens",
        "train_supervised_e2e",
    ])
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    if args.task == "cache_only":
        load_events_fast(PRETRAIN_PARQUET, include_labels=SM_LABELS)
        load_events_fast(EVAL_PARQUET, include_labels=None)
    elif args.task == "extract_existing":
        run_extract_existing(device)
    elif args.task == "train_ssl_kinematic":
        train_ssl_kinematic(args.dim, drop_prob=0.0, epochs=args.epochs or 20, device=device)
    elif args.task == "train_ssl_kinematic_drop":
        train_ssl_kinematic(args.dim, drop_prob=0.10, epochs=args.epochs or 20, device=device)
    elif args.task == "train_ae_transformer":
        train_ae_transformer(args.dim, epochs=args.epochs or 30, device=device)
    elif args.task == "train_ssl_tokens":
        train_ssl_tokens(args.dim, epochs=args.epochs or 15, device=device)
    elif args.task == "train_supervised_e2e":
        train_supervised_e2e(args.dim, epochs=args.epochs or 20, device=device)


if __name__ == "__main__":
    main()
