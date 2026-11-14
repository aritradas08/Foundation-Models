#!/usr/bin/env python3
"""
Shared data loading, feature engineering, kinematically consistent GPU augmentations,
and standardized 8-layer/8-head Pre-LN GELU SetTransformer architectures.
"""
import os
import pickle
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

RAW_COLS = ["d0", "z0", "theta", "p", "eta", "phi", "pt"]
TRACK_FEAT_NAMES = ["d0", "z0", "p", "pt", "tx", "ty", "tz"]
N_TRACK_FEATS = 7
MAX_TRACKS_DEFAULT = 256
MAX_TRACKS_VQVAE = 128

SM_LABELS = [0, 1, 2]
PROCESS_NAMES = {
    0: "ttbar",
    1: "ggf",
    2: "dihiggs",
    3: "higgs_portal_v2",
    4: "hidden_valley",
    5: "higgs_portal_v1",
}


def symlog_np(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def symlog_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def load_events_fast(
    parquet_path: str | Path,
    max_tracks: int = MAX_TRACKS_DEFAULT,
    include_labels: list[int] | None = None,
    cache_dir: str | Path | None = "/pscratch/sd/d/danieltm/fm_cache",
) -> dict[str, np.ndarray]:
    """
    Vectorized PyArrow loader that sorts tracks within each event by descending pT,
    truncates/pads to max_tracks, and caches the resulting arrays on /pscratch for instant reload.
    Returns dict with:
      - raw6: (N, max_tracks, 6) float32 [symlog(d0), symlog(z0), p, pt, phi, theta]
      - mask: (N, max_tracks) bool (True = valid track)
      - lengths: (N,) int32 (number of valid tracks up to max_tracks)
      - labels: (N,) int64
      - event_ids: (N,) int64
    """
    parquet_path = Path(parquet_path)
    mtime = int(parquet_path.stat().st_mtime)
    lbl_tag = "all" if include_labels is None else "_".join(map(str, sorted(include_labels)))
    cache_file = None
    if cache_dir is not None:
        cdir = Path(cache_dir)
        cdir.mkdir(parents=True, exist_ok=True)
        cache_file = cdir / f"{parquet_path.stem}_mt{max_tracks}_lbl{lbl_tag}_{mtime}.npz"
        if cache_file.exists():
            print(f"[load_events_fast] Loading cached arrays from {cache_file}...")
            data = np.load(cache_file)
            return {k: data[k] for k in data.files}

    print(f"[load_events_fast] Reading {parquet_path} (max_tracks={max_tracks}, labels={include_labels})...")
    cols_to_read = ["event_id", "label", "d0", "z0", "p", "pt", "phi", "theta"]
    tbl = pq.read_table(parquet_path, columns=cols_to_read)

    if include_labels is not None:
        lbl_np = tbl["label"].to_numpy()
        keep_lbl = np.isin(lbl_np, include_labels)
        if not np.all(keep_lbl):
            tbl = tbl.filter(pa.array(keep_lbl))

    lengths_full = pc.list_value_length(tbl["d0"]).to_numpy()
    keep_nonempty = lengths_full > 0
    if not np.all(keep_nonempty):
        tbl = tbl.filter(pa.array(keep_nonempty))
        lengths_full = lengths_full[keep_nonempty]

    if tbl.num_rows == 0:
        raise ValueError(f"No events matched filter include_labels={include_labels} in {parquet_path}")
    if tbl["d0"].num_chunks > 1:
        tbl = tbl.combine_chunks()

    n_events = tbl.num_rows
    event_ids = tbl["event_id"].to_numpy().astype(np.int64)
    labels = tbl["label"].to_numpy().astype(np.int64)

    d0_flat = symlog_np(tbl["d0"].chunk(0).values.to_numpy().astype(np.float32))
    z0_flat = symlog_np(tbl["z0"].chunk(0).values.to_numpy().astype(np.float32))
    p_flat = tbl["p"].chunk(0).values.to_numpy().astype(np.float32)
    pt_flat = tbl["pt"].chunk(0).values.to_numpy().astype(np.float32)
    phi_flat = tbl["phi"].chunk(0).values.to_numpy().astype(np.float32)
    theta_flat = tbl["theta"].chunk(0).values.to_numpy().astype(np.float32)
    del tbl

    # Sort tracks per event by descending pT
    ev_idx = np.repeat(np.arange(n_events, dtype=np.int64), lengths_full)
    order = np.lexsort((-pt_flat, ev_idx))
    ev_sorted = ev_idx[order]

    starts = np.zeros(n_events + 1, dtype=np.int64)
    np.cumsum(lengths_full, out=starts[1:])
    rank_in_event = np.arange(len(ev_sorted), dtype=np.int64) - starts[ev_sorted]

    keep_trk = rank_in_event < max_tracks
    sel = order[keep_trk]
    ev_sel = ev_sorted[keep_trk]
    rank_sel = rank_in_event[keep_trk]

    raw6 = np.zeros((n_events, max_tracks, 6), dtype=np.float32)
    raw6[ev_sel, rank_sel, 0] = d0_flat[sel]
    raw6[ev_sel, rank_sel, 1] = z0_flat[sel]
    raw6[ev_sel, rank_sel, 2] = p_flat[sel]
    raw6[ev_sel, rank_sel, 3] = pt_flat[sel]
    raw6[ev_sel, rank_sel, 4] = phi_flat[sel]
    raw6[ev_sel, rank_sel, 5] = theta_flat[sel]

    lengths = np.minimum(lengths_full, max_tracks).astype(np.int32)
    mask = (np.arange(max_tracks)[None, :] < lengths[:, None])

    out = {
        "raw6": raw6,
        "mask": mask,
        "lengths": lengths,
        "labels": labels,
        "event_ids": event_ids,
    }
    if cache_file is not None:
        tmp_file = cache_file.with_name(f".{cache_file.stem}.tmp.{os.getpid()}.npz")
        np.savez(tmp_file, **out)
        tmp_file.replace(cache_file)
        print(f"[load_events_fast] Saved cache to {cache_file}")
    return out


def raw6_to_feat7_np(raw6: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Convert (N, T, 6) [symlog(d0), symlog(z0), p, pt, phi, theta] into
    (N, T, 7) [symlog(d0), symlog(z0), p, pt, tx, ty, tz].
    """
    N, T, _ = raw6.shape
    feat7 = np.zeros((N, T, 7), dtype=np.float32)
    feat7[:, :, 0:4] = raw6[:, :, 0:4]
    phi = raw6[:, :, 4]
    theta = raw6[:, :, 5]
    sin_t = np.sin(theta)
    feat7[:, :, 4] = np.cos(phi) * sin_t
    feat7[:, :, 5] = np.sin(phi) * sin_t
    feat7[:, :, 6] = np.cos(theta)
    feat7[~mask] = 0.0
    return feat7


def raw6_to_vqvae6_np(raw6: np.ndarray, mask: np.ndarray, max_tracks_vq: int = MAX_TRACKS_VQVAE) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert (N, T, 6) [symlog(d0), symlog(z0), p, pt, phi, theta] (sorted by descending pT)
    into the 6 VQ-VAE track features [d0_symlog, z0_symlog, log_pt, eta, cos_dphi, sin_dphi]
    truncated to max_tracks_vq=128.
    """
    raw6 = raw6[:, :max_tracks_vq, :]
    mask_vq = mask[:, :max_tracks_vq]
    N, T, _ = raw6.shape
    vq6 = np.zeros((N, T, 6), dtype=np.float32)
    vq6[:, :, 0] = raw6[:, :, 0]
    vq6[:, :, 1] = raw6[:, :, 1]
    vq6[:, :, 2] = np.log(np.clip(raw6[:, :, 3], 1e-3, None))
    half_theta = np.clip(raw6[:, :, 5].astype(np.float64) * 0.5, 1e-6, np.pi / 2 - 1e-6)
    vq6[:, :, 3] = (-np.log(np.tan(half_theta))).astype(np.float32)
    phi_lead = raw6[:, 0:1, 4]
    dphi = (raw6[:, :, 4] - phi_lead + np.pi) % (2.0 * np.pi) - np.pi
    vq6[:, :, 4] = np.cos(dphi)
    vq6[:, :, 5] = np.sin(dphi)
    vq6[~mask_vq] = 0.0
    return vq6, mask_vq


def fit_track_scaler(feat7: np.ndarray, mask: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(feat7[mask])
    return scaler


def compute_summaries28_np(
    feat7: np.ndarray,
    mask: np.ndarray,
    scaler: StandardScaler | None = None,
    chunk_size: int = 32768,
) -> np.ndarray:
    """
    Chunked vectorized computation of 28-D per-event summary statistics [mean, std, max, min]
    over valid tracks (where mask is True), avoiding multi-GB intermediate array allocations.
    """
    N, T, F = feat7.shape
    out = np.zeros((N, 4 * F), dtype=np.float32)
    if scaler is not None:
        s_mean = scaler.mean_.astype(np.float32).reshape(1, 1, F)
        s_scale = scaler.scale_.astype(np.float32).reshape(1, 1, F)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        m_c = mask[start:end]
        max_t = max(int(m_c.sum(axis=1).max()), 1)
        m_c = m_c[:, :max_t]
        x_c = feat7[start:end, :max_t, :]
        if scaler is not None:
            x_c = (x_c - s_mean) / s_scale
        m_f = m_c[:, :, None].astype(np.float32)
        counts = np.maximum(m_f.sum(axis=1), 1.0)

        x_masked = x_c * m_f
        mean = x_masked.sum(axis=1) / counts

        diff = (x_c - mean[:, None, :]) * m_f
        var = (diff ** 2).sum(axis=1) / counts
        std = np.sqrt(np.maximum(var, 0.0))

        x_for_max = np.where(m_c[:, :, None], x_c, -np.inf)
        x_for_min = np.where(m_c[:, :, None], x_c, np.inf)
        max_v = x_for_max.max(axis=1)
        min_v = x_for_min.min(axis=1)

        out[start:end] = np.concatenate([mean, std, max_v, min_v], axis=1)

    return out


def augment_batch_kinematic_torch(
    raw6: torch.Tensor,
    mask: torch.Tensor,
    scaler_mean: torch.Tensor,
    scaler_scale: torch.Tensor,
    d0_noise_rel: float = 0.10,
    z0_noise_rel: float = 0.10,
    p_noise_rel: float = 0.05,
    theta_noise_rad: float = 0.02,
    drop_prob: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Kinematically consistent batch track augmentation on GPU:
      1. Global azimuthal rotation phi' = phi + dphi, dphi ~ U(-pi, pi) per event
         plus small per-track polar & azimuthal resolution smearing
      2. Additive Gaussian smearing on symlog(d0), symlog(z0) scaled by pretrain feature std
      3. Multiplicative log-normal momentum smearing p' = p * exp(eps_p), eps_p ~ N(0, p_noise_rel^2),
         followed by exact recomputation pT' = p' * |sin(theta')| (preserving positivity & pT = p sin(theta))
      4. Recompute unit direction (tx', ty', tz') from (phi', theta') so tz is not an exact fingerprint
      5. Optional random track drop (with probability drop_prob), always keeping rank-0 (leading-pT) track
      6. Standardize with pretrain (scaler_mean, scaler_scale)
    """
    B, T, _ = raw6.shape
    device = raw6.device

    d0 = raw6[:, :, 0]
    z0 = raw6[:, :, 1]
    p = raw6[:, :, 2]
    phi = raw6[:, :, 4]
    theta = raw6[:, :, 5]

    # 1. Event-level azimuthal rotation + per-track angular resolution smearing
    dphi = (torch.rand((B, 1), device=device, dtype=raw6.dtype) * (2.0 * torch.pi)) - torch.pi
    phi_aug = phi + dphi + torch.randn_like(phi) * theta_noise_rad
    theta_aug = torch.clamp(theta + torch.randn_like(theta) * theta_noise_rad, 1e-3, torch.pi - 1e-3)

    # 2. Impact parameter smearing in symlog space
    d0_aug = d0 + torch.randn_like(d0) * (d0_noise_rel * scaler_scale[0])
    z0_aug = z0 + torch.randn_like(z0) * (z0_noise_rel * scaler_scale[1])

    # 3. Positive log-normal momentum smearing + exact pT = p * |sin(theta')|
    eps_p = torch.randn_like(p) * p_noise_rel
    p_aug = p * torch.exp(eps_p)
    sin_theta = torch.sin(theta_aug)
    pt_aug = p_aug * torch.abs(sin_theta)

    # 4. Unit direction vectors
    tx_aug = torch.cos(phi_aug) * sin_theta
    ty_aug = torch.sin(phi_aug) * sin_theta
    tz_aug = torch.cos(theta_aug)

    feat7_aug = torch.stack([d0_aug, z0_aug, p_aug, pt_aug, tx_aug, ty_aug, tz_aug], dim=-1)
    feat7_scaled = (feat7_aug - scaler_mean.view(1, 1, 7)) / scaler_scale.view(1, 1, 7)

    # 5. Optional track drop (never drop index 0 so every event has >= 1 valid track)
    if drop_prob > 0.0:
        keep_rand = torch.rand((B, T), device=device) >= drop_prob
        keep_rand[:, 0] = True
        mask_aug = mask & keep_rand
    else:
        mask_aug = mask

    feat7_scaled = feat7_scaled.masked_fill(~mask_aug.unsqueeze(-1), 0.0)
    return feat7_scaled, mask_aug


class SetTransformerEncoder(nn.Module):
    """
    Standardized 8-layer, 8-head Pre-LN GELU Set Transformer Encoder with [CLS] pooling
    and output LayerNorm before projection to latent_dim.
    """
    def __init__(
        self,
        in_dim: int = N_TRACK_FEATS,
        model_dim: int = 64,
        n_heads: int = 8,
        n_layers: int = 8,
        ffn_dim: int = 256,
        latent_dim: int = 64,
        dropout: float = 0.025,
        norm_first: bool = True,
        activation: str = "gelu",
        use_out_norm: bool = True,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, model_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, model_dim) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=norm_first,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers, enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(model_dim) if use_out_norm else nn.Identity()
        self.out_proj = nn.Linear(model_dim, latent_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        h = self.input_proj(x)
        cls = self.cls_token.expand(B, 1, -1)
        h = torch.cat([cls, h], dim=1)

        cls_valid = torch.ones((B, 1), dtype=torch.bool, device=mask.device)
        full_valid = torch.cat([cls_valid, mask], dim=1)
        pad_mask = ~full_valid

        h = self.transformer(h, src_key_padding_mask=pad_mask)
        cls_out = self.out_norm(h[:, 0, :])
        return self.out_proj(cls_out)


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int = 64, hidden_dim: int = 64, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(z), dim=-1)


class AEDecoder(nn.Module):
    def __init__(self, latent_dim: int = 64, hidden_1: int = 128, hidden_2: int = 256, out_dim: int = 28):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_1),
            nn.ReLU(),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Linear(hidden_2, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int = 28, hidden_1: int = 256, hidden_2: int = 128, latent_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_1),
            nn.ReLU(),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Linear(hidden_2, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
