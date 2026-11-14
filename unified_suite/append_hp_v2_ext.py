#!/usr/bin/env python3
"""
Loads the newly generated higgs_portal v2 (pu0) runs (runs 10..105) from
/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/data/higgs_portal_v2_ext/parquet/reco/tracks/,
selects 90,002 valid events (n_tracks > 0) so that total higgs_portal v2 (label=3) = 100,000 events,
saves them to hp_v2_ext_90k.parquet, and appends their embeddings to all existing .npz files in results/embeddings/.
"""
import glob
import os
import pickle
import sys
import time
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
import torch.nn as nn

SUITE_DIR = Path(__file__).resolve().parent
REPO_ROOT = SUITE_DIR.parent
sys.path.insert(0, str(SUITE_DIR))
sys.path.insert(0, str(REPO_ROOT / "Tokenization-scripts" / "scripts"))

from common import (
    MAX_TRACKS_DEFAULT,
    MLPEncoder,
    SetTransformerEncoder,
    compute_summaries28_np,
    load_events_fast,
    raw6_to_feat7_np,
    raw6_to_vqvae6_np,
)
from train_and_extract import (
    CKPT_BASE,
    DATA_DIR,
    EMBED_DIR,
    RESULTS_BASE,
    extract_set_transformer_embeddings,
    set_seed,
)

CACHE_DIR = Path("/pscratch/sd/d/danieltm/fm_cache")
EXT_PQ_DIR = DATA_DIR / "higgs_portal_v2_ext" / "parquet" / "reco" / "tracks"
EXT_PARQUET_OUT = DATA_DIR / "hp_v2_ext_90k.parquet"


def build_ext_parquet(target_count: int = 90_002) -> Path:
    if EXT_PARQUET_OUT.exists():
        print(f"[append_hp_v2_ext] {EXT_PARQUET_OUT} already exists.")
        return EXT_PARQUET_OUT

    files = sorted(glob.glob(str(EXT_PQ_DIR / "hard_scatter.higgs_portal.v2.reco.tracks.events*.parquet")))
    print(f"[append_hp_v2_ext] Found {len(files)} parquet files in {EXT_PQ_DIR}")
    tables = []
    total_valid = 0

    for f in files:
        t = pq.read_table(f)
        lengths = pc.list_value_length(t["d0"]).to_numpy()
        keep = lengths > 0
        if not np.all(keep):
            t = t.filter(pa.array(keep))
        if t.num_rows == 0:
            continue
        if total_valid + t.num_rows > target_count:
            t = t.slice(0, target_count - total_valid)

        if t["d0"].num_chunks > 1:
            t = t.combine_chunks()

        offsets = t["d0"].chunk(0).offsets
        qop_flat = t["qop"].chunk(0).values.to_numpy().astype(np.float32)
        theta_flat = t["theta"].chunk(0).values.to_numpy().astype(np.float32)

        p_flat = (1.0 / np.maximum(np.abs(qop_flat), 1e-9)).astype(np.float32)
        pt_flat = (p_flat * np.sin(theta_flat)).astype(np.float32)
        th_c = np.clip(theta_flat, 1e-6, np.pi - 1e-6)
        eta_flat = (-np.log(np.maximum(np.tan(0.5 * th_c), 1e-12))).astype(np.float32)

        n_rows = t.num_rows
        t = t.set_column(0, "event_id", pa.array(t["event_id"].to_numpy().astype(np.int64), type=pa.int64()))
        t = t.append_column("pt", pa.ListArray.from_arrays(offsets, pa.array(pt_flat, type=pa.float32())))
        t = t.append_column("eta", pa.ListArray.from_arrays(offsets, pa.array(eta_flat, type=pa.float32())))
        t = t.append_column("p", pa.ListArray.from_arrays(offsets, pa.array(p_flat, type=pa.float32())))
        t = t.append_column("label", pa.array(np.full(n_rows, 3, dtype=np.int64), type=pa.int64()))
        t = t.append_column("process", pa.array(["higgs_portal_v2"] * n_rows, type=pa.string()))

        tables.append(t)
        total_valid += n_rows
        if total_valid >= target_count:
            break

    full_tbl = pa.concat_tables(tables)
    pq.write_table(full_tbl, EXT_PARQUET_OUT)
    print(f"[append_hp_v2_ext] Wrote {full_tbl.num_rows} valid events to {EXT_PARQUET_OUT}")
    return EXT_PARQUET_OUT


def compute_model_embedding_on_ext(model_name: str, ext_data: dict, feat7_ext: np.ndarray, vq_tokens_ext: np.ndarray, vq_mask_ext: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray | None]:
    mask_ext = ext_data["mask"]
    if model_name.startswith("supcon_d"):
        d = int(model_name.split("_d")[-1])
        cdir = Path(f"/global/cfs/cdirs/m4958/usr/aritra08/cont_learn/supervised/results/d{d}/checkpoints")
        with open(cdir / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        enc = SetTransformerEncoder(latent_dim=d, norm_first=True, activation="gelu", use_out_norm=True).to(device)
        enc.load_state_dict(torch.load(cdir / "encoder_supervised_simclr.pt", map_location=device))
        return extract_set_transformer_embeddings(enc, feat7_ext, mask_ext, scaler, device), None

    if model_name.startswith("ssl_legacy_d"):
        d = int(model_name.split("_d")[-1])
        ssl_paths = {
            8: Path("/global/cfs/cdirs/m4958/usr/aritra08/cont_learn/ssl/results/d8/checkpoints"),
            32: Path("/global/cfs/cdirs/m4958/usr/aritra08/cont_learn/ssl/results/d32/checkpoints"),
            64: Path("/global/cfs/cdirs/m4958/usr/aritra08/token/vqvae_tracks/self_supervised/sm_only_v1/checkpoints"),
        }
        cdir = ssl_paths[d]
        with open(cdir / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        enc = SetTransformerEncoder(latent_dim=d, norm_first=True, activation="gelu", use_out_norm=True).to(device)
        enc.load_state_dict(torch.load(cdir / "encoder_ssl_simclr.pt", map_location=device))
        return extract_set_transformer_embeddings(enc, feat7_ext, mask_ext, scaler, device), None

    if model_name.startswith("ae_mlp_d"):
        d = int(model_name.split("_d")[-1])
        mlp_runs = {8: "sm_only_v1_run2", 32: "sm_only_v2_run2", 64: "sm_only_v3_run2"}
        cdir = Path(f"/global/cfs/cdirs/m4958/usr/aritra08/autoencoders/mlp/results/{mlp_runs[d]}/checkpoints")
        with open(cdir / "scaler.pkl", "rb") as f:
            trk_scaler = pickle.load(f)
        with open(cdir / "target_scaler.pkl", "rb") as f:
            tgt_scaler = pickle.load(f)
        sum28 = compute_summaries28_np(feat7_ext, mask_ext, scaler=trk_scaler)
        sum28_s = tgt_scaler.transform(sum28).astype(np.float32)
        enc = MLPEncoder(latent_dim=d).to(device)
        enc.load_state_dict(torch.load(cdir / "encoder_ae_mlp.pt", map_location=device))
        enc.eval()
        with torch.no_grad():
            z = enc(torch.from_numpy(sum28_s).to(device)).cpu().numpy()
        return z, None

    if model_name.startswith("ae_transformer_postln_d"):
        d = int(model_name.split("_d")[-1])
        tr_runs = {8: "sm_only_v4", 32: "sm_only_v2", 64: "sm_only_v3"}
        cdir = Path(f"/global/cfs/cdirs/m4958/usr/aritra08/autoencoders/transformer/results/{tr_runs[d]}/checkpoints")
        with open(cdir / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        enc = SetTransformerEncoder(latent_dim=d, norm_first=False, activation="relu", use_out_norm=False).to(device)
        enc.load_state_dict(torch.load(cdir / "encoder_ae_transformer.pt", map_location=device))
        return extract_set_transformer_embeddings(enc, feat7_ext, mask_ext, scaler, device), None

    if model_name.startswith("random_init_d"):
        d = int(model_name.split("_d")[-1])
        ref_scaler_path = Path("/global/cfs/cdirs/m4958/usr/aritra08/cont_learn/supervised/results/d64/checkpoints/scaler.pkl")
        with open(ref_scaler_path, "rb") as f:
            ref_scaler = pickle.load(f)
        set_seed(1000 + d)
        enc = SetTransformerEncoder(latent_dim=d, norm_first=True, activation="gelu", use_out_norm=True).to(device)
        return extract_set_transformer_embeddings(enc, feat7_ext, mask_ext, ref_scaler, device), None

    if model_name in ("vqvae_encoder_d128", "vqvae_code_d8"):
        from vqvae_tracks import TrackVQVAE
        vq_dir = Path("/global/cfs/cdirs/m4958/usr/aritra08/token/vqvae_tracks/sm_only_v1")
        with open(vq_dir / "scaler.pkl", "rb") as f:
            vq_scaler = pickle.load(f)
        vq_ckpt = torch.load(vq_dir / "vqvae_best.pt", map_location=device)
        vqvae = TrackVQVAE(**vq_ckpt["cfg"]).to(device)
        vqvae.load_state_dict(vq_ckpt["state_dict"])
        vqvae.eval()
        vq6_ext, m_vq = raw6_to_vqvae6_np(ext_data["raw6"], mask_ext)
        vq_mean = vq_scaler.mean_.astype(np.float32).reshape(1, 1, 6)
        vq_scale = vq_scaler.scale_.astype(np.float32).reshape(1, 1, 6)
        z_out = []
        with torch.no_grad():
            for start in range(0, len(mask_ext), 1024):
                end = min(start + 1024, len(mask_ext))
                m_b = m_vq[start:end]
                x_b = (vq6_ext[start:end] - vq_mean) / vq_scale
                x_b[~m_b] = 0.0
                x_t = torch.from_numpy(x_b).to(device)
                m_t = torch.from_numpy(m_b).to(device)
                h = vqvae.encoder(vqvae.enc_in(x_t), src_key_padding_mask=~m_t)
                h_norm = vqvae.enc_out[0](h)
                z_e = vqvae.enc_out[1](h_norm)
                m_f = m_t.unsqueeze(-1).float()
                denom = m_f.sum(dim=1).clamp(min=1.0)
                if model_name == "vqvae_encoder_d128":
                    z_out.append(((h_norm * m_f).sum(dim=1) / denom).cpu().numpy())
                else:
                    z_q, _, _, _, _ = vqvae.vq(z_e, m_t)
                    z_out.append(((z_q * m_f).sum(dim=1) / denom).cpu().numpy())
        return np.concatenate(z_out, axis=0), None

    if model_name == "ssl_tokens_d64":
        from pretrain_ssl_contrastive_tokens import TokenSetTransformerEncoder
        tok_ckpt = Path("/global/cfs/cdirs/m4958/usr/aritra08/token/ssl_tokens/sm_only_v1/encoder_ssl_tokens.pt")
        tok_enc = TokenSetTransformerEncoder(codebook_size=8192, latent_dim=64).to(device)
        tok_enc.load_state_dict(torch.load(tok_ckpt, map_location=device))
        tok_enc.eval()
        z_out = []
        with torch.no_grad():
            for start in range(0, len(vq_tokens_ext), 2048):
                end = min(start + 2048, len(vq_tokens_ext))
                t_b = torch.from_numpy(vq_tokens_ext[start:end].astype(np.int64)).clamp(min=0).to(device)
                m_b = torch.from_numpy(vq_mask_ext[start:end]).to(device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    z_out.append(tok_enc(t_b, m_b).float().cpu().numpy())
        return np.concatenate(z_out, axis=0), None

    if model_name.startswith("ssl_tokens_d"):
        from pretrain_ssl_contrastive_tokens import TokenSetTransformerEncoder
        d = int(model_name.split("_d")[-1])
        cdir = CKPT_BASE / model_name
        tok_enc = TokenSetTransformerEncoder(codebook_size=8192, latent_dim=d).to(device)
        tok_enc.load_state_dict(torch.load(cdir / "encoder.pt", map_location=device))
        tok_enc.eval()
        z_out = []
        with torch.no_grad():
            for start in range(0, len(vq_tokens_ext), 2048):
                end = min(start + 2048, len(vq_tokens_ext))
                t_b = torch.from_numpy(vq_tokens_ext[start:end].astype(np.int64)).clamp(min=0).to(device)
                m_b = torch.from_numpy(vq_mask_ext[start:end]).to(device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    z_out.append(tok_enc(t_b, m_b).float().cpu().numpy())
        return np.concatenate(z_out, axis=0), None

    cdir = CKPT_BASE / model_name
    d = int(model_name.split("_d")[-1])
    with open(cdir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    enc = SetTransformerEncoder(latent_dim=d, norm_first=True, activation="gelu", use_out_norm=True).to(device)
    enc.load_state_dict(torch.load(cdir / "encoder.pt", map_location=device))
    z = extract_set_transformer_embeddings(enc, feat7_ext, mask_ext, scaler, device)
    logits = None
    if model_name.startswith("supervised_e2e_d"):
        head = nn.Linear(d, 3).to(device)
        head.load_state_dict(torch.load(cdir / "head.pt", map_location=device))
        head.eval()
        with torch.no_grad():
            logits = head(torch.from_numpy(z).to(device)).cpu().numpy()
    return z, logits


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pq_path = build_ext_parquet(target_count=90_002)
    ext_data = load_events_fast(pq_path)

    tok_ext_cache = CACHE_DIR / "hp_v2_ext_vq_tokens.npz"
    if tok_ext_cache.exists():
        tc = np.load(tok_ext_cache)
        vq_tokens_ext = tc["tokens"]
        vq_mask_ext = tc["mask"]
    else:
        from vqvae_tracks import TrackVQVAE
        vq_dir = Path("/global/cfs/cdirs/m4958/usr/aritra08/token/vqvae_tracks/sm_only_v1")
        with open(vq_dir / "scaler.pkl", "rb") as f:
            vq_scaler = pickle.load(f)
        vq_ckpt = torch.load(vq_dir / "vqvae_best.pt", map_location=device)
        vqvae = TrackVQVAE(**vq_ckpt["cfg"]).to(device)
        vqvae.load_state_dict(vq_ckpt["state_dict"])
        vqvae.eval()
        vq6_ext, vq_mask_ext = raw6_to_vqvae6_np(ext_data["raw6"], ext_data["mask"])
        vq_mean = vq_scaler.mean_.astype(np.float32).reshape(1, 1, 6)
        vq_scale = vq_scaler.scale_.astype(np.float32).reshape(1, 1, 6)
        vq_tokens_ext = np.full((len(ext_data["labels"]), vq6_ext.shape[1]), -1, dtype=np.int16)
        with torch.no_grad():
            for start in range(0, len(ext_data["labels"]), 1024):
                end = min(start + 1024, len(ext_data["labels"]))
                m_b = vq_mask_ext[start:end]
                x_b = (vq6_ext[start:end] - vq_mean) / vq_scale
                x_b[~m_b] = 0.0
                x_t = torch.from_numpy(x_b).to(device)
                m_t = torch.from_numpy(m_b).to(device)
                z_e = vqvae.encode(x_t, m_t)
                _, idx, _, _, _ = vqvae.vq(z_e, m_t)
                vq_tokens_ext[start:end] = idx.cpu().numpy().astype(np.int16)
        np.savez(tok_ext_cache, tokens=vq_tokens_ext, mask=vq_mask_ext)

    feat7_ext = raw6_to_feat7_np(ext_data["raw6"], ext_data["mask"])

    for npz_path in sorted(EMBED_DIR.glob("*.npz")):
        model_name = npz_path.stem
        d_old = np.load(npz_path)
        n_old = len(d_old["labels"])
        if n_old >= 1_000_000:
            print(f"[append_hp_v2_ext] {model_name}.npz already has N={n_old}, skipping.")
            continue
        t0 = time.time()
        z_ext, logits_ext = compute_model_embedding_on_ext(
            model_name, ext_data, feat7_ext, vq_tokens_ext, vq_mask_ext, device
        )
        save_kwargs = {
            "z": np.concatenate([d_old["z"], z_ext.astype(np.float32)], axis=0),
            "labels": np.concatenate([d_old["labels"], ext_data["labels"]], axis=0),
            "n_tracks": np.concatenate([d_old["n_tracks"], ext_data["lengths"]], axis=0),
            "event_ids": np.concatenate([d_old["event_ids"], ext_data["event_ids"]], axis=0),
        }
        if "sm3_logits" in d_old and logits_ext is not None:
            save_kwargs["sm3_logits"] = np.concatenate([d_old["sm3_logits"], logits_ext.astype(np.float32)], axis=0)
        tmp_path = npz_path.with_suffix(".tmp.npz")
        np.savez(tmp_path, **save_kwargs)
        os.replace(tmp_path, npz_path)
        print(f"[append_hp_v2_ext] Updated {model_name}.npz: {n_old} -> {len(save_kwargs['labels'])} ({time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
