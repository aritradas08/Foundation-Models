#!/usr/bin/env python3
"""
Construct strictly disjoint Pretraining SM (900k) and Downstream Evaluation datasets
using vectorized PyArrow operations, enforcing non-empty track filtering (n_tracks > 0)
and explicit event_id disjointness checks.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import glob
import json
import os
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

RAW_BASE_DIR = Path("/global/cfs/cdirs/m4958/data/ColliderML/simulation/hard_scatter")
ARITRA_DATA_DIR = Path("/global/cfs/cdirs/m4958/usr/aritra08/Anomaly_Detection/data")
EXT_HP_V2_DIR = Path("/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/data/higgs_portal_v2_ext/parquet/reco/tracks")
DEFAULT_OUT_DIR = Path("/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/FoundationModels/data")

TRACK_COLS = ["event_id", "d0", "z0", "phi", "theta", "qop"]
LIST_F32 = pa.list_(pa.field("element", pa.float32()))


def read_and_derive_parquet(fpath: str) -> pa.Table:
    """Vectorized read of a single raw ColliderML reco/tracks parquet file + derived (pt, eta, p)."""
    tbl = pq.read_table(fpath, columns=TRACK_COLS)
    lengths = pc.list_value_length(tbl["d0"]).to_numpy()
    keep = lengths > 0
    if not np.all(keep):
        tbl = tbl.filter(pa.array(keep))

    offsets = tbl["theta"].chunk(0).offsets if tbl["theta"].num_chunks == 1 else tbl.combine_chunks()["theta"].chunk(0).offsets
    if tbl["theta"].num_chunks != 1:
        tbl = tbl.combine_chunks()
        offsets = tbl["theta"].chunk(0).offsets

    theta = tbl["theta"].chunk(0).values.to_numpy().astype(np.float32)
    qop = tbl["qop"].chunk(0).values.to_numpy().astype(np.float32)

    p = np.abs(1.0 / (qop + np.float32(1e-12))).astype(np.float32)
    pt = (p * np.abs(np.sin(theta))).astype(np.float32)
    eta = (-np.log(np.tan(np.clip(theta * np.float32(0.5), np.float32(1e-9), np.float32(np.pi / 2 - 1e-9))))).astype(np.float32)

    cols = {
        "event_id": pc.cast(tbl["event_id"], pa.int64()),
        "d0": pa.ListArray.from_arrays(offsets, tbl["d0"].chunk(0).values, type=LIST_F32),
        "z0": pa.ListArray.from_arrays(offsets, tbl["z0"].chunk(0).values, type=LIST_F32),
        "phi": pa.ListArray.from_arrays(offsets, tbl["phi"].chunk(0).values, type=LIST_F32),
        "theta": pa.ListArray.from_arrays(offsets, tbl["theta"].chunk(0).values, type=LIST_F32),
        "qop": pa.ListArray.from_arrays(offsets, tbl["qop"].chunk(0).values, type=LIST_F32),
        "pt": pa.ListArray.from_arrays(offsets, pa.array(pt, type=pa.float32()), type=LIST_F32),
        "eta": pa.ListArray.from_arrays(offsets, pa.array(eta, type=pa.float32()), type=LIST_F32),
        "p": pa.ListArray.from_arrays(offsets, pa.array(p, type=pa.float32()), type=LIST_F32),
    }
    return pa.table(cols)


def load_process_from_files(files: list[str], target_events: int | None, label: int, process_name: str, exclude_ids: set[int] | None = None) -> pa.Table:
    with ThreadPoolExecutor(max_workers=16) as ex:
        tables = list(ex.map(read_and_derive_parquet, files))
    combined = pa.concat_tables(tables)
    if exclude_ids:
        ev_ids = combined["event_id"].to_numpy()
        mask = np.array([int(eid) not in exclude_ids for eid in ev_ids], dtype=bool)
        if not np.all(mask):
            combined = combined.filter(pa.array(mask))
    if target_events is not None and combined.num_rows > target_events:
        combined = combined.slice(0, target_events)
    n = combined.num_rows
    combined = combined.append_column("label", pa.array(np.full(n, label, dtype=np.int64), type=pa.int64()))
    combined = combined.append_column("process", pa.array([process_name] * n, type=pa.string()))
    return combined


def load_existing_processed(path: Path, target_events: int | None, label: int, process_name: str) -> pa.Table:
    cols = ["event_id", "d0", "z0", "phi", "theta", "qop", "pt", "eta", "p"]
    tbl = pq.read_table(path, columns=cols)
    lengths = pc.list_value_length(tbl["d0"]).to_numpy()
    keep = lengths > 0
    if not np.all(keep):
        tbl = tbl.filter(pa.array(keep))
    if target_events is not None and tbl.num_rows > target_events:
        tbl = tbl.slice(0, target_events)
    n = tbl.num_rows
    tbl = tbl.append_column("label", pa.array(np.full(n, label, dtype=np.int64), type=pa.int64()))
    tbl = tbl.append_column("process", pa.array([process_name] * n, type=pa.string()))
    return tbl


def build_splits(out_dir: Path, n_pretrain_per_sm: int = 300_000, n_eval_per_sm: int = 250_000, n_eval_bsm: int = 100_000):
    out_dir.mkdir(parents=True, exist_ok=True)
    sm_channels = [("ttbar", 0), ("ggf", 1), ("dihiggs", 2)]

    pretrain_tables = []
    eval_tables = []
    manifest = {"pretrain": {}, "eval": {}, "disjoint_verified": True}

    for proc, lbl in sm_channels:
        print(f"[SM: {proc}] Loading pretraining split ({n_pretrain_per_sm} events)...")
        existing_pre = ARITRA_DATA_DIR / f"{proc}_pu0_300k.parquet"
        if existing_pre.exists():
            t_pre = load_existing_processed(existing_pre, n_pretrain_per_sm, lbl, proc)
        else:
            raw_files = sorted(glob.glob(str(RAW_BASE_DIR / proc / "v1" / "parquet" / "reco" / "tracks" / "*.parquet")))
            t_pre = load_process_from_files(raw_files[:310], n_pretrain_per_sm, lbl, proc)

        pre_ids = set(t_pre["event_id"].to_pylist())
        pretrain_tables.append(t_pre)
        manifest["pretrain"][proc] = {
            "label": lbl,
            "n_events": t_pre.num_rows,
            "min_event_id": int(pc.min(t_pre["event_id"]).as_py()),
            "max_event_id": int(pc.max(t_pre["event_id"]).as_py()),
        }

        print(f"[SM: {proc}] Loading disjoint evaluation split ({n_eval_per_sm} events)...")
        raw_files = sorted(glob.glob(str(RAW_BASE_DIR / proc / "v1" / "parquet" / "reco" / "tracks" / "*.parquet")))
        eval_candidate_files = raw_files[301 : 301 + (n_eval_per_sm // 1000) + 15]
        t_eval = load_process_from_files(eval_candidate_files, n_eval_per_sm, lbl, proc, exclude_ids=pre_ids)
        eval_ids = set(t_eval["event_id"].to_pylist())

        overlap = len(pre_ids & eval_ids)
        assert overlap == 0, f"CRITICAL: {proc} pretrain and eval overlap by {overlap} events!"
        eval_tables.append(t_eval)
        manifest["eval"][proc] = {
            "label": lbl,
            "n_events": t_eval.num_rows,
            "overlap_with_pretrain": overlap,
            "min_event_id": int(pc.min(t_eval["event_id"]).as_py()),
            "max_event_id": int(pc.max(t_eval["event_id"]).as_py()),
        }
        print(f"  -> {proc}: pretrain={t_pre.num_rows}, eval={t_eval.num_rows}, overlap={overlap}")

    # BSM 1: higgs_portal v2 (fixed displaced LLP + 3T seeding)
    hp_v2_files = sorted(glob.glob(str(RAW_BASE_DIR / "higgs_portal" / "v2" / "parquet" / "reco" / "tracks" / "*.parquet")))
    if EXT_HP_V2_DIR.exists():
        ext_files = sorted(glob.glob(str(EXT_HP_V2_DIR / "*.parquet")))
        hp_v2_files.extend(ext_files)
    print(f"[BSM: higgs_portal_v2] Found {len(hp_v2_files)} parquet files...")
    t_hp_v2 = load_process_from_files(hp_v2_files, n_eval_bsm, 3, "higgs_portal")
    eval_tables.append(t_hp_v2)
    manifest["eval"]["higgs_portal_v2"] = {
        "label": 3,
        "process": "higgs_portal",
        "version": "v2",
        "n_files": len(hp_v2_files),
        "n_events": t_hp_v2.num_rows,
    }
    print(f"  -> higgs_portal (v2): {t_hp_v2.num_rows} non-empty events loaded")

    # BSM 2: hidden_valley v1
    print(f"[BSM: hidden_valley] Loading {n_eval_bsm} events...")
    existing_hv = ARITRA_DATA_DIR / "hidden_valley_pu0_300k.parquet"
    if existing_hv.exists():
        t_hv = load_existing_processed(existing_hv, n_eval_bsm, 4, "hidden_valley")
    else:
        hv_files = sorted(glob.glob(str(RAW_BASE_DIR / "hidden_valley" / "v1" / "parquet" / "reco" / "tracks" / "*.parquet")))
        t_hv = load_process_from_files(hv_files[:110], n_eval_bsm, 4, "hidden_valley")
    eval_tables.append(t_hv)
    manifest["eval"]["hidden_valley"] = {
        "label": 4,
        "process": "hidden_valley",
        "version": "v1",
        "n_events": t_hv.num_rows,
    }
    print(f"  -> hidden_valley (v1): {t_hv.num_rows} non-empty events loaded")

    # BSM 3 (Comparison): higgs_portal v1 (legacy prompt bug)
    print(f"[BSM legacy comparison: higgs_portal_v1] Loading {n_eval_bsm} events...")
    existing_hp_v1 = ARITRA_DATA_DIR / "higgs_portal_pu0_300k.parquet"
    if existing_hp_v1.exists():
        t_hp_v1 = load_existing_processed(existing_hp_v1, n_eval_bsm, 5, "higgs_portal_v1")
    else:
        hp_v1_files = sorted(glob.glob(str(RAW_BASE_DIR / "higgs_portal" / "v1" / "parquet" / "reco" / "tracks" / "*.parquet")))
        t_hp_v1 = load_process_from_files(hp_v1_files[:110], n_eval_bsm, 5, "higgs_portal_v1")
    eval_tables.append(t_hp_v1)
    manifest["eval"]["higgs_portal_v1"] = {
        "label": 5,
        "process": "higgs_portal_v1",
        "version": "v1",
        "n_events": t_hp_v1.num_rows,
    }
    print(f"  -> higgs_portal_v1 (legacy): {t_hp_v1.num_rows} non-empty events loaded")

    pretrain_all = pa.concat_tables(pretrain_tables)
    eval_all = pa.concat_tables(eval_tables)

    pretrain_path = out_dir / "pretrain_sm_900k.parquet"
    eval_path = out_dir / "eval_downstream.parquet"
    manifest_path = out_dir / "split_manifest.json"

    print(f"Writing {pretrain_path} ({pretrain_all.num_rows} rows)...")
    pq.write_table(pretrain_all, pretrain_path, compression="snappy")

    print(f"Writing {eval_path} ({eval_all.num_rows} rows)...")
    pq.write_table(eval_all, eval_path, compression="snappy")

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved split manifest to {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-pretrain-per-sm", type=int, default=300_000)
    parser.add_argument("--n-eval-per-sm", type=int, default=250_000)
    parser.add_argument("--n-eval-bsm", type=int, default=100_000)
    args = parser.parse_args()
    build_splits(args.out_dir, args.n_pretrain_per_sm, args.n_eval_per_sm, args.n_eval_bsm)
