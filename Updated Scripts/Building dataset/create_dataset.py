import os
import glob
import argparse
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

base_dir = "/global/cfs/cdirs/m4958/data/ColliderML/simulation/hard_scatter"
out_dir = "/global/cfs/cdirs/m4958/usr/aritra08/Anomaly_Detection/data"
track_cols = ["event_id", "d0", "z0", "phi", "theta", "qop"]

os.makedirs(out_dir, exist_ok=True)


def compute_derived(theta_arr, qop_arr):
    # p, pt and eta from theta and q/p
    theta = np.array(theta_arr, dtype=np.float32)
    qop = np.array(qop_arr, dtype=np.float32)

    p = np.abs(1.0 / (qop + 1e-12))
    pt = p * np.abs(np.sin(theta))
    eta = -np.log(np.tan(np.clip(theta / 2.0, 1e-9, np.pi / 2 - 1e-9)))

    return pt, eta, p


def get_parquet_files(process, version="v1"):
    pattern = os.path.join(base_dir, process, version, "parquet", "reco", "tracks", "*.parquet")
    files = sorted(glob.glob(pattern))
    return files


def create_dataset(process, n_events=100_000, version="v1"):
    files = get_parquet_files(process, version)

    rows = []
    total = 0

    for fpath in tqdm(files, desc="reading files"):
        if total >= n_events:
            break

        df = pq.read_table(fpath, columns=track_cols).to_pandas()

        for _, row in df.iterrows():
            if total >= n_events:
                break

            theta_arr = np.array(row["theta"], dtype=np.float32)
            qop_arr = np.array(row["qop"], dtype=np.float32)

            pt, eta, p = compute_derived(theta_arr, qop_arr)

            rows.append({
                "event_id": int(row["event_id"]),
                "d0": list(np.array(row["d0"], dtype=np.float32)),
                "z0": list(np.array(row["z0"], dtype=np.float32)),
                "phi": list(np.array(row["phi"], dtype=np.float32)),
                "theta": list(theta_arr),
                "qop": list(qop_arr),
                "pt": list(pt),
                "eta": list(eta),
                "p": list(p),
            })
            total += 1

    out_df = pd.DataFrame(rows)

    out_path = os.path.join(out_dir, f"{process}_pu0_{n_events // 1000}k.parquet")
    out_df.to_parquet(out_path, index=False)
    print(f"saved {out_df.shape[0]} events to {out_path}")

    return out_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--process", type=str, default="ttbar")
    parser.add_argument("--n_events", type=int, default=100_000)
    parser.add_argument("--version", type=str, default="v1")
    args = parser.parse_args()

    create_dataset(args.process, args.n_events, args.version)
