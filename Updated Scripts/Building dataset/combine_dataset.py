import os
import pandas as pd

dataset_dir = "/global/cfs/cdirs/m4958/usr/aritra08/Anomaly_Detection/data"

processes = {
    "ttbar": 0,
    "ggf": 1,
    "dihiggs": 2,
    "higgs_portal": 3,
    "hidden_valley": 4,
}

n_events = 300_000
out_path = os.path.join(dataset_dir, "combined_pu0_1500k.parquet")

dfs = []

for process, label in processes.items():
    path = os.path.join(dataset_dir, f"{process}_pu0_{n_events // 1000}k.parquet")

    df = pd.read_parquet(path)
    df["label"] = label
    df["process"] = process
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)
combined.to_parquet(out_path, index=False)

print(f"saved {combined.shape[0]} events to {out_path}")
print(combined.groupby(["label", "process"]).size().to_string())
