import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", required=True, help="windows.csv from step 1")
    ap.add_argument("--window-anom", required=True, help="windows_anom.csv with column window_is_anomaly (0/1)")
    ap.add_argument("--output", required=True, help="Output windows_future.csv")
    ap.add_argument("--horizon-min", type=int, default=10)
    ap.add_argument("--window-sec", type=int, default=60)
    args = ap.parse_args()

    w = pd.read_csv(args.windows)
    a = pd.read_csv(args.window_anom)

    w["bucket"] = pd.to_datetime(w["bucket"], errors="coerce")
    a["bucket"] = pd.to_datetime(a["bucket"], errors="coerce")

    df = w.merge(a[["bucket", "window_is_anomaly"]], on="bucket", how="left")
    df["window_is_anomaly"] = df["window_is_anomaly"].fillna(0).astype(int)
    df = df.sort_values("bucket").reset_index(drop=True)

    horizon_windows = int((args.horizon_min * 60) / args.window_sec)

    arr = df["window_is_anomaly"].values
    y_future = np.zeros_like(arr)

    for i in range(len(arr)):
        j1 = i + 1
        j2 = min(len(arr), i + 1 + horizon_windows)
        y_future[i] = arr[j1:j2].max() if j1 < j2 else 0

    df["y_future"] = y_future.astype(int)
    df["horizon_min"] = args.horizon_min

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print("✅ Saved:", out_path)
    print("Positives:", int(df["y_future"].sum()), "/", len(df))

if __name__ == "__main__":
    main()
"""
python src/linux/02_label_future.py \
  --input data/linux/processed/linux_windowz.csv \
  --output data/linux/labeled/linux_windowz_future.csv \
  --horizon-min 10 \
  --window-sec 60
"""
