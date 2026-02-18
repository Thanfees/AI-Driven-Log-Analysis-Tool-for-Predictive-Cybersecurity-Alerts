import argparse
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", required=True, help="Linux_labeled.csv (has timestamp + is_anomaly)")
    ap.add_argument("--output", required=True, help="Output windows_anom.csv with bucket + window_is_anomaly")
    ap.add_argument("--window", default="60s")
    ap.add_argument("--year", type=int, default=2026)
    args = ap.parse_args()

    df = pd.read_csv(args.labeled)

    dt = pd.to_datetime(df["timestamp"], errors="coerce")
    if dt.isna().mean() > 0.2:
        dt2 = pd.to_datetime(df["timestamp"].astype(str) + f" {args.year}",
                             format="%b %d %H:%M:%S %Y", errors="coerce")
        dt = dt.fillna(dt2)

    df["datetime"] = dt
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    df["bucket"] = df["datetime"].dt.floor(args.window)
    df["is_anomaly"] = df["is_anomaly"].fillna(0).astype(int)

    out = df.groupby("bucket")["is_anomaly"].max().reset_index(name="window_is_anomaly")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print("✅ Saved:", out_path, " rows=", len(out))

if __name__ == "__main__":
    main()
"""
python src/linux/02a_make_window_anom_from_labeled.py \
  --labeled data/linux/labeled/Linux_labeled.csv \
  --output data/linux/labeled/windows_anom.csv \
  --window 60s
  """
