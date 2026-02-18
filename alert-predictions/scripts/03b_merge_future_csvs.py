import argparse
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, help="Folder containing *_future.csv files")
    ap.add_argument("--output", required=True, help="Output merged CSV path")
    ap.add_argument("--pattern", default="*_future.csv", help="Glob pattern")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    files = sorted(in_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"❌ No files found: {in_dir}/{args.pattern}")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df["source_file"] = f.name   # keep provenance
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True, sort=False)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)

    print("✅ Merged files:", len(files))
    print("✅ Rows:", len(merged))
    print("✅ Saved:", out_path)

if __name__ == "__main__":
    main()
"""
python scripts/merge_future_csvs.py \
  --input-dir data/linux/labeled \
  --output data/linux/labeled/combined_linux_future.csv

  Then train:

python src/linux/03_train_baseline.py \
  --input data/linux/labeled/combined_linux_future.csv \
  --model-dir models/linux/baseline_combined \
  --target-precision 0.60 \
  --min-lines 3


"""
