#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path
import pandas as pd


def run(cmd: list[str]) -> None:
    print("\n🟦 RUN:", " ".join(cmd))
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(p.stdout)
    if p.returncode != 0:
        raise SystemExit(f"❌ Command failed: {' '.join(cmd)}")


def base_name_before_extension(p: Path) -> str:
    """
    Use original name before final extension.
    Example:
      server1.csv -> server1
      a.b.c.csv   -> a.b.c
    """
    return p.name.rsplit(".", 1)[0]


def concat_csvs(files: list[Path], out_path: Path) -> None:
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df["source_file"] = f.name
        dfs.append(df)
    merged = pd.concat(dfs, ignore_index=True, sort=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"✅ Combined dataset saved: {out_path} (rows={len(merged)})")


def main():
    # Recommend venv activation
    if os.environ.get("VIRTUAL_ENV") is None:
        print("⚠️ Not running inside a virtualenv.")
        print("   Activate first: source \"/home/hackgodx/Projects/RP/venv/bin/activate\"")
    ap = argparse.ArgumentParser(description="Run full Linux pipeline. Accepts .csv or auto-converts .log to CSV.")

    # Inputs/Outputs
    ap.add_argument("--raw-dir", default="raw_logs/linux", help="Folder containing input logs (.csv or .log)")
    ap.add_argument("--converted-dir", default="data/linux/raw_csv", help="Where to store converted CSVs from .log inputs")
    ap.add_argument("--processed-dir", default="data/linux/processed", help="Output processed folder")
    ap.add_argument("--labeled-dir", default="data/linux/labeled", help="Output labeled folder")
    ap.add_argument("--outputs-dir", default="outputs/linux", help="Predictions output folder")
    ap.add_argument("--models-dir", default="models/linux", help="Models output folder")

    # Pipeline params
    ap.add_argument("--window", default="5min", help="Window size for 01_windowize.py (e.g., 30s, 60s, 5min)")
    ap.add_argument("--horizon-min", type=int, default=30, help="Predict anomaly within next H minutes")
    ap.add_argument("--use-trends", action="store_true", help="Add rolling trend features before future labeling")
    ap.add_argument("--recursive", action="store_true", help="Recurse when converting .log files in --raw-dir")

    # Training params (reduce false positives)
    ap.add_argument("--target-precision", type=float, default=0.60,
                    help="Threshold selection target precision (higher => fewer false positives).")
    ap.add_argument("--min-lines", type=int, default=3,
                    help="Filter windows with lines < min-lines during training & inference.")
    ap.add_argument("--k-confirm", type=int, default=2,
                    help="Inference: require K consecutive positive windows to confirm alert (reduces FP).")

    # Optional sequence model
    ap.add_argument("--train-seq", action="store_true", help="Also train GRU sequence model")
    ap.add_argument("--seq-len", type=int, default=6, help="Sequence length (e.g., 6 windows for 5min -> 30min history)")

    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    labeled_dir = Path(args.labeled_dir)
    outputs_dir = Path(args.outputs_dir)
    models_dir = Path(args.models_dir)
    converted_dir = Path(args.converted_dir)

    processed_dir.mkdir(parents=True, exist_ok=True)
    labeled_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # If .log files exist in raw_dir, convert them to CSVs first
    data_source_dir = raw_dir
    log_files = list(raw_dir.rglob("*.log") if args.recursive else raw_dir.glob("*.log"))
    if log_files:
        print(f"🔁 Converting {len(log_files)} .log file(s) from {raw_dir} -> {converted_dir}")
        run([
            sys.executable, "src/linux/pipeline/01_convert_log_to_csv.py",
            "--log-path", str(raw_dir),
            "--output-dir", str(converted_dir),
            *( ["--recursive"] if args.recursive else [] )
        ])
        data_source_dir = converted_dir

    # Collect CSV logs from data_source_dir
    csv_files = sorted(data_source_dir.glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"❌ No CSV files found in: {data_source_dir.resolve()}")

    print(f"✅ Found {len(csv_files)} CSV files in {data_source_dir}")

    future_files: list[Path] = []
    windowz_files: list[Path] = []

    for f in csv_files:
        stem = base_name_before_extension(f)
        print(f"\n====================\n📄 Processing: {f.name}  (base='{stem}')\n====================")

        # 1) Windowize
        windowz_path = processed_dir / f"{stem}_windowz.csv"
        run([
            sys.executable, "src/linux/pipeline/02_windowize.py",
            "--input", str(f),
            "--output", str(windowz_path),
            "--window", args.window
        ])
        windowz_files.append(windowz_path)

        # 2) Label windows (rules) - YOUR filename
        labeled_windowz_path = labeled_dir / f"{stem}_windowz_labeled.csv"
        run([
            sys.executable, "src/linux/pipeline/03_label_windows.py",
            "--input", str(windowz_path),
            "--output", str(labeled_windowz_path),
            "--text-col", "text_with_proc"
        ])

        labeled_for_future = labeled_windowz_path

        # 3) Optional: add trend features
        if args.use_trends:
            trends_path = labeled_dir / f"{stem}_windowz_labeled_trends.csv"
            run([
                sys.executable, "src/linux/pipeline/04_add_trends.py",
                "--input", str(labeled_windowz_path),
                "--output", str(trends_path)
            ])
            labeled_for_future = trends_path

        # 4) Future labels (your renamed script)
        future_path = labeled_dir / f"{stem}_future.csv"
        run([
            sys.executable, "src/linux/pipeline/05_make_future_labels.py",
            "--input", str(labeled_for_future),
            "--output", str(future_path),
            "--horizon-min", str(args.horizon_min)
        ])
        future_files.append(future_path)

    # 5) Merge all future files into ONE combined training dataset
    combined_future = labeled_dir / "combined_linux_future.csv"
    concat_csvs(future_files, combined_future)

    # 6) Train one baseline model on combined dataset
    baseline_model_dir = models_dir / f"baseline_combined_w{args.window}_h{args.horizon_min}m"
    run([
        sys.executable, "src/linux/pipeline/06_train_baseline.py",
        "--input", str(combined_future),
        "--model-dir", str(baseline_model_dir),
        "--target-precision", str(args.target_precision),
        "--min-lines", str(args.min_lines)
    ])

    # 7) Inference per file using the single combined model (low FP settings)
    # Use the same feature table used to generate future labels (includes trends if enabled)
    for f in csv_files:
        stem = base_name_before_extension(f)
        # Determine which feature file to use for inference
        labeled_path = labeled_dir / f"{stem}_windowz_labeled.csv"
        trends_path = labeled_dir / f"{stem}_windowz_labeled_trends.csv"
        inference_source = trends_path if (args.use_trends and trends_path.exists()) else labeled_path
        pred_path = outputs_dir / f"{stem}_predictions.csv"
        run([
            sys.executable, "src/linux/pipeline/07_infer_baseline.py",
            "--input", str(inference_source),
            "--model-dir", str(baseline_model_dir),
            "--output", str(pred_path),
            "--min-lines", str(args.min_lines),
            "--k-confirm", str(args.k_confirm)
        ])

    # 8) Optional: train GRU sequence model (on combined dataset)
    if args.train_seq:
        seq_dir = models_dir / f"seq_combined_w{args.window}_h{args.horizon_min}m_len{args.seq_len}"
        run([
            sys.executable, "src/linux/pipeline/08_train_seq_gru.py",
            "--input", str(combined_future),
            "--model-dir", str(seq_dir),
            "--seq-len", str(args.seq_len)
        ])

    print("\n✅ ALL DONE 🎉")
    print(f"📌 Combined training file: {combined_future}")
    print(f"📌 Baseline model dir:     {baseline_model_dir}")
    print(f"📌 Predictions folder:     {outputs_dir}")


if __name__ == "__main__":
    main()
"""
This is your “balanced low-FP” setup:

source "/home/hackgodx/Projects/RP/venv/bin/activate"
python scripts/00_run_linux_pipeline.py \
  --raw-dir data/linux/raw_csv \
  --window 5min \
  --horizon-min 30 \
  --use-trends \
  --target-precision 0.60 \
  --min-lines 3 \
  --k-confirm 2


Want even fewer false positives? 👇

source "/home/hackgodx/Projects/RP/venv/bin/activate"
python scripts/00_run_linux_pipeline.py \
  --raw-dir data/linux/raw_csv \
  --window 5min \
  --horizon-min 30 \
  --use-trends \
  --target-precision 0.70 \
  --min-lines 5 \
  --k-confirm 3


"""
