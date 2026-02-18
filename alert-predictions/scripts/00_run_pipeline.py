#!/usr/bin/env python3
"""
00_run_pipeline.py - Unified multi-OS pipeline runner.

Runs the full log-forecast pipeline for Linux, Mac, or Windows logs.
Uses OS-specific parsers/labelers for steps 01-03, then shared pipeline
steps 04-08.

Usage:
    python scripts/00_run_pipeline.py --os linux  --raw-dir raw_logs/linux   --window 60s --horizon-min 15 --use-trends
    python scripts/00_run_pipeline.py --os mac    --raw-dir raw_logs/mac     --window 60s --horizon-min 15 --use-trends
    python scripts/00_run_pipeline.py --os windows --raw-dir raw_logs/windows --window 60s --horizon-min 15 --use-trends
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
import pandas as pd


# OS-specific pipeline script paths
OS_SCRIPTS = {
    "linux": {
        "convert": "src/linux/pipeline/01_convert_log_to_csv.py",
        "windowize": "src/linux/pipeline/02_windowize.py",
        "label": "src/linux/pipeline/03_label_windows.py",
        "trends": "src/linux/pipeline/04_add_trends.py",
        "future": "src/linux/pipeline/05_make_future_labels.py",
        "train": "src/linux/pipeline/06_train_baseline.py",
        "infer": "src/linux/pipeline/07_infer_baseline.py",
        "seq": "src/linux/pipeline/08_train_seq_gru.py",
    },
    "mac": {
        "convert": "src/mac/pipeline/01_convert_log_to_csv.py",
        "windowize": "src/mac/pipeline/02_windowize.py",
        "label": "src/mac/pipeline/03_label_windows.py",
        # Steps 04-08 are shared (OS-agnostic)
        "trends": "src/linux/pipeline/04_add_trends.py",
        "future": "src/linux/pipeline/05_make_future_labels.py",
        "train": "src/linux/pipeline/06_train_baseline.py",
        "infer": "src/linux/pipeline/07_infer_baseline.py",
        "seq": "src/linux/pipeline/08_train_seq_gru.py",
    },
    "windows": {
        "convert": "src/windows/pipeline/01_convert_log_to_csv.py",
        "windowize": "src/windows/pipeline/02_windowize.py",
        "label": "src/windows/pipeline/03_label_windows.py",
        # Steps 04-08 are shared (OS-agnostic)
        "trends": "src/linux/pipeline/04_add_trends.py",
        "future": "src/linux/pipeline/05_make_future_labels.py",
        "train": "src/linux/pipeline/06_train_baseline.py",
        "infer": "src/linux/pipeline/07_infer_baseline.py",
        "seq": "src/linux/pipeline/08_train_seq_gru.py",
    },
}

# Map OS to expected raw log file patterns
OS_LOG_FILES = {
    "linux": ["raw_logs/linux/linux.log", "raw_logs/linux/synth_80k.log", "raw_logs/linux/synth_80k_loanom.log", "raw_logs/linux/synthetic_60k.log"],
    "mac": ["raw_logs/mac/Mac.log"],
    "windows": ["raw_logs/windows/windows20k.log"],
}


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
    ap = argparse.ArgumentParser(
        description="Run full pipeline for Linux, Mac, or Windows logs."
    )

    # OS selection
    ap.add_argument("--os", required=True, choices=["linux", "mac", "windows"],
                    help="Target operating system: linux, mac, or windows")

    # Inputs/Outputs (auto-set based on --os, but can be overridden)
    ap.add_argument("--raw-dir", default=None,
                    help="Folder containing input logs (.csv or .log). Default: raw_logs/{os} (logs) or data/{os}/raw_csv (CSVs)")
    ap.add_argument("--converted-dir", default=None,
                    help="Where to store converted CSVs. Default: data/{os}/raw_csv")
    ap.add_argument("--processed-dir", default=None,
                    help="Output processed folder. Default: data/{os}/processed")
    ap.add_argument("--labeled-dir", default=None,
                    help="Output labeled folder. Default: data/{os}/labeled")
    ap.add_argument("--outputs-dir", default=None,
                    help="Predictions output folder. Default: outputs/{os}")
    ap.add_argument("--models-dir", default=None,
                    help="Models output folder. Default: models/{os}")

    # Pipeline params
    ap.add_argument("--window", default="5min",
                    help="Window size (e.g., 30s, 60s, 5min)")
    ap.add_argument("--horizon-min", type=int, default=30,
                    help="Predict anomaly within next H minutes")
    ap.add_argument("--use-trends", action="store_true",
                    help="Add rolling trend features before future labeling")
    ap.add_argument("--recursive", action="store_true",
                    help="Recurse when converting .log files")

    # Training params
    ap.add_argument("--target-precision", type=float, default=0.60,
                    help="Threshold selection target precision")
    ap.add_argument("--min-lines", type=int, default=3,
                    help="Filter windows with lines < min-lines")
    ap.add_argument("--k-confirm", type=int, default=2,
                    help="Require K consecutive positive windows to confirm alert")

    # Optional sequence model
    ap.add_argument("--train-seq", action="store_true",
                    help="Also train GRU sequence model")
    ap.add_argument("--seq-len", type=int, default=6,
                    help="Sequence length for GRU model")

    args = ap.parse_args()

    target_os = args.os
    scripts = OS_SCRIPTS[target_os]

    # Set default directories based on OS
    raw_dir = Path(args.raw_dir or f"raw_logs/{target_os}")
    converted_dir = Path(args.converted_dir or f"data/{target_os}/raw_csv")
    processed_dir = Path(args.processed_dir or f"data/{target_os}/processed")
    labeled_dir = Path(args.labeled_dir or f"data/{target_os}/labeled")
    outputs_dir = Path(args.outputs_dir or f"outputs/{target_os}")
    models_dir = Path(args.models_dir or f"models/{target_os}")

    processed_dir.mkdir(parents=True, exist_ok=True)
    labeled_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🖥️  Pipeline target: {target_os.upper()}")
    print(f"📂 Raw input:  {raw_dir}")
    print(f"📂 Output:     {outputs_dir}")
    print(f"📂 Models:     {models_dir}")

    # If .log files exist in raw_dir, convert them to CSVs first
    data_source_dir = raw_dir
    log_files = list(raw_dir.rglob("*.log") if args.recursive else raw_dir.glob("*.log"))
    if log_files:
        print(f"\n🔁 Converting {len(log_files)} .log file(s) from {raw_dir} -> {converted_dir}")
        run([
            sys.executable, scripts["convert"],
            "--log-path", str(raw_dir),
            "--output-dir", str(converted_dir),
            *(["--recursive"] if args.recursive else [])
        ])
        data_source_dir = converted_dir

    # Collect CSV logs from data_source_dir
    csv_files = sorted(data_source_dir.glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"❌ No CSV files found in: {data_source_dir.resolve()}")

    print(f"\n✅ Found {len(csv_files)} CSV files in {data_source_dir}")

    future_files: list[Path] = []
    windowz_files: list[Path] = []

    for f in csv_files:
        stem = base_name_before_extension(f)
        print(f"\n====================\n📄 Processing: {f.name}  (base='{stem}')\n====================")

        # 1) Windowize
        windowz_path = processed_dir / f"{stem}_windowz.csv"
        run([
            sys.executable, scripts["windowize"],
            "--input", str(f),
            "--output", str(windowz_path),
            "--window", args.window
        ])
        windowz_files.append(windowz_path)

        # 2) Label windows (OS-specific rules)
        labeled_windowz_path = labeled_dir / f"{stem}_windowz_labeled.csv"
        run([
            sys.executable, scripts["label"],
            "--input", str(windowz_path),
            "--output", str(labeled_windowz_path),
            "--text-col", "text_with_proc"
        ])

        labeled_for_future = labeled_windowz_path

        # 3) Optional: add trend features (shared step)
        if args.use_trends:
            trends_path = labeled_dir / f"{stem}_windowz_labeled_trends.csv"
            run([
                sys.executable, scripts["trends"],
                "--input", str(labeled_windowz_path),
                "--output", str(trends_path)
            ])
            labeled_for_future = trends_path

        # 4) Future labels (shared step)
        future_path = labeled_dir / f"{stem}_future.csv"
        run([
            sys.executable, scripts["future"],
            "--input", str(labeled_for_future),
            "--output", str(future_path),
            "--horizon-min", str(args.horizon_min)
        ])
        future_files.append(future_path)

    # 5) Merge all future files into ONE combined training dataset
    combined_future = labeled_dir / f"combined_{target_os}_future.csv"
    concat_csvs(future_files, combined_future)

    # 6) Train one baseline model on combined dataset
    baseline_model_dir = models_dir / f"baseline_combined_w{args.window}_h{args.horizon_min}m"
    run([
        sys.executable, scripts["train"],
        "--input", str(combined_future),
        "--model-dir", str(baseline_model_dir),
        "--target-precision", str(args.target_precision),
        "--min-lines", str(args.min_lines)
    ])

    # 7) Inference per file using the single combined model
    for f in csv_files:
        stem = base_name_before_extension(f)
        labeled_path = labeled_dir / f"{stem}_windowz_labeled.csv"
        trends_path = labeled_dir / f"{stem}_windowz_labeled_trends.csv"
        inference_source = trends_path if (args.use_trends and trends_path.exists()) else labeled_path
        pred_path = outputs_dir / f"{stem}_predictions.csv"
        run([
            sys.executable, scripts["infer"],
            "--input", str(inference_source),
            "--model-dir", str(baseline_model_dir),
            "--output", str(pred_path),
            "--min-lines", str(args.min_lines),
            "--k-confirm", str(args.k_confirm)
        ])

    # 8) Optional: train GRU sequence model
    if args.train_seq:
        seq_dir = models_dir / f"seq_combined_w{args.window}_h{args.horizon_min}m_len{args.seq_len}"
        run([
            sys.executable, scripts["seq"],
            "--input", str(combined_future),
            "--model-dir", str(seq_dir),
            "--seq-len", str(args.seq_len)
        ])

    print(f"\n✅ ALL DONE ({target_os.upper()}) 🎉")
    print(f"📌 Combined training file: {combined_future}")
    print(f"📌 Baseline model dir:     {baseline_model_dir}")
    print(f"📌 Predictions folder:     {outputs_dir}")


if __name__ == "__main__":
    main()
