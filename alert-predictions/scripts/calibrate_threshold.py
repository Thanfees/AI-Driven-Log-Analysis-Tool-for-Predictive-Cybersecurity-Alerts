#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def k_confirm_preds(y_raw: pd.Series, k: int) -> pd.Series:
    if k <= 1:
        return y_raw.astype(int)
    roll = (
        y_raw.astype(int)
        .rolling(k, min_periods=k)
        .sum()
        .fillna(0)
        .astype(int)
    )
    return (roll >= k).astype(int)


def count_alerts_per_day(bucket: pd.Series, y_conf: pd.Series) -> float:
    # Count rising edges per day
    df = pd.DataFrame({"bucket": bucket, "y": y_conf}).dropna(subset=["bucket"]).copy()
    df = df.sort_values("bucket").reset_index(drop=True)
    # group by day -> count rising edges in confirmed predictions
    df["day"] = df["bucket"].dt.floor("D")
    df["rise"] = df["y"].diff().fillna(0)
    per_day = df.groupby("day")["rise"].apply(lambda s: int((s == 1).sum()))
    if len(per_day) == 0:
        return 0.0
    return float(per_day.mean())


def main():
    ap = argparse.ArgumentParser(description="Calibrate decision threshold to target alerts/day on validation")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--val-scores", default=None, help="Optional path to val_scores.csv (otherwise use model-dir)")
    ap.add_argument("--k-confirm", type=int, default=3)
    ap.add_argument("--target-alerts-per-day", type=float, default=3.0)
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    scores_path = Path(args.val_scores) if args.val_scores else (model_dir / "val_scores.csv")
    if not scores_path.exists():
        raise SystemExit(f"❌ Missing val_scores.csv: {scores_path}")

    df = pd.read_csv(scores_path)
    if "proba" not in df.columns or "y_true" not in df.columns:
        raise SystemExit("❌ val_scores.csv must contain columns: proba, y_true, and optional bucket")

    # Prepare thresholds grid from quantiles of proba
    probs = pd.Series(df["proba"].values)
    qs = np.unique(np.linspace(0.50, 0.999, 100))
    thr_grid = np.unique(np.quantile(probs, qs))
    if thr_grid.size == 0:
        thr_grid = np.array([float(probs.median())])

    best_thr = None
    best_diff = None
    best_stats = None

    has_bucket = "bucket" in df.columns
    if has_bucket:
        df["bucket"] = pd.to_datetime(df["bucket"], errors="coerce")

    for thr in thr_grid[::-1]:  # high -> low thresholds
        y_raw = (df["proba"] >= thr).astype(int)
        y_conf = k_confirm_preds(y_raw, args.k_confirm)
        # alerts/day
        if has_bucket and df["bucket"].notna().any():
            apd = count_alerts_per_day(df["bucket"], y_conf)
        else:
            # fallback: use rate per N windows (~per batch). Not ideal.
            apd = float((y_conf.diff().fillna(0) == 1).sum())

        diff = abs(apd - args.target_alerts_per_day)
        if (best_diff is None) or (diff < best_diff):
            best_diff = diff
            best_thr = float(thr)
            # compute precision/recall with confirmed preds
            tp = int(((y_conf == 1) & (df["y_true"] == 1)).sum())
            fp = int(((y_conf == 1) & (df["y_true"] == 0)).sum())
            fn = int(((y_conf == 0) & (df["y_true"] == 1)).sum())
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            best_stats = {
                "alerts_per_day": apd,
                "precision_k": precision,
                "recall_k": recall,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }

    # Save threshold override
    (model_dir / "threshold.txt").write_text(str(best_thr))

    # Write calibration report
    rep = (
        f"target_alerts_per_day={args.target_alerts_per_day}\n"
        f"k_confirm={args.k_confirm}\n"
        f"best_threshold={best_thr}\n"
        f"alerts_per_day={best_stats['alerts_per_day'] if best_stats else 'n/a'}\n"
        f"precision_k={best_stats['precision_k'] if best_stats else 'n/a'}\n"
        f"recall_k={best_stats['recall_k'] if best_stats else 'n/a'}\n"
        f"tp={best_stats['tp'] if best_stats else 'n/a'}\n"
        f"fp={best_stats['fp'] if best_stats else 'n/a'}\n"
        f"fn={best_stats['fn'] if best_stats else 'n/a'}\n"
    )
    (model_dir / "calibration.txt").write_text(rep)

    print("✅ Calibration complete")
    print(rep)


if __name__ == "__main__":
    main()

