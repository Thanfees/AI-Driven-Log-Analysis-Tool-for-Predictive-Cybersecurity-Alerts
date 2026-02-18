#!/usr/bin/env python3
"""
07_infer_baseline.py - Run batch inference with trained baseline model.

Loads the trained model and generates predictions with K-consecutive
confirmation for reducing false positives.

Usage:
    python src/linux/pipeline/07_infer_baseline.py \\
        --input data/linux/labeled/logs_windowz_labeled_trends.csv \\
        --model-dir models/linux/baseline_combined \\
        --output outputs/linux/predictions.csv \\
        --min-lines 5 \\
        --k-confirm 3
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

import joblib
import pandas as pd

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    from src.linux.common.constants import (
        DEFAULT_MIN_LINES,
        DEFAULT_K_CONFIRM,
        setup_logging,
        get_logger,
    )
except ImportError:
    DEFAULT_MIN_LINES = 5
    DEFAULT_K_CONFIRM = 3
    
    def setup_logging(level: int = logging.INFO) -> None:
        logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s")
    
    def get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)

logger = get_logger(__name__)


def load_threshold(model_dir: Path) -> float:
    """
    Load decision threshold from model directory.
    
    Tries JSON format first (new), falls back to plain text (legacy).
    
    Args:
        model_dir: Path to model directory
        
    Returns:
        Decision threshold value
    """
    json_path = model_dir / "threshold.json"
    txt_path = model_dir / "threshold.txt"
    
    if json_path.exists():
        data = json.loads(json_path.read_text())
        return float(data["threshold"])
    elif txt_path.exists():
        return float(txt_path.read_text().strip())
    else:
        raise FileNotFoundError(f"No threshold file found in {model_dir}")


def align_numeric_columns(df: pd.DataFrame, model) -> pd.DataFrame:
    """
    Ensure all numeric columns expected by the model exist in the DataFrame.
    
    Missing columns are added with zeros to prevent ColumnTransformer errors.
    
    Args:
        df: Input DataFrame
        model: Trained sklearn pipeline
        
    Returns:
        DataFrame with aligned columns
    """
    try:
        features = getattr(model, "named_steps", {}).get("features")
        expected_num_cols = []
        if features is not None and hasattr(features, "transformers"):
            for name, trans, cols in features.transformers:
                if name == "num" and isinstance(cols, list):
                    expected_num_cols = list(cols)
                    break
        
        missing = [c for c in expected_num_cols if c not in df.columns]
        if missing:
            for c in missing:
                df[c] = 0
            logger.info(f"ℹ️ Added {len(missing)} missing numeric columns as zeros")
    except Exception as e:
        logger.warning(f"Could not align numeric columns: {e}")
    
    return df


def apply_k_confirm(predictions: pd.Series, k: int) -> pd.Series:
    """
    Apply K-consecutive confirmation to reduce false positives.
    
    A window is only marked as positive if K consecutive windows
    are all predicted positive.
    
    Args:
        predictions: Raw binary predictions
        k: Number of consecutive positives required
        
    Returns:
        Confirmed binary predictions
    """
    if k <= 1:
        return predictions.astype(int)
    
    confirmed = (
        predictions.astype(int)
        .rolling(k, min_periods=k)
        .sum()
        .fillna(0)
        .astype(int)
    )
    return (confirmed >= k).astype(int)


def format_row(row: pd.Series) -> str:
    """Format a prediction row for console output."""
    bucket = str(row.get("bucket", ""))[:19]
    score = f"{float(row['early_warning_score']):.4f}"
    pr = int(row.get("predict_raw", 0))
    pc = int(row.get("predict_confirmed", 0))
    ln = int(row.get("lines", 0))
    ex = str(row.get("example_text", ""))[:120].replace("\n", " ")
    return f"{bucket}  score={score}  raw={pr}  conf={pc}  lines={ln}  {ex}"


def main() -> None:
    """Main entry point for baseline inference script."""
    setup_logging()
    
    ap = argparse.ArgumentParser(
        description="Run batch inference with trained baseline model."
    )
    ap.add_argument("--input", required=True, help="Input labeled/trends CSV")
    ap.add_argument("--model-dir", required=True, help="Directory with trained model")
    ap.add_argument("--output", required=True, help="Output predictions CSV")
    ap.add_argument("--text-col", default="text_with_proc")
    ap.add_argument(
        "--min-lines", 
        type=int, 
        default=DEFAULT_MIN_LINES,
        help=f"Ignore windows with lines < min-lines (default: {DEFAULT_MIN_LINES})"
    )
    ap.add_argument(
        "--k-confirm", 
        type=int, 
        default=DEFAULT_K_CONFIRM,
        help=f"Require K consecutive positives to confirm (default: {DEFAULT_K_CONFIRM})"
    )
    ap.add_argument("--print-top", type=int, default=0, help="Print top-N rows by score")
    ap.add_argument("--print-all", action="store_true", help="Print every prediction")
    args = ap.parse_args()

    in_path = Path(args.input)
    model_dir = Path(args.model_dir)
    out_path = Path(args.output)
    
    # Validate inputs
    if not in_path.exists():
        logger.error(f"Input file not found: {in_path}")
        raise FileNotFoundError(f"Input file not found: {in_path}")
    
    model_path = model_dir / "final_model.joblib"
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load model and threshold
    logger.info(f"Loading model from: {model_dir}")
    model = joblib.load(model_path)
    threshold = load_threshold(model_dir)
    logger.info(f"Using threshold: {threshold:.4f}")

    # Load and validate data
    logger.info(f"Reading input: {in_path}")
    df = pd.read_csv(in_path)

    if args.text_col not in df.columns:
        raise ValueError(f"Missing '{args.text_col}'. Columns: {df.columns.tolist()}")

    df[args.text_col] = df[args.text_col].fillna("").astype(str)

    # Align numeric columns
    df = align_numeric_columns(df, model)

    # Optional noise filter
    if args.min_lines > 0 and "lines" in df.columns:
        before = len(df)
        df = df[df["lines"].fillna(0).astype(int) >= args.min_lines].copy()
        df = df.reset_index(drop=True)
        logger.info(f"Filtered by lines >= {args.min_lines}: {before} -> {len(df)}")

    # Sort by bucket for k-confirm
    if "bucket" in df.columns:
        df["bucket"] = pd.to_datetime(df["bucket"], errors="coerce")
        df = df.sort_values("bucket").reset_index(drop=True)

    # Run inference
    logger.info("Running inference...")
    proba = model.predict_proba(df)[:, 1]
    pred_raw = (proba >= threshold).astype(int)

    # Apply K-confirm
    k = max(1, int(args.k_confirm))
    pred_confirm = apply_k_confirm(pd.Series(pred_raw), k).values

    # Build output DataFrame
    out = pd.DataFrame({
        "bucket": df["bucket"] if "bucket" in df.columns else range(len(df)),
        "early_warning_score": proba,
        "predict_raw": pred_raw,
        "predict_confirmed": pred_confirm,
        "threshold": threshold,
        "k_confirm": k,
    })

    if "lines" in df.columns:
        out["lines"] = df["lines"].values

    out["example_text"] = (
        df[args.text_col]
        .str.slice(0, 250)
        .str.replace("\n", " ", regex=False)
    )

    out.to_csv(out_path, index=False)
    
    raw_count = int(out["predict_raw"].sum())
    conf_count = int(out["predict_confirmed"].sum())
    
    logger.info(f"✅ Saved predictions: {out_path}")
    logger.info(f"Raw positives: {raw_count}")
    logger.info(f"Confirmed positives: {conf_count}")

    # Optional console output
    if args.print_all:
        logger.info("=== Predictions (all) ===")
        for _, r in out.iterrows():
            print(format_row(r))
    elif args.print_top and args.print_top > 0:
        top = out.sort_values("early_warning_score", ascending=False).head(args.print_top)
        logger.info(f"=== Top {args.print_top} by score ===")
        for _, r in top.iterrows():
            print(format_row(r))


if __name__ == "__main__":
    main()
