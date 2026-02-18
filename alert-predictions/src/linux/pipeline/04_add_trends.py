#!/usr/bin/env python3
"""
04_add_trends.py - Add rolling trend features to labeled window data.

Computes rolling means, sums, and diffs for process count and keyword
features to capture temporal patterns.

Usage:
    python src/linux/pipeline/04_add_trends.py \\
        --input data/linux/labeled/logs_windowz_labeled.csv \\
        --output data/linux/labeled/logs_windowz_labeled_trends.csv
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

import pandas as pd

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    from src.linux.common.constants import setup_logging, get_logger
except ImportError:
    def setup_logging(level: int = logging.INFO) -> None:
        logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s")
    
    def get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)

logger = get_logger(__name__)


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Find columns that should have trend features computed.
    
    Args:
        df: Input DataFrame
        
    Returns:
        List of column names for trend computation
    """
    cnt_cols = [c for c in df.columns if c.startswith("cnt_proc_")]
    kw_cols = [c for c in df.columns if c.startswith("kw_")]
    return cnt_cols + kw_cols


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling trend features to the DataFrame.
    
    For each process count and keyword column, adds:
    - {col}_roll3_mean: 3-window rolling mean
    - {col}_roll6_sum: 6-window rolling sum
    - {col}_diff1: First difference (current - previous)
    
    Args:
        df: Input DataFrame with bucket, cnt_proc_*, and kw_* columns
        
    Returns:
        DataFrame with added trend features
    """
    df = df.copy()
    
    # Ensure bucket is datetime and sorted
    df["bucket"] = pd.to_datetime(df["bucket"])
    df = df.sort_values("bucket").reset_index(drop=True)
    
    feature_cols = get_feature_columns(df)
    logger.info(f"Adding trend features for {len(feature_cols)} columns")
    
    for c in feature_cols:
        df[f"{c}_roll3_mean"] = df[c].rolling(3, min_periods=1).mean()
        df[f"{c}_roll6_sum"] = df[c].rolling(6, min_periods=1).sum()
        df[f"{c}_diff1"] = df[c].diff().fillna(0)
    
    # Fill any remaining NaN values
    df = df.fillna(0)
    
    return df


def validate_input(df: pd.DataFrame) -> None:
    """
    Validate input DataFrame has required columns.
    
    Args:
        df: Input DataFrame
        
    Raises:
        ValueError: If bucket column is missing
    """
    if "bucket" not in df.columns:
        raise ValueError("Missing required column: bucket")


def main() -> None:
    """Main entry point for trend features script."""
    setup_logging()
    
    ap = argparse.ArgumentParser(
        description="Add rolling trend features to labeled window data."
    )
    ap.add_argument("--input", required=True, help="Input labeled CSV")
    ap.add_argument("--output", required=True, help="Output CSV with trend features")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        logger.error(f"Input file not found: {in_path}")
        raise FileNotFoundError(f"Input file not found: {in_path}")

    logger.info(f"Reading input: {in_path}")
    df = pd.read_csv(args.input)
    
    validate_input(df)
    
    df = add_trend_features(df)
    
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    
    # Count new trend columns
    trend_cols = [c for c in df.columns if "_roll" in c or "_diff" in c]
    logger.info(f"✅ Saved with {len(trend_cols)} trend features: {out}")


if __name__ == "__main__":
    main()
