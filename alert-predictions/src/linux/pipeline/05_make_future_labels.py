#!/usr/bin/env python3
"""
05_make_future_labels.py - Create prediction targets for early warning.

For each window, creates a y_future label indicating whether an anomaly
will occur within the specified horizon.

Usage:
    python src/linux/pipeline/05_make_future_labels.py \\
        --input data/linux/labeled/logs_windowz_labeled.csv \\
        --output data/linux/labeled/logs_future.csv \\
        --horizon-min 15
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
    from src.linux.common.constants import (
        DEFAULT_HORIZON_MIN,
        setup_logging,
        get_logger,
    )
except ImportError:
    DEFAULT_HORIZON_MIN = 15
    
    def setup_logging(level: int = logging.INFO) -> None:
        logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s")
    
    def get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)

logger = get_logger(__name__)


def validate_input(df: pd.DataFrame) -> None:
    """
    Validate input DataFrame has required columns.
    
    Args:
        df: Input DataFrame
        
    Raises:
        ValueError: If required columns are missing
    """
    if "bucket" not in df.columns:
        raise ValueError("Missing required column: bucket")
    if "window_is_anomaly" not in df.columns:
        raise ValueError("Missing column: window_is_anomaly (run 03_label_windows.py first)")


def create_future_labels(
    df: pd.DataFrame,
    horizon_min: int = DEFAULT_HORIZON_MIN
) -> pd.DataFrame:
    """
    Create y_future labels for early warning prediction.
    
    For each row, y_future=1 if any anomaly occurs within the next
    horizon_min minutes.
    
    Args:
        df: Input DataFrame with bucket and window_is_anomaly columns
        horizon_min: Prediction horizon in minutes
        
    Returns:
        DataFrame with y_future and horizon_min columns added
    """
    df = df.copy()
    
    # Parse bucket time
    df["bucket"] = pd.to_datetime(df["bucket"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["bucket"]).sort_values("bucket").reset_index(drop=True)
    after = len(df)

    if after == 0:
        raise ValueError("All bucket timestamps became NaT. Check bucket format.")
    
    if after < before:
        logger.warning(f"Dropped {before - after} rows with invalid bucket timestamps")

    # Ensure anomaly column is int
    df["window_is_anomaly"] = df["window_is_anomaly"].fillna(0).astype(int).clip(0, 1)

    horizon = pd.Timedelta(minutes=horizon_min)

    # Efficient future-labeling using sorted anomaly times
    anomaly_times = df.loc[df["window_is_anomaly"] == 1, "bucket"].tolist()

    j = 0
    y_future: List[int] = []
    for i in range(len(df)):
        t = df.at[i, "bucket"]
        # Move pointer j to first anomaly strictly after current time
        while j < len(anomaly_times) and anomaly_times[j] <= t:
            j += 1
        # Check if next anomaly is within horizon
        if j < len(anomaly_times) and anomaly_times[j] <= t + horizon:
            y_future.append(1)
        else:
            y_future.append(0)

    df["y_future"] = y_future
    df["horizon_min"] = horizon_min
    
    return df


def main() -> None:
    """Main entry point for future labels script."""
    setup_logging()
    
    ap = argparse.ArgumentParser(
        description="Create prediction targets for early warning."
    )
    ap.add_argument(
        "--input", 
        required=True, 
        help="Input CSV with bucket + window_is_anomaly"
    )
    ap.add_argument(
        "--output", 
        required=True, 
        help="Output CSV with y_future added"
    )
    ap.add_argument(
        "--horizon-min", 
        type=int, 
        default=DEFAULT_HORIZON_MIN, 
        help=f"Predict anomaly within next H minutes (default: {DEFAULT_HORIZON_MIN})"
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    
    if not in_path.exists():
        logger.error(f"Input file not found: {in_path}")
        raise FileNotFoundError(f"Input file not found: {in_path}")
    
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Reading input: {in_path}")
    df = pd.read_csv(in_path)
    
    validate_input(df)
    
    logger.info(f"Creating future labels with horizon={args.horizon_min}min")
    df = create_future_labels(df, horizon_min=args.horizon_min)

    df.to_csv(out_path, index=False)

    logger.info(f"✅ Output: {out_path}")
    logger.info(f"Rows: {len(df)}")
    logger.info(f"Window anomalies: {int(df['window_is_anomaly'].sum())} / {len(df)}")
    logger.info(f"y_future positives: {int(df['y_future'].sum())} / {len(df)}")
    logger.info(f"Median bucket diff: {df['bucket'].diff().median()}")


if __name__ == "__main__":
    main()
