#!/usr/bin/env python3
"""
02_windowize.py - Create time-bucketed windows from macOS CSV log data.

Creates aggregated windows with text content and feature counts for
downstream labeling and model training. Uses macOS-specific keyword
patterns for feature extraction.

Usage:
    python src/mac/pipeline/02_windowize.py \\
        --input data/mac/raw_csv/Mac.log.csv \\
        --output data/mac/processed/Mac_windowz.csv \\
        --window 60s
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    from src.mac.common.constants import (
        KW_PATTERNS,
        setup_logging,
        get_logger,
    )
    from src.common.constants import DEFAULT_YEAR, DEFAULT_WINDOW
except ImportError:
    KW_PATTERNS = {
        "kw_error": r"\berror\b|\bfail(ed|ure)?\b|\bdenied\b|\bcritical\b",
        "kw_sandbox": r"\bSandbox\b.*\bdeny\b|\bSandbox violation\b",
        "kw_kernel": r"\bkernel panic\b|\bKernel panic\b|\bpanic\b.*\bcpu\b",
        "kw_power": r"\bSleep\b|\bWake\b|\bpower\b.*\bstate\b|\bthermal pressure\b",
        "kw_wifi": r"\bLink Down\b|\bAirPort\b.*\b(error|fail)\b|\bchannel changed\b",
        "kw_disk": r"\bdisk\d+s\d+\b.*\berror\b|\bI/O error\b|\bfilevault\b",
        "kw_memory": r"\bmemory pressure\b|\bjetsam\b|\bVM Compressor\b",
        "kw_bluetooth": r"\bBluetooth\b.*\b(error|fail|disconnect)\b",
        "kw_usb": r"\bUSB\b.*\b(error|fail|disconnect)\b|\bIOUSB\b",
        "kw_thunderbolt": r"\bThunderbolt\b.*\b(error|fail)\b|\bIOThunderbolt\b",
    }
    DEFAULT_YEAR = 2026
    DEFAULT_WINDOW = "60s"

    def setup_logging(level: int = logging.INFO) -> None:
        logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s")

    def get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)

logger = get_logger(__name__)


def validate_input_dataframe(df: pd.DataFrame) -> None:
    """
    Validate that required columns exist in the input DataFrame.

    Args:
        df: Input DataFrame to validate

    Raises:
        ValueError: If required columns are missing
    """
    required_cols = ["timestamp"]
    optional_cols = ["process", "message", "line_no"]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available: {df.columns.tolist()}")

    for col in optional_cols:
        if col not in df.columns:
            logger.warning(f"Optional column '{col}' not found, will use defaults")


def parse_timestamps(
    df: pd.DataFrame,
    year: int = DEFAULT_YEAR
) -> pd.Series:
    """
    Parse timestamp column, handling syslog-style timestamps without year.

    Args:
        df: DataFrame with 'timestamp' column
        year: Year to assume for syslog-style timestamps

    Returns:
        Series of parsed datetime values
    """
    dt = pd.to_datetime(df["timestamp"], errors="coerce")

    # If many NaT values, likely syslog-style timestamps (no year)
    if dt.isna().mean() > 0.2:
        logger.info(f"Detected syslog-style timestamps, adding year {year}")
        dt2 = pd.to_datetime(
            df["timestamp"].astype(str) + f" {year}",
            format="%b %d %H:%M:%S %Y",
            errors="coerce"
        )
        dt = dt.fillna(dt2)

    return dt


def aggregate_window_text(group: pd.DataFrame) -> str:
    """
    Aggregate process and message columns into a single text string.

    Args:
        group: DataFrame group for a single time window

    Returns:
        Concatenated string of "process: message" entries
    """
    return " || ".join((group["proc"] + ": " + group["msg"]).tolist())


def create_windows(
    df: pd.DataFrame,
    window: str = DEFAULT_WINDOW,
    year: int = DEFAULT_YEAR
) -> pd.DataFrame:
    """
    Create time-bucketed windows from macOS log data.

    Args:
        df: Input DataFrame with log entries
        window: Window size (e.g., "30s", "60s", "5min")
        year: Year for syslog-style timestamps

    Returns:
        DataFrame with aggregated window features
    """
    validate_input_dataframe(df)

    # Parse datetime
    df = df.copy()
    df["datetime"] = parse_timestamps(df, year)

    initial_count = len(df)
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    final_count = len(df)

    if final_count == 0:
        raise ValueError("No valid timestamps found after parsing")

    if final_count < initial_count:
        logger.warning(f"Dropped {initial_count - final_count} rows with invalid timestamps")

    # Window bucket
    df["bucket"] = df["datetime"].dt.floor(window)

    # Build window text
    df["proc"] = df["process"].fillna("").astype(str) if "process" in df.columns else ""
    df["msg"] = df["message"].fillna("").astype(str) if "message" in df.columns else ""

    # Aggregate windows
    windows = (
        df.groupby("bucket")
        .agg(
            lines=(df.columns[0], "size"),
            text=("msg", lambda s: " || ".join(s.astype(str).tolist())),
        )
        .reset_index()
    )

    # Add text_with_proc column
    text_with_proc = df.groupby("bucket").apply(
        lambda g: " || ".join((g["proc"] + ": " + g["msg"]).tolist())
    )
    windows["text_with_proc"] = windows["bucket"].map(text_with_proc)

    # Add process counts (top 12 most common)
    top_procs = df["proc"].value_counts().head(12).index.tolist()
    for p in top_procs:
        if p:
            cnt = df[df["proc"] == p].groupby("bucket").size()
            windows[f"cnt_proc_{p}"] = windows["bucket"].map(cnt).fillna(0).astype(int)

    # Add keyword counts using macOS-specific patterns
    for col, pattern in KW_PATTERNS.items():
        df[col] = df["msg"].str.contains(pattern, case=False, regex=True, na=False)
        per_bucket = df.groupby("bucket")[col].sum()
        windows[col] = windows["bucket"].map(per_bucket).fillna(0).astype(int)

    # Unique process count per window
    unique_proc = df.groupby("bucket")["proc"].nunique()
    windows["kw_unique_procs"] = windows["bucket"].map(unique_proc).fillna(0).astype(int)

    return windows


def main() -> None:
    """Main entry point for Mac windowization script."""
    setup_logging()

    ap = argparse.ArgumentParser(
        description="Create time-bucketed windows from macOS CSV log data."
    )
    ap.add_argument(
        "--input",
        required=True,
        help="Input CSV with columns: timestamp, process, message"
    )
    ap.add_argument(
        "--output",
        required=True,
        help="Output windows CSV"
    )
    ap.add_argument(
        "--window",
        default=DEFAULT_WINDOW,
        help=f"Window size: 30s, 60s, 5min etc. (default: {DEFAULT_WINDOW})"
    )
    ap.add_argument(
        "--year",
        type=int,
        default=DEFAULT_YEAR,
        help=f"Year for syslog-style timestamps (default: {DEFAULT_YEAR})"
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        logger.error(f"Input file not found: {in_path}")
        raise FileNotFoundError(f"Input file not found: {in_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Reading input: {in_path}")
    try:
        df = pd.read_csv(in_path)
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        raise

    logger.info(f"Creating windows with size={args.window}")
    windows = create_windows(df, window=args.window, year=args.year)

    windows.to_csv(out_path, index=False)
    logger.info(f"✅ Saved windows: {out_path} (rows={len(windows)})")

    # Print summary
    kw_cols = [c for c in windows.columns if c.startswith("kw_")]
    for col in kw_cols:
        total = int(windows[col].sum())
        if total > 0:
            logger.info(f"  {col}: {total} occurrences")


if __name__ == "__main__":
    main()
