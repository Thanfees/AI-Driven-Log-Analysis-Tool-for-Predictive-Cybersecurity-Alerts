#!/usr/bin/env python3
"""
03_label_windows.py - Apply rule-based anomaly labels to windowed Windows log data.

Uses pattern matching to identify anomalous windows based on Windows-specific
CBS/CSI events (manifest errors, update failures, corruption, etc.).

Usage:
    python src/windows/pipeline/03_label_windows.py \\
        --input data/windows/processed/windows20k_windowz.csv \\
        --output data/windows/labeled/windows20k_windowz_labeled.csv \\
        --text-col text_with_proc
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple

import pandas as pd

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    from src.windows.common.constants import (
        LABELING_RULES,
        setup_logging,
        get_logger,
    )
except ImportError:
    import re
    LABELING_RULES = [
        ("CBS_FAIL", re.compile(r"\bCBS\b.*\b(fail|error)\b", re.IGNORECASE)),
        ("MANIFEST_ERROR", re.compile(r"\bCBS_E_MANIFEST\b|\bmanifest\b.*\binvalid\b", re.IGNORECASE)),
        ("UPDATE_FAIL", re.compile(r"\bWindows\s*Update\b.*\bfail\b", re.IGNORECASE)),
        ("HRESULT_ERROR", re.compile(r"\bHRESULT\s*=\s*0x8[0-9a-fA-F]{7}\b", re.IGNORECASE)),
    ]

    def setup_logging(level: int = logging.INFO) -> None:
        logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s")

    def get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)

logger = get_logger(__name__)


def label_window(text: str) -> Tuple[int, str]:
    """
    Apply Windows labeling rules to window text to detect anomalies.

    Args:
        text: Concatenated text content of the window

    Returns:
        Tuple of (is_anomaly: 0 or 1, label: rule name or "NORMAL")
    """
    if not isinstance(text, str):
        return 0, "NORMAL"

    for name, rx in LABELING_RULES:
        if rx.search(text):
            return 1, name

    return 0, "NORMAL"


def validate_input(df: pd.DataFrame, text_col: str) -> None:
    """
    Validate that required columns exist in the input DataFrame.

    Args:
        df: Input DataFrame
        text_col: Name of text column to use for labeling

    Raises:
        ValueError: If text column is missing
    """
    if text_col not in df.columns:
        available = df.columns.tolist()
        raise ValueError(f"Missing column '{text_col}'. Available: {available}")


def main() -> None:
    """Main entry point for Windows window labeling script."""
    setup_logging()

    ap = argparse.ArgumentParser(
        description="Apply rule-based anomaly labels to windowed Windows log data."
    )
    ap.add_argument("--input", required=True, help="Input windowed CSV")
    ap.add_argument("--output", required=True, help="Output labeled CSV")
    ap.add_argument(
        "--text-col",
        default="text_with_proc",
        help="Column containing window text (default: text_with_proc)"
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        logger.error(f"Input file not found: {in_path}")
        raise FileNotFoundError(f"Input file not found: {in_path}")

    logger.info(f"Reading input: {in_path}")
    df = pd.read_csv(args.input)

    validate_input(df, args.text_col)

    df[args.text_col] = df[args.text_col].fillna("").astype(str)

    logger.info(f"Applying {len(LABELING_RULES)} Windows labeling rules...")
    labels = df[args.text_col].apply(label_window)
    df["window_is_anomaly"] = labels.apply(lambda x: x[0]).astype(int)
    df["window_label"] = labels.apply(lambda x: x[1]).astype(str)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    anomaly_count = int(df["window_is_anomaly"].sum())
    total_count = len(df)

    logger.info(f"✅ Saved: {out_path}")
    logger.info(f"Anomaly windows: {anomaly_count} / {total_count} ({100*anomaly_count/total_count:.2f}%)")

    # Show label distribution
    label_counts = df["window_label"].value_counts()
    logger.info("Label distribution:")
    for label, count in label_counts.head(10).items():
        logger.info(f"  {label}: {count}")


if __name__ == "__main__":
    main()
