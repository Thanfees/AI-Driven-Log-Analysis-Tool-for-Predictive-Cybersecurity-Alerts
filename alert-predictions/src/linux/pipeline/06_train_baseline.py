#!/usr/bin/env python3
"""
06_train_baseline.py - Train logistic regression model for early warning prediction.

Trains a combined TF-IDF + numeric feature model to predict future anomalies.
Includes threshold calibration based on precision targets.

Usage:
    python src/linux/pipeline/06_train_baseline.py \\
        --input data/linux/labeled/combined_linux_future.csv \\
        --model-dir models/linux/baseline_combined \\
        --target-precision 0.80 \\
        --min-lines 5
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    from src.linux.common.constants import (
        DEFAULT_TARGET_PRECISION,
        DEFAULT_MIN_LINES,
        setup_logging,
        get_logger,
    )
except ImportError:
    DEFAULT_TARGET_PRECISION = 0.60
    DEFAULT_MIN_LINES = 0
    
    def setup_logging(level: int = logging.INFO) -> None:
        logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s")
    
    def get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)

logger = get_logger(__name__)


def pick_numeric_cols(df: pd.DataFrame) -> List[str]:
    """
    Automatically identify numeric feature columns.
    
    Includes:
        - cnt_proc_* (process count columns)
        - kw_* (keyword count columns)
        - *_roll*_*, *_diff* (trend features)
        - lines (window line count)
    
    Args:
        df: Input DataFrame
        
    Returns:
        List of numeric column names suitable for training
    """
    cols = []
    for c in df.columns:
        if c.startswith("cnt_proc_") or c.startswith("kw_"):
            cols.append(c)
        elif ("_roll" in c) or ("_diff" in c):
            cols.append(c)

    if "lines" in df.columns:
        cols.append("lines")

    # Keep only numeric columns
    out = []
    for c in cols:
        if c in df.columns and df[c].dtype != object:
            out.append(c)

    # De-duplicate while preserving order
    seen = set()
    final = []
    for c in out:
        if c not in seen:
            seen.add(c)
            final.append(c)
    return final


def choose_threshold_by_precision(
    y_true: np.ndarray, 
    proba: np.ndarray, 
    target_precision: float
) -> float:
    """
    Pick threshold where precision >= target_precision and recall is maximized.
    
    If no threshold satisfies the precision target, falls back to the threshold
    that maximizes precision * recall.
    
    Args:
        y_true: True binary labels
        proba: Predicted probabilities
        target_precision: Minimum required precision
        
    Returns:
        Optimal decision threshold
    """
    prec, rec, thr = precision_recall_curve(y_true, proba)

    valid = np.where(prec[:-1] >= target_precision)[0]
    if len(valid) > 0:
        best_idx = valid[np.argmax(rec[valid])]
        return float(thr[best_idx])

    # Fallback: maximize precision * recall
    score = prec[:-1] * rec[:-1]
    best_idx = int(np.argmax(score))
    return float(thr[best_idx])


def validate_input(df: pd.DataFrame, text_col: str) -> None:
    """
    Validate input DataFrame has required columns.
    
    Args:
        df: Input DataFrame
        text_col: Name of text column
        
    Raises:
        ValueError: If required columns are missing
    """
    if text_col not in df.columns:
        raise ValueError(f"Missing text column '{text_col}'. Available: {df.columns.tolist()}")
    if "y_future" not in df.columns:
        raise ValueError("Missing target column 'y_future'. Run 05_make_future_labels.py first.")


def save_threshold_json(
    model_dir: Path,
    threshold: float,
    policy: str,
    target_precision: Optional[float] = None,
    percentile: Optional[float] = None,
    pr_auc: Optional[float] = None,
) -> None:
    """
    Save threshold with metadata in JSON format for better reproducibility.
    
    Args:
        model_dir: Directory to save threshold file
        threshold: Decision threshold value
        policy: Threshold selection policy used
        target_precision: Target precision if precision policy
        percentile: Percentile if percentile policy
        pr_auc: PR-AUC score
    """
    threshold_data = {
        "threshold": threshold,
        "policy": policy,
        "created_at": datetime.now().isoformat(),
    }
    if target_precision is not None:
        threshold_data["target_precision"] = target_precision
    if percentile is not None:
        threshold_data["percentile"] = percentile
    if pr_auc is not None:
        threshold_data["pr_auc"] = pr_auc
    
    # Save as JSON (primary)
    (model_dir / "threshold.json").write_text(json.dumps(threshold_data, indent=2))
    # Also save plain text for backwards compatibility
    (model_dir / "threshold.txt").write_text(str(threshold))
    
    logger.info(f"Saved threshold: {threshold:.4f} (policy={policy})")


def main() -> None:
    """Main entry point for baseline training script."""
    setup_logging()
    
    ap = argparse.ArgumentParser(
        description="Train logistic regression model for early warning prediction."
    )
    ap.add_argument(
        "--input", 
        required=True, 
        help="Input CSV with y_future column"
    )
    ap.add_argument(
        "--model-dir", 
        required=True, 
        help="Directory for model artifacts"
    )
    ap.add_argument("--text-col", default="text_with_proc")
    ap.add_argument(
        "--target-precision", 
        type=float, 
        default=DEFAULT_TARGET_PRECISION,
        help=f"Target precision for threshold selection (default: {DEFAULT_TARGET_PRECISION})"
    )
    ap.add_argument(
        "--min-lines", 
        type=int, 
        default=DEFAULT_MIN_LINES,
        help="Filter windows with lines < min-lines"
    )
    ap.add_argument("--max-features", type=int, default=50000)
    ap.add_argument("--ngram-max", type=int, default=2)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument(
        "--threshold-policy", 
        choices=["precision", "pr_max", "percentile"], 
        default="precision",
        help="How to pick decision threshold"
    )
    ap.add_argument(
        "--threshold-percentile", 
        type=float, 
        default=0.99,
        help="Percentile for percentile policy"
    )
    ap.add_argument(
        "--eval-k-confirm", 
        type=int, 
        default=0,
        help="If > 0, evaluate with K-consecutive confirmation"
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    model_dir = Path(args.model_dir)
    
    if not in_path.exists():
        logger.error(f"Input file not found: {in_path}")
        raise FileNotFoundError(f"Input file not found: {in_path}")
    
    model_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Reading input: {in_path}")
    df = pd.read_csv(in_path)
    
    validate_input(df, args.text_col)

    df[args.text_col] = df[args.text_col].fillna("").astype(str)
    df["y_future"] = df["y_future"].fillna(0).astype(int).clip(0, 1)

    # Optional filter to reduce noise/FP
    if args.min_lines > 0 and "lines" in df.columns:
        before = len(df)
        df = df[df["lines"].fillna(0).astype(int) >= args.min_lines].copy()
        df.reset_index(drop=True, inplace=True)
        logger.info(f"Filtered by lines >= {args.min_lines}: {before} -> {len(df)}")

    # Time-based split
    split_idx = int(len(df) * (1 - args.test_frac))
    if split_idx < 50 or split_idx >= len(df) - 10:
        raise ValueError("Not enough data after filtering. Lower --min-lines or collect more logs.")

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    # Early-warning training hygiene: avoid learning the anomaly itself
    # IMPORTANT: We filter current anomalies from TRAIN only to prevent the model
    # from learning to detect ongoing anomalies rather than predicting future ones.
    # Test set is kept untouched for honest evaluation.
    if "window_is_anomaly" in train_df.columns:
        before = len(train_df)
        train_df = train_df[train_df["window_is_anomaly"].fillna(0).astype(int) == 0].copy()
        train_df.reset_index(drop=True, inplace=True)
        logger.info(f"[Feature Leakage Prevention] Dropped current-anomaly windows in TRAIN: {before} -> {len(train_df)}")

    numeric_cols = pick_numeric_cols(df)
    if not numeric_cols:
        logger.warning("No numeric columns found. Training text-only.")
    else:
        logger.info(f"Using {len(numeric_cols)} numeric features")

    # Build feature transformer
    transformers = [
        ("txt",
         TfidfVectorizer(
             ngram_range=(1, max(1, args.ngram_max)),
             min_df=2,
             max_features=args.max_features
         ),
         args.text_col)
    ]

    if numeric_cols:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler(with_mean=False)),
        ])
        transformers.append(("num", num_pipe, numeric_cols))

    pre = ColumnTransformer(transformers=transformers, remainder="drop")

    clf = LogisticRegression(max_iter=4000, class_weight="balanced")
    model = Pipeline([("features", pre), ("clf", clf)])

    # Train
    logger.info("Training model...")
    model.fit(train_df, train_df["y_future"].values)

    # Evaluate
    proba = model.predict_proba(test_df)[:, 1]
    y_true = test_df["y_future"].values

    pr_auc = float(average_precision_score(y_true, proba))

    # Choose threshold per policy
    if args.threshold_policy == "precision":
        threshold = choose_threshold_by_precision(y_true, proba, args.target_precision)
    elif args.threshold_policy == "percentile":
        q = min(max(args.threshold_percentile, 0.0), 1.0)
        threshold = float(pd.Series(proba).quantile(q))
    else:  # pr_max
        threshold = choose_threshold_by_precision(y_true, proba, target_precision=-1.0)
    
    y_pred = (proba >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4)

    # Save validation scores for downstream calibration
    try:
        val = pd.DataFrame({
            "proba": proba,
            "y_true": y_true,
        })
        if "bucket" in test_df.columns:
            val["bucket"] = pd.to_datetime(test_df["bucket"], errors="coerce")
        if "lines" in test_df.columns:
            val["lines"] = test_df["lines"].values
        (model_dir / "val_scores.csv").write_text(val.to_csv(index=False))
    except Exception as e:
        logger.warning(f"Could not write val_scores.csv: {e}")

    # Optional evaluation with K-confirm
    k_report = None
    if args.eval_k_confirm and args.eval_k_confirm > 1:
        k = int(args.eval_k_confirm)
        y_pred_k = (
            pd.Series(y_pred)
            .rolling(k, min_periods=k)
            .sum()
            .fillna(0)
            .astype(int)
        )
        y_pred_k = (y_pred_k >= k).astype(int).values
        k_report = classification_report(y_true, y_pred_k, digits=4)

    logger.info(f"✅ PR-AUC: {pr_auc:.4f}")
    logger.info(f"Threshold ({args.threshold_policy}): {threshold:.4f}")
    logger.info(f"Confusion matrix:\n{cm}")
    logger.info(f"Classification report:\n{report}")

    # Save artifacts
    joblib.dump(model, model_dir / "final_model.joblib")
    
    # Save threshold as JSON with metadata
    save_threshold_json(
        model_dir=model_dir,
        threshold=threshold,
        policy=args.threshold_policy,
        target_precision=args.target_precision if args.threshold_policy == "precision" else None,
        percentile=args.threshold_percentile if args.threshold_policy == "percentile" else None,
        pr_auc=pr_auc,
    )

    # Save metrics
    metrics_body = (
        f"pr_auc={pr_auc}\n"
        f"threshold_policy={args.threshold_policy}\n"
    )
    if args.threshold_policy == "precision":
        metrics_body += f"target_precision={args.target_precision}\n"
    elif args.threshold_policy == "percentile":
        metrics_body += f"threshold_percentile={args.threshold_percentile}\n"
    metrics_body += (
        f"threshold={threshold}\n"
        f"confusion_matrix=\n{cm}\n\n"
        f"classification_report=\n{report}\n"
    )
    if k_report is not None:
        metrics_body += f"\n# K-confirm={args.eval_k_confirm}\nclassification_report_k=\n{k_report}\n"
    (model_dir / "metrics.txt").write_text(metrics_body)

    # Save comprehensive metadata JSON
    meta = {
        "task": "early_warning_prediction",
        "target": "y_future",
        "text_col": args.text_col,
        "numeric_cols_count": len(numeric_cols),
        "numeric_cols": numeric_cols,
        "threshold_policy": args.threshold_policy,
        "target_precision": args.target_precision,
        "threshold_percentile": args.threshold_percentile,
        "threshold": threshold,
        "pr_auc": pr_auc,
        "min_lines": args.min_lines,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "positive_test": int(y_true.sum()),
        "created_at": datetime.now().isoformat(),
    }
    (model_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    logger.info(f"💾 Model saved to: {model_dir}")


if __name__ == "__main__":
    main()
