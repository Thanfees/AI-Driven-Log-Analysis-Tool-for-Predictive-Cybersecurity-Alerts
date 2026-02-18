#!/usr/bin/env python3
"""
08_train_seq_gru.py - Train GRU sequence model for early warning prediction.

Uses TF-IDF + SVD embeddings fed into a GRU network for sequential
prediction of future anomalies.

Usage:
    python src/linux/pipeline/08_train_seq_gru.py \\
        --input data/linux/labeled/logs_future.csv \\
        --model-dir models/linux/seq \\
        --seq-len 10
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

# Use keras directly (TF 2.16+ compatible)
# Fallback to tensorflow.keras for older versions
try:
    import keras
    from keras import layers
except ImportError:
    from tensorflow import keras
    from tensorflow.keras import layers

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


def make_sequences(
    X: np.ndarray, 
    y: np.ndarray, 
    seq_len: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding window sequences for RNN training.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target labels (n_samples,)
        seq_len: Number of timesteps per sequence
        
    Returns:
        Tuple of (X_sequences, y_labels)
    """
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


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
        raise ValueError("Missing target column 'y_future'")


def build_gru_model(seq_len: int, n_features: int) -> keras.Model:
    """
    Build GRU model architecture.
    
    Args:
        seq_len: Sequence length (timesteps)
        n_features: Number of features per timestep
        
    Returns:
        Compiled Keras model
    """
    inp = keras.Input(shape=(seq_len, n_features))
    x = layers.GRU(64, return_sequences=False)(inp)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(curve="PR", name="pr_auc")]
    )
    return model


def main() -> None:
    """Main entry point for GRU training script."""
    setup_logging()
    
    ap = argparse.ArgumentParser(
        description="Train GRU sequence model for early warning prediction."
    )
    ap.add_argument("--input", required=True, help="Input CSV with y_future")
    ap.add_argument("--model-dir", required=True, help="Directory for model artifacts")
    ap.add_argument("--text-col", default="text_with_proc")
    ap.add_argument("--seq-len", type=int, default=10, help="Sequence length (timesteps)")
    ap.add_argument("--n-components", type=int, default=256, help="SVD components")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        logger.error(f"Input file not found: {in_path}")
        raise FileNotFoundError(f"Input file not found: {in_path}")

    logger.info(f"Reading input: {in_path}")
    df = pd.read_csv(args.input)
    
    validate_input(df, args.text_col)
    
    df[args.text_col] = df[args.text_col].fillna("").astype(str)
    y = df["y_future"].fillna(0).astype(int).values

    # Train/test split
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")

    # TF-IDF -> SVD to get dense vectors per window
    logger.info("Building TF-IDF + SVD embeddings...")
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000)
    X_train_t = tfidf.fit_transform(train_df[args.text_col])
    X_test_t = tfidf.transform(test_df[args.text_col])

    svd = TruncatedSVD(n_components=args.n_components, random_state=42)
    X_train = svd.fit_transform(X_train_t)
    X_test = svd.transform(X_test_t)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    y_train = train_df["y_future"].values
    y_test = test_df["y_future"].values

    # Create sequences
    logger.info(f"Creating sequences with length={args.seq_len}")
    Xtr, ytr = make_sequences(X_train, y_train, args.seq_len)
    Xte, yte = make_sequences(X_test, y_test, args.seq_len)

    logger.info(f"Sequence shapes: train={Xtr.shape}, test={Xte.shape}")

    # Build and train model
    model = build_gru_model(args.seq_len, Xtr.shape[-1])

    logger.info("Training GRU model...")
    model.fit(
        Xtr, ytr,
        validation_split=0.1,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)],
        verbose=1
    )

    eval_res = model.evaluate(Xte, yte, verbose=0)
    test_pr_auc = float(eval_res[1])
    logger.info(f"✅ Test PR-AUC: {test_pr_auc:.4f}")

    # Save artifacts
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    model.save(model_dir / "gru.keras")
    joblib.dump(tfidf, model_dir / "tfidf.joblib")
    joblib.dump(svd, model_dir / "svd.joblib")
    joblib.dump(scaler, model_dir / "scaler.joblib")

    # Save metadata
    meta = {
        "model_type": "GRU_sequence",
        "seq_len": args.seq_len,
        "n_components": args.n_components,
        "test_pr_auc": test_pr_auc,
        "train_sequences": int(len(ytr)),
        "test_sequences": int(len(yte)),
        "created_at": datetime.now().isoformat(),
    }
    (model_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    logger.info(f"💾 Saved seq model to: {model_dir}")


if __name__ == "__main__":
    main()
