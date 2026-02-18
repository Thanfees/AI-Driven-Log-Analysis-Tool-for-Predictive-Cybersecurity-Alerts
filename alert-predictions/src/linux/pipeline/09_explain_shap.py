#!/usr/bin/env python3
"""
09_explain_shap.py - Generate SHAP explanations for model predictions.

Provides per-prediction explanations showing which features drove each
alert prediction. Supports both global (dataset-level) and local
(per-window) explanations.

Outputs:
    - SHAP summary CSV with top features per prediction
    - Global feature importance ranking
    - Optional SHAP waterfall/beeswarm plots

Usage:
    # Explain predictions for a file
    python src/linux/pipeline/09_explain_shap.py \\
        --input data/linux/labeled/combined_linux_future.csv \\
        --model-dir models/linux/baseline_combined_w60s_h15m \\
        --output outputs/linux/explanations.csv \\
        --top-k 10

    # With plots
    python src/linux/pipeline/09_explain_shap.py \\
        --input data/linux/labeled/combined_linux_future.csv \\
        --model-dir models/linux/baseline_combined_w60s_h15m \\
        --output outputs/linux/explanations.csv \\
        --plots-dir outputs/linux/shap_plots \\
        --top-k 10
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
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


def get_feature_names(model) -> List[str]:
    """
    Extract feature names from the trained sklearn Pipeline.

    Handles ColumnTransformer with TF-IDF + numeric features.

    Args:
        model: Trained sklearn Pipeline

    Returns:
        List of feature names in the order used by the model
    """
    feature_names = []
    features_step = model.named_steps.get("features")

    if features_step is None:
        logger.warning("No 'features' step found in pipeline")
        return feature_names

    for name, transformer, cols in features_step.transformers_:
        if name == "txt":
            # TF-IDF features
            if hasattr(transformer, "get_feature_names_out"):
                tfidf_names = transformer.get_feature_names_out().tolist()
            elif hasattr(transformer, "vocabulary_"):
                vocab = transformer.vocabulary_
                tfidf_names = sorted(vocab.keys(), key=lambda k: vocab[k])
            else:
                tfidf_names = [f"tfidf_{i}" for i in range(transformer.max_features or 0)]
            feature_names.extend([f"txt:{n}" for n in tfidf_names])
        elif name == "num":
            # Numeric features (cnt_proc_*, kw_*, etc.)
            if isinstance(cols, list):
                feature_names.extend(cols)
            else:
                feature_names.extend([f"num_{i}" for i in range(len(cols))])

    return feature_names


def transform_features(model, df: pd.DataFrame) -> np.ndarray:
    """
    Apply the feature transformation step of the model pipeline.

    Args:
        model: Trained sklearn Pipeline
        df: Input DataFrame

    Returns:
        Transformed feature matrix (sparse or dense)
    """
    features_step = model.named_steps["features"]
    X = features_step.transform(df)

    # Convert sparse to dense if needed
    if hasattr(X, "toarray"):
        X = X.toarray()

    return X


def compute_shap_values(
    model,
    X_transformed: np.ndarray,
    feature_names: List[str],
    max_samples: int = 1000,
) -> np.ndarray:
    """
    Compute SHAP values for the model predictions.

    Uses LinearExplainer for LogisticRegression (exact, fast).
    Falls back to KernelExplainer if LinearExplainer fails.

    Args:
        model: Trained sklearn Pipeline
        X_transformed: Transformed feature matrix
        feature_names: Feature names
        max_samples: Max samples for background data (KernelExplainer fallback)

    Returns:
        SHAP values matrix (n_samples x n_features)
    """
    import shap

    clf = model.named_steps["clf"]

    try:
        # LinearExplainer: exact and fast for linear models
        explainer = shap.LinearExplainer(clf, X_transformed)
        shap_values = explainer.shap_values(X_transformed)
        logger.info("Using SHAP LinearExplainer (exact)")
    except Exception as e:
        logger.warning(f"LinearExplainer failed ({e}), falling back to KernelExplainer")
        # Subsample background for KernelExplainer
        bg_size = min(max_samples, X_transformed.shape[0])
        bg_idx = np.random.choice(X_transformed.shape[0], bg_size, replace=False)
        background = X_transformed[bg_idx]

        explainer = shap.KernelExplainer(
            lambda x: clf.predict_proba(x)[:, 1],
            background
        )
        shap_values = explainer.shap_values(X_transformed)

    return shap_values


def get_top_features_per_row(
    shap_values: np.ndarray,
    feature_names: List[str],
    top_k: int = 5,
) -> List[List[Tuple[str, float]]]:
    """
    Get the top-K most influential features for each prediction.

    Args:
        shap_values: SHAP values matrix
        feature_names: Feature names
        top_k: Number of top features to return per row

    Returns:
        List of lists of (feature_name, shap_value) tuples
    """
    results = []
    for i in range(shap_values.shape[0]):
        row_vals = shap_values[i]
        # Sort by absolute SHAP value (most important first)
        sorted_idx = np.argsort(np.abs(row_vals))[::-1][:top_k]
        top = [(feature_names[j], float(row_vals[j])) for j in sorted_idx]
        results.append(top)
    return results


def compute_global_importance(
    shap_values: np.ndarray,
    feature_names: List[str],
    top_k: int = 20,
) -> pd.DataFrame:
    """
    Compute global feature importance by mean absolute SHAP value.

    Args:
        shap_values: SHAP values matrix
        feature_names: Feature names
        top_k: Number of top features to return

    Returns:
        DataFrame with feature, importance, direction columns
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    mean_signed = shap_values.mean(axis=0)

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs,
        "mean_shap": mean_signed,
        "direction": ["↑ risk" if s > 0 else "↓ risk" for s in mean_signed],
    })

    importance_df = importance_df.sort_values("mean_abs_shap", ascending=False)
    return importance_df.head(top_k).reset_index(drop=True)


def generate_plots(
    shap_values: np.ndarray,
    feature_names: List[str],
    X_transformed: np.ndarray,
    plots_dir: Path,
    top_k: int = 20,
) -> List[str]:
    """
    Generate SHAP visualization plots.

    Args:
        shap_values: SHAP values matrix
        feature_names: Feature names
        X_transformed: Transformed features
        plots_dir: Directory to save plots
        top_k: Number of top features to show

    Returns:
        List of saved plot paths
    """
    import shap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    # 1. Global bar plot (mean |SHAP|)
    try:
        fig, ax = plt.subplots(figsize=(12, 8))
        mean_abs = np.abs(shap_values).mean(axis=0)
        sorted_idx = np.argsort(mean_abs)[::-1][:top_k]
        top_features = [feature_names[i] for i in sorted_idx]
        top_values = mean_abs[sorted_idx]

        bars = ax.barh(range(len(top_features)), top_values[::-1], color="#2196F3", alpha=0.8)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features[::-1], fontsize=9)
        ax.set_xlabel("Mean |SHAP value|", fontsize=11)
        ax.set_title("Feature Importance (Global)", fontsize=14, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()

        path = plots_dir / "global_importance.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(path))
        logger.info(f"Saved: {path}")
    except Exception as e:
        logger.warning(f"Failed to create global importance plot: {e}")

    # 2. Beeswarm plot (SHAP summary)
    try:
        fig, ax = plt.subplots(figsize=(12, 8))
        # Use subset for readability
        n_show = min(shap_values.shape[0], 500)
        mean_abs = np.abs(shap_values).mean(axis=0)
        top_idx = np.argsort(mean_abs)[::-1][:top_k]

        shap_subset = shap_values[:n_show, :][:, top_idx]
        feat_subset = [feature_names[i] for i in top_idx]

        for feat_i, feat_name in enumerate(feat_subset):
            vals = shap_subset[:, feat_i]
            jitter = np.random.uniform(-0.3, 0.3, size=len(vals))
            colors = ["#E53E3E" if v > 0 else "#3182CE" for v in vals]
            ax.scatter(vals, [feat_i] * len(vals) + jitter,
                      c=colors, alpha=0.3, s=8, edgecolors="none")

        ax.set_yticks(range(len(feat_subset)))
        ax.set_yticklabels(feat_subset, fontsize=9)
        ax.set_xlabel("SHAP value (impact on prediction)", fontsize=11)
        ax.set_title("SHAP Beeswarm Plot", fontsize=14, fontweight="bold")
        ax.axvline(x=0, color="gray", linewidth=0.5, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#E53E3E',
                   markersize=8, label='Increases risk'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#3182CE',
                   markersize=8, label='Decreases risk'),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

        plt.tight_layout()
        path = plots_dir / "beeswarm.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(path))
        logger.info(f"Saved: {path}")
    except Exception as e:
        logger.warning(f"Failed to create beeswarm plot: {e}")

    # 3. Top-K positive window explanations (waterfall-style)
    try:
        mean_shap_per_row = shap_values.mean(axis=1)
        top_positive_idx = np.argsort(mean_shap_per_row)[::-1][:5]

        for rank, row_idx in enumerate(top_positive_idx):
            fig, ax = plt.subplots(figsize=(10, 6))
            row_vals = shap_values[row_idx]
            sorted_feat_idx = np.argsort(np.abs(row_vals))[::-1][:10]

            feat_names_local = [feature_names[i] for i in sorted_feat_idx]
            feat_vals = [row_vals[i] for i in sorted_feat_idx]

            colors = ["#E53E3E" if v > 0 else "#3182CE" for v in feat_vals]
            ax.barh(range(len(feat_names_local)), feat_vals[::-1], color=colors[::-1], alpha=0.8)
            ax.set_yticks(range(len(feat_names_local)))
            ax.set_yticklabels(feat_names_local[::-1], fontsize=9)
            ax.set_xlabel("SHAP value", fontsize=11)
            ax.set_title(f"Window #{row_idx} Explanation (Rank {rank+1})",
                        fontsize=13, fontweight="bold")
            ax.axvline(x=0, color="gray", linewidth=0.5, linestyle="--")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()

            path = plots_dir / f"window_{row_idx}_rank{rank+1}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved.append(str(path))

        logger.info(f"Saved {len(top_positive_idx)} individual window explanations")
    except Exception as e:
        logger.warning(f"Failed to create window explanation plots: {e}")

    return saved


def align_numeric_columns(df: pd.DataFrame, model) -> pd.DataFrame:
    """Ensure all numeric columns expected by the model exist in the DataFrame."""
    try:
        features = model.named_steps.get("features")
        if features:
            for name, trans, cols in features.transformers:
                if name == "num" and isinstance(cols, list):
                    for c in cols:
                        if c not in df.columns:
                            df[c] = 0
    except Exception:
        pass
    return df


def main() -> None:
    """Main entry point for SHAP explanation script."""
    setup_logging()

    ap = argparse.ArgumentParser(
        description="Generate SHAP explanations for model predictions."
    )
    ap.add_argument("--input", required=True,
                    help="Input CSV (labeled/trends data)")
    ap.add_argument("--model-dir", required=True,
                    help="Directory with trained model")
    ap.add_argument("--output", required=True,
                    help="Output explanations CSV")
    ap.add_argument("--text-col", default="text_with_proc",
                    help="Text column name")
    ap.add_argument("--top-k", type=int, default=10,
                    help="Number of top features per prediction (default: 10)")
    ap.add_argument("--min-lines", type=int, default=0,
                    help="Filter windows with lines < min-lines")
    ap.add_argument("--plots-dir", default=None,
                    help="Directory for SHAP plots (optional)")
    ap.add_argument("--global-importance-out", default=None,
                    help="Output path for global importance CSV (optional)")
    ap.add_argument("--max-samples", type=int, default=2000,
                    help="Max samples for SHAP computation")
    args = ap.parse_args()

    in_path = Path(args.input)
    model_dir = Path(args.model_dir)
    out_path = Path(args.output)

    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    model_path = model_dir / "final_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info(f"Loading model from: {model_dir}")
    model = joblib.load(model_path)

    # Load data
    logger.info(f"Reading input: {in_path}")
    df = pd.read_csv(in_path)

    if args.text_col not in df.columns:
        raise ValueError(f"Missing '{args.text_col}'. Columns: {df.columns.tolist()}")

    df[args.text_col] = df[args.text_col].fillna("").astype(str)
    df = align_numeric_columns(df, model)

    # Optional filter
    if args.min_lines > 0 and "lines" in df.columns:
        df = df[df["lines"].fillna(0).astype(int) >= args.min_lines].copy()
        df = df.reset_index(drop=True)

    # Subsample if very large
    if len(df) > args.max_samples:
        logger.info(f"Subsampling from {len(df)} to {args.max_samples} rows")
        df = df.sample(n=args.max_samples, random_state=42).reset_index(drop=True)

    # Get feature names and transform
    logger.info("Extracting feature names...")
    feature_names = get_feature_names(model)
    logger.info(f"Total features: {len(feature_names)}")

    logger.info("Transforming features...")
    X = transform_features(model, df)

    # Compute SHAP values
    logger.info("Computing SHAP values (this may take a moment)...")
    shap_values = compute_shap_values(model, X, feature_names)

    # Get predictions
    proba = model.predict_proba(df)[:, 1]

    # Per-row top features
    logger.info(f"Extracting top-{args.top_k} features per prediction...")
    top_features = get_top_features_per_row(shap_values, feature_names, args.top_k)

    # Build output DataFrame
    out_rows = []
    for i, (features_list) in enumerate(top_features):
        row = {
            "row_idx": i,
            "bucket": df.iloc[i].get("bucket", ""),
            "prediction_score": float(proba[i]),
        }

        for rank, (feat_name, shap_val) in enumerate(features_list, 1):
            row[f"feature_{rank}"] = feat_name
            row[f"shap_{rank}"] = round(shap_val, 6)

        # Human-readable explanation
        top3 = features_list[:3]
        explanation_parts = []
        for feat_name, shap_val in top3:
            direction = "↑" if shap_val > 0 else "↓"
            clean_name = feat_name.replace("txt:", "text:\"") + "\"" if feat_name.startswith("txt:") else feat_name
            explanation_parts.append(f"{direction} {clean_name} ({shap_val:+.4f})")
        row["explanation"] = " | ".join(explanation_parts)

        out_rows.append(row)

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(out_path, index=False)
    logger.info(f"✅ Saved explanations: {out_path} ({len(out_df)} rows)")

    # Global importance
    global_importance = compute_global_importance(shap_values, feature_names, top_k=30)
    logger.info("\n🔍 Global Feature Importance (Top 20):")
    for _, row in global_importance.head(20).iterrows():
        logger.info(f"  {row['direction']:8s} {row['feature']:40s} {row['mean_abs_shap']:.6f}")

    if args.global_importance_out:
        gi_path = Path(args.global_importance_out)
        gi_path.parent.mkdir(parents=True, exist_ok=True)
        global_importance.to_csv(gi_path, index=False)
        logger.info(f"Saved global importance: {gi_path}")
    else:
        # Save alongside explanations
        gi_path = out_path.parent / (out_path.stem + "_global_importance.csv")
        global_importance.to_csv(gi_path, index=False)
        logger.info(f"Saved global importance: {gi_path}")

    # Generate plots
    if args.plots_dir:
        plots_dir = Path(args.plots_dir)
        logger.info(f"Generating SHAP plots in: {plots_dir}")
        saved_plots = generate_plots(shap_values, feature_names, X, plots_dir, top_k=20)
        logger.info(f"Generated {len(saved_plots)} plots")

    logger.info("✅ SHAP explanation complete!")


if __name__ == "__main__":
    main()
