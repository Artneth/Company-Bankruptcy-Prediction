"""
Reusable evaluation utilities: metrics at a given decision threshold,
and confusion-matrix / feature-importance plots. Used by train.py, and
importable on their own for further analysis (e.g. back in a notebook).
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe backend for scripts/servers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


def apply_threshold(probs: np.ndarray, threshold: float) -> np.ndarray:
    """Convert positive-class probabilities into 0/1 predictions at `threshold`."""
    return (probs >= threshold).astype(int)


def compute_metrics(y_true, y_pred) -> dict:
    """Core metrics at whatever predictions are passed in."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def evaluate_model(model, X, y, threshold: float = 0.5) -> dict:
    """
    Evaluate `model` on (X, y) at the given probability threshold.
    Returns metrics plus the full sklearn classification report as a dict.
    """
    probs = model.predict_proba(X)[:, 1]
    preds = apply_threshold(probs, threshold)

    metrics = compute_metrics(y, preds)
    metrics["threshold"] = threshold
    metrics["classification_report"] = classification_report(
        y, preds, output_dict=True, zero_division=0
    )
    return metrics


def plot_confusion_matrix(model, X, y, threshold: float = 0.5, save_path: Path | None = None):
    """Plot (and optionally save) a confusion matrix at the given threshold."""
    probs = model.predict_proba(X)[:, 1]
    preds = apply_threshold(probs, threshold)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y, preds, ax=ax)
    ax.set_title(f"Confusion Matrix (threshold={threshold})")

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        logger.info("Saved confusion matrix plot to %s", save_path)
    plt.close(fig)
    return fig


def plot_feature_importance(
    model, feature_names: list[str], top_n: int = 10, save_path: Path | None = None
):
    """Plot (and optionally save) the top-N Gini feature importances."""
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values()

    fig, ax = plt.subplots(figsize=(8, 6))
    importances.tail(top_n).plot(kind="barh", ax=ax)
    ax.set_xlabel("Gini Importance")
    ax.set_ylabel("Feature")
    ax.set_title(f"Top {top_n} Feature Importances")

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        logger.info("Saved feature importance plot to %s", save_path)
    plt.close(fig)
    return fig
