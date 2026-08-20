"""
Train a Random Forest bankruptcy classifier end to end: load data, split,
tune (or use fixed) hyperparameters, evaluate, and export the model plus
everything needed to run inference on it later.

Usage:
    python -m src.train                # full grid search (slow, thorough)
    python -m src.train --no-tune      # skip the search, use config.FIXED_BEST_PARAMS (fast)
    python -m src.train --threshold 0.3
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from src import config, evaluate, preprocessing
from src.data_loader import load_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def tune_model(X_train, y_train) -> RandomForestClassifier:
    """Grid search over config.PARAM_GRID, optimizing for recall (see config.py)."""
    base_estimator = RandomForestClassifier(
        class_weight=config.CLASS_WEIGHT, random_state=config.RANDOM_STATE
    )
    search = GridSearchCV(
        base_estimator,
        param_grid=config.PARAM_GRID,
        cv=config.CV_FOLDS,
        scoring=config.SCORING,
        n_jobs=-1,
        verbose=1,
    )
    logger.info("Starting grid search over %d parameter combinations (cv=%d)...",
                _grid_size(config.PARAM_GRID), config.CV_FOLDS)
    start = time.time()
    search.fit(X_train, y_train)
    logger.info("Grid search finished in %.1fs.", time.time() - start)
    logger.info("Best params: %s", search.best_params_)
    logger.info("Best CV %s: %.4f", config.SCORING, search.best_score_)

    # search.best_estimator_ is already refit on the full X_train with the
    # winning params (GridSearchCV's default refit=True) and carries the
    # random_state set on `base_estimator` above, so it's reproducible --
    # no need to re-instantiate/re-fit a second "best" model by hand.
    return search.best_estimator_


def fit_fixed_model(X_train, y_train) -> RandomForestClassifier:
    """Skip the search and fit directly with config.FIXED_BEST_PARAMS. Fast path for smoke tests/CI."""
    logger.info("Skipping grid search; using fixed params: %s", config.FIXED_BEST_PARAMS)
    model = RandomForestClassifier(
        **config.FIXED_BEST_PARAMS,
        class_weight=config.CLASS_WEIGHT,
        random_state=config.RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


def _grid_size(param_grid: dict) -> int:
    size = 1
    for values in param_grid.values():
        size *= len(values)
    return size


def save_artifacts(model, X_train, threshold: float, train_metrics: dict, test_metrics: dict) -> None:
    """Export the model plus everything else inference needs to reproduce these results."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, config.MODEL_PATH)
    logger.info("Saved model to %s", config.MODEL_PATH)

    medians = X_train.median().to_dict()
    joblib.dump(medians, config.FEATURE_MEDIANS_PKL_PATH)
    with open(config.FEATURE_MEDIANS_JSON_PATH, "w") as f:
        json.dump(medians, f, indent=2)
    logger.info("Saved feature medians to %s / .json", config.FEATURE_MEDIANS_PKL_PATH.with_suffix(""))

    # Everything a downstream app (e.g. a Streamlit UI) or a future
    # evaluation needs, in one place, so the model file is never used
    # without also knowing the feature order and the threshold it was
    # evaluated at.
    inference_config = {
        "feature_order": list(X_train.columns),
        "decision_threshold": threshold,
        "model_params": model.get_params(),
        "train_metrics": {k: v for k, v in train_metrics.items() if k != "classification_report"},
        "test_metrics": {k: v for k, v in test_metrics.items() if k != "classification_report"},
    }
    with open(config.INFERENCE_CONFIG_PATH, "w") as f:
        json.dump(inference_config, f, indent=2)
    logger.info("Saved inference config to %s", config.INFERENCE_CONFIG_PATH)


def run(tune: bool = True, threshold: float = config.DECISION_THRESHOLD) -> None:
    df = load_data()
    X, y = preprocessing.split_features_target(df)
    X_train, X_test, y_train, y_test = preprocessing.train_test_split_data(X, y)

    baseline_accuracy = y_train.value_counts(normalize=True).max()
    logger.info("Baseline (majority class) accuracy: %.4f", baseline_accuracy)

    model = tune_model(X_train, y_train) if tune else fit_fixed_model(X_train, y_train)

    train_metrics = evaluate.evaluate_model(model, X_train, y_train, threshold=threshold)
    test_metrics = evaluate.evaluate_model(model, X_test, y_test, threshold=threshold)

    logger.info(
        "Train -- accuracy: %.4f, recall: %.4f, precision: %.4f, f1: %.4f",
        train_metrics["accuracy"], train_metrics["recall"], train_metrics["precision"], train_metrics["f1"],
    )
    logger.info(
        "Test  -- accuracy: %.4f, recall: %.4f, precision: %.4f, f1: %.4f",
        test_metrics["accuracy"], test_metrics["recall"], test_metrics["precision"], test_metrics["f1"],
    )
    logger.info("Test classification report:\n%s",
                json.dumps(test_metrics["classification_report"], indent=2))

    evaluate.plot_confusion_matrix(
        model, X_test, y_test, threshold=threshold, save_path=config.CONFUSION_MATRIX_PLOT_PATH
    )
    evaluate.plot_feature_importance(
        model, list(X_train.columns), save_path=config.FEATURE_IMPORTANCE_PLOT_PATH
    )

    save_artifacts(model, X_train, threshold, train_metrics, test_metrics)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the bankruptcy prediction model.")
    parser.add_argument(
        "--no-tune",
        dest="tune",
        action="store_false",
        help="Skip grid search and fit directly with config.FIXED_BEST_PARAMS.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=config.DECISION_THRESHOLD,
        help="Probability threshold for the positive (bankrupt) class.",
    )
    parser.set_defaults(tune=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(tune=args.tune, threshold=args.threshold)
