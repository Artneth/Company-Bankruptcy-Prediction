import numpy as np

from src.evaluate import apply_threshold, compute_metrics


def test_apply_threshold_basic():
    probs = np.array([0.1, 0.24, 0.25, 0.6, 0.9])
    preds = apply_threshold(probs, threshold=0.25)
    assert list(preds) == [0, 0, 1, 1, 1]


def test_apply_threshold_lower_threshold_predicts_more_positives():
    probs = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    preds_strict = apply_threshold(probs, threshold=0.5)
    preds_lenient = apply_threshold(probs, threshold=0.2)
    assert preds_lenient.sum() >= preds_strict.sum()


def test_compute_metrics_perfect_predictions():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["f1"] == 1.0


def test_compute_metrics_all_wrong_on_positive_class():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 0, 0]
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["recall"] == 0.0
    assert metrics["confusion_matrix"] == [[2, 0], [2, 0]]
