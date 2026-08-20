import numpy as np
import pandas as pd

from src.preprocessing import (
    compute_feature_medians,
    split_features_target,
    train_test_split_data,
)


def _synthetic_df(n=200, n_features=5, positive_rate=0.1, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    y = rng.random(n) < positive_rate
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    df["Bankrupt?"] = y.astype(int)
    return df


def test_split_features_target_shapes():
    df = _synthetic_df()
    X, y = split_features_target(df)
    assert "Bankrupt?" not in X.columns
    assert len(X) == len(y) == len(df)


def test_train_test_split_data_is_stratified():
    df = _synthetic_df(n=500, positive_rate=0.1, seed=1)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y, test_size=0.2, random_state=42)

    assert len(X_train) + len(X_test) == len(X)

    train_rate = y_train.mean()
    test_rate = y_test.mean()
    full_rate = y.mean()
    # stratified split should keep class balance close to the full dataset's
    assert abs(train_rate - full_rate) < 0.05
    assert abs(test_rate - full_rate) < 0.05


def test_train_test_split_data_is_reproducible():
    df = _synthetic_df(seed=2)
    X, y = split_features_target(df)
    split_a = train_test_split_data(X, y, random_state=42)
    split_b = train_test_split_data(X, y, random_state=42)
    for a, b in zip(split_a, split_b):
        pd.testing.assert_frame_equal(a, b) if isinstance(a, pd.DataFrame) else pd.testing.assert_series_equal(a, b)


def test_compute_feature_medians_matches_columns():
    df = _synthetic_df()
    X, _ = split_features_target(df)
    medians = compute_feature_medians(X)
    assert set(medians.keys()) == set(X.columns)
    for col in X.columns:
        assert medians[col] == X[col].median()
