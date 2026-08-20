"""
Preprocessing steps shared by training and inference: splitting features
from the target, the train/test split, and computing feature medians (used
downstream to impute missing values for single-record predictions).
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


def split_features_target(
    df: pd.DataFrame, target_col: str = config.TARGET_COL
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a DataFrame into features (X) and target (y)."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def train_test_split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
):
    """
    Stratified train/test split. Stratifying on `y` matters here because
    only ~3% of rows are the positive class -- an unstratified split risks
    over- or under-representing bankruptcies in the test set, which would
    make the reported recall unreliable.
    """
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def compute_feature_medians(X_train: pd.DataFrame) -> dict:
    """
    Per-feature medians computed on the training split only (never on the
    full dataset or the test split, to avoid leakage). Used at inference
    time to fill in any features a caller doesn't supply.
    """
    medians = X_train.median()
    assert list(medians.index) == list(X_train.columns)
    return medians.to_dict()
