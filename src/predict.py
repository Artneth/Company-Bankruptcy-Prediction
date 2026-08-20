"""
Load the trained model and score new records with it. This is what a
downstream app (e.g. a Streamlit UI) or a batch job would import; it's also
runnable standalone against a JSON file of records.

Usage:
    python -m src.predict --input sample_records.json

Input JSON is a single record or a list of records, e.g.:
    {"ROA(C) before interest and depreciation before interest": 0.37, ...}
Any feature not supplied is filled in with its training-set median.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import pandas as pd

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_artifacts(
    model_path: Path = config.MODEL_PATH,
    medians_path: Path = config.FEATURE_MEDIANS_PKL_PATH,
    inference_config_path: Path = config.INFERENCE_CONFIG_PATH,
):
    """Load the trained model, feature medians, and inference config (feature order + threshold)."""
    for path in (model_path, medians_path, inference_config_path):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found -- run `python -m src.train` first.")

    model = joblib.load(model_path)
    medians = joblib.load(medians_path)
    with open(inference_config_path) as f:
        inference_config = json.load(f)
    return model, medians, inference_config


def _build_input_frame(records: list[dict], medians: dict, feature_order: list[str]) -> pd.DataFrame:
    """
    Build a correctly-ordered, fully-populated input frame, filling gaps with medians.

    Incoming keys are whitespace-stripped before matching. The source Kaggle
    CSV has columns like " Debt ratio %" (leading space); data_loader.py
    strips that during training, so `medians`/`feature_order` are keyed on
    the clean name ("Debt ratio %"). Records built from the raw CSV headers
    -- which is the natural thing to do -- would otherwise match nothing and
    silently fall back to the median for every single feature.
    """
    rows = []
    for record in records:
        record = {k.strip(): v for k, v in record.items()}
        row = dict(medians)  # start from medians, so any missing feature is imputed
        row.update({k: v for k, v in record.items() if k in medians})
        unknown_keys = set(record.keys()) - set(medians.keys())
        if unknown_keys:
            logger.warning("Ignoring unrecognized fields: %s", unknown_keys)
        rows.append(row)
    return pd.DataFrame(rows)[feature_order]


def predict(records: list[dict]) -> list[dict]:
    """
    Score one or more records. Each input dict may contain any subset of the
    model's features; missing ones are filled with the training median.

    Returns a list of {"bankrupt_probability": float, "bankrupt_prediction": int}.
    """
    model, medians, inference_config = load_artifacts()
    feature_order = inference_config["feature_order"]
    threshold = inference_config["decision_threshold"]

    X = _build_input_frame(records, medians, feature_order)
    probs = model.predict_proba(X)[:, 1]

    return [
        {
            "bankrupt_probability": float(p),
            "bankrupt_prediction": int(p >= threshold),
        }
        for p in probs
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score records against the trained model.")
    parser.add_argument("--input", required=True, help="Path to a JSON file (one record or a list of records).")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    with open(args.input) as f:
        payload = json.load(f)
    records = payload if isinstance(payload, list) else [payload]

    results = predict(records)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()