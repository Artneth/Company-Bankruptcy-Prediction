import numpy as np

from src.predict import _build_input_frame


MEDIANS = {"feature_a": 1.0, "feature_b": 2.0, "feature_c": 3.0}
FEATURE_ORDER = ["feature_a", "feature_b", "feature_c"]


def test_build_input_frame_fills_missing_with_medians():
    records = [{"feature_a": 10.0}]  # feature_b, feature_c omitted
    df = _build_input_frame(records, MEDIANS, FEATURE_ORDER)
    assert df.loc[0, "feature_a"] == 10.0
    assert df.loc[0, "feature_b"] == MEDIANS["feature_b"]
    assert df.loc[0, "feature_c"] == MEDIANS["feature_c"]


def test_build_input_frame_preserves_feature_order():
    records = [{"feature_c": 5.0, "feature_a": 1.0}]
    df = _build_input_frame(records, MEDIANS, FEATURE_ORDER)
    assert list(df.columns) == FEATURE_ORDER


def test_build_input_frame_ignores_unrecognized_fields(caplog):
    records = [{"feature_a": 1.0, "some_unknown_field": 99}]
    df = _build_input_frame(records, MEDIANS, FEATURE_ORDER)
    assert "some_unknown_field" not in df.columns
    assert "Ignoring unrecognized fields" in caplog.text


def test_build_input_frame_handles_multiple_records():
    records = [{"feature_a": 1.0}, {"feature_a": 2.0}, {"feature_a": 3.0}]
    df = _build_input_frame(records, MEDIANS, FEATURE_ORDER)
    assert len(df) == 3
    assert np.allclose(df["feature_a"].values, [1.0, 2.0, 3.0])
