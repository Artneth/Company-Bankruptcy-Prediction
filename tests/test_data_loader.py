import pandas as pd
import pytest

from src.data_loader import validate_schema


def _good_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Bankrupt?": [0, 0, 0, 1],
            "ROA(C) before interest and depreciation before interest": [0.37, 0.46, 0.40, 0.20],
        }
    )


def test_validate_schema_passes_on_good_data():
    validate_schema(_good_df())  # should not raise


def test_validate_schema_raises_on_missing_target_column():
    df = _good_df().drop(columns=["Bankrupt?"])
    with pytest.raises(ValueError, match="target column"):
        validate_schema(df)


def test_validate_schema_raises_on_empty_dataframe():
    with pytest.raises(ValueError, match="empty"):
        validate_schema(pd.DataFrame())


def test_validate_schema_raises_on_missing_target_values():
    df = _good_df()
    df.loc[0, "Bankrupt?"] = None
    with pytest.raises(ValueError, match="missing values"):
        validate_schema(df)


def test_validate_schema_raises_on_unexpected_labels():
    df = _good_df()
    df.loc[0, "Bankrupt?"] = 2
    with pytest.raises(ValueError, match="unexpected labels"):
        validate_schema(df)


def test_validate_schema_warns_but_does_not_raise_on_duplicates(caplog):
    df = pd.concat([_good_df(), _good_df().iloc[[0]]], ignore_index=True)
    validate_schema(df)  # should not raise
    assert "duplicate rows" in caplog.text
