"""
Fetches the "Company Bankruptcy Prediction" dataset from Kaggle and loads it
into a validated pandas DataFrame.

Auth: the Kaggle API needs credentials available either as
  - a ~/.kaggle/kaggle.json file (default location the `kaggle` package
    looks for), or
  - the KAGGLE_USERNAME / KAGGLE_KEY environment variables.
Generate a token at https://www.kaggle.com/settings -> "Create New Token".
Never commit kaggle.json or these env vars to source control.

Run directly to download + validate the data and print a quick summary:
    python -m src.data_loader
    python -m src.data_loader --force   # re-download even if the file exists
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)    


def download_dataset(
    dest_dir: Path = config.DATA_DIR,
    dataset: str = config.KAGGLE_DATASET,
    force: bool = False,
) -> Path:
    """
    Download and unzip the Kaggle dataset into `dest_dir`.

    Skips the download if `config.RAW_DATA_PATH` already exists, unless
    `force=True`. Returns the path to the raw CSV.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    raw_path = dest_dir / "data.csv"

    if raw_path.exists() and not force:
        logger.info("Found existing data at %s (use --force to re-download).", raw_path)
        return raw_path

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise ImportError(
            "The 'kaggle' package is required to download the dataset. "
            "Install it with `pip install kaggle`, then configure credentials "
            "as described in this module's docstring."
        ) from exc

    logger.info("Downloading '%s' from Kaggle into %s ...", dataset, dest_dir)
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset, path=str(dest_dir), unzip=True)

    if not raw_path.exists():
        # The dataset ships a single CSV; if Kaggle ever renames it, surface
        # a clear error instead of failing later with a confusing FileNotFoundError.
        downloaded = list(dest_dir.glob("*.csv"))
        if len(downloaded) == 1:
            downloaded[0].rename(raw_path)
        else:
            raise FileNotFoundError(
                f"Expected a single CSV in {dest_dir} after download, found: {downloaded}"
            )

    logger.info("Download complete: %s", raw_path)
    return raw_path


def validate_schema(df: pd.DataFrame, target_col: str = config.TARGET_COL) -> None:
    """
    Lightweight sanity checks on the loaded data. Raises ValueError on
    anything that would silently break downstream training; logs warnings
    for things worth a human's attention but not fatal.
    """
    if df.empty:
        raise ValueError("Loaded DataFrame is empty.")

    if target_col not in df.columns:
        raise ValueError(f"Expected target column '{target_col}' not found in columns.")

    if df[target_col].isna().any():
        raise ValueError(f"Target column '{target_col}' contains missing values.")

    unexpected_labels = set(df[target_col].unique()) - {0, 1}
    if unexpected_labels:
        raise ValueError(f"Target column has unexpected labels: {unexpected_labels}")

    n_missing_cells = int(df.isna().sum().sum())
    if n_missing_cells:
        logger.warning("Data contains %d missing cells across all columns.", n_missing_cells)

    n_duplicates = int(df.duplicated().sum())
    if n_duplicates:
        logger.warning("Data contains %d duplicate rows.", n_duplicates)

    class_balance = df[target_col].value_counts(normalize=True)
    logger.info("Class balance:\n%s", class_balance.to_string())


def load_data(
    raw_path: Path = config.RAW_DATA_PATH,
    download_if_missing: bool = True,
) -> pd.DataFrame:
    """
    Load the raw CSV into a DataFrame, downloading it first if needed.

    Column names in the source CSV have inconsistent leading/trailing
    whitespace (e.g. " Net Income to Total Assets"); this is fixed here so
    every downstream module can rely on clean column names.
    """
    if not raw_path.exists():
        if not download_if_missing:
            raise FileNotFoundError(f"No data found at {raw_path} and download_if_missing=False.")
        raw_path = download_dataset()

    df = pd.read_csv(raw_path)
    df.columns = df.columns.str.strip()
    validate_schema(df)
    return df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and validate the bankruptcy dataset.")
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if the file already exists."
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    download_dataset(force=args.force)
    df = load_data()
    logger.info("Loaded data: %d rows, %d columns.", *df.shape)
    logger.info("Duplicate rows: %d", df.duplicated().sum())


if __name__ == "__main__":
    main()
