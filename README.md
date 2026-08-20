# Company Bankruptcy Prediction

Predicting corporate bankruptcy risk from financial ratios using a Random Forest classifier, with a full training/inference pipeline and a Streamlit app for interactive scoring.

**🔗 Live demo:** https://bankruptcyprevention-btrwedmajcnqrdqjsmsrnp.streamlit.app/
![alt text](https://raw.githubusercontent.com/Artneth/Company-Bankruptcy-Prediction/refs/heads/main/assets/app_screenshot/app_screenshot.png)

**📓 Repo:** https://github.com/Artneth/Company-Bankruptcy-Prediction

---

## Table of Contents
- [Problem](#problem)
- [Approach](#approach)
- [Why Random Forest](#why-random-forest)
- [Results](#results)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Get the data](#1-get-the-data)
  - [2. Train the model](#2-train-the-model)
  - [3. Evaluate](#3-evaluate)
  - [4. Predict on new records](#4-predict-on-new-records)
  - [5. Run the Streamlit app](#5-run-the-streamlit-app)
- [Running the tests](#running-the-tests)
- [A note on the `src/` scripts](#a-note-on-the-src-scripts)
- [Data source](#data-source)
- [License](#license)

---

## Problem

Bankruptcy is expensive and hard to see coming — by the time the signs are obvious in the headlines, most of the financial damage to creditors, investors, and employees has already happened. The goal of this project is to **flag companies at risk of bankruptcy early**, using nothing but standard financial ratios (profitability, leverage, liquidity, turnover, etc.) that are already reported in financial statements.

Concretely, this is framed as a **binary classification problem**: given a company's financial ratios, predict whether it will go bankrupt (`1`) or not (`0`).

The dataset is also **severely imbalanced** — bankrupt companies make up only a small fraction of the data — which means the real challenge isn't overall accuracy, it's making sure the model doesn't quietly miss the bankrupt companies while looking accurate on paper.

## Approach

1. **EDA** on the raw dataset (95 financial-ratio features) — class balance, feature distributions, outlier behavior, and a Spearman correlation heatmap to check for multicollinearity.
2. **Stratified train/test split** (80/20) so the ~minority bankrupt class is represented proportionally in both sets — an unstratified split risks a test set that either overstates or understates recall.
3. **Random Forest classifier** with `class_weight="balanced"` to counteract class imbalance at the algorithm level (in addition to stratification).
4. **Hyperparameter tuning** via `GridSearchCV` (5-fold CV), scored on **recall** rather than accuracy — see [Why recall](#results) below.
5. **Custom decision threshold** (0.25 instead of the default 0.5) applied at inference time, trading some precision for higher recall — deliberately, since missing a real bankruptcy is far more costly than a false alarm.
6. **Exported inference artifacts** (model, per-feature medians, feature order, threshold) so the model can be reused outside the training run — by `predict.py`, or by the Streamlit app — without any risk of feature-order mismatches or silent leakage from the training data.

### Why "recall" as the scoring metric?

For this problem, a **false negative** (predicting "not bankrupt" for a company that actually goes bankrupt) is far more costly than a **false positive** (flagging a healthy company for review). So the grid search optimizes for **recall** on the positive (bankrupt) class, and the default classification threshold is lowered from 0.5 to 0.25 to catch more true bankruptcies, accepting more false alarms in exchange.

## Why Random Forest

1. **High dimensionality** — the dataset has 95 feature columns, and Random Forest handles high-dimensional feature spaces well without needing manual feature selection.
2. **Robust to multicollinearity** — many of the financial ratios are correlated with each other (confirmed in the EDA correlation heatmap); Random Forest's performance doesn't degrade from this the way linear models' can.
3. **Robust to outliers** — financial ratio data is naturally prone to extreme values (e.g. near-zero denominators), and tree-based splits aren't distorted by them the way distance- or gradient-based models can be.
4. **No feature scaling required** — trees split on thresholds per feature, so there's no need for standardization/normalization, which simplifies the pipeline and inference code.

## Results

> **Note:** The numbers below are placeholders — swap in your actual run's metrics (from the classification report / `models/inference_config.json` produced by `train.py`).

Evaluated on the held-out test set (20% stratified split), at the default decision threshold of **0.25**:

| Metric |  | Test |
|---|---|---|
| **Recall**(bankrupt class) |  | `0.94` |
|  Accuracy |  | `0.82` |
| Precision (bankrupt class) |  | `0.60` |
| F1-score (bankrupt class) |  | `0.63` |

- **Baseline (majority-class) accuracy:** `BASELINE_ACC` — i.e. the accuracy of always predicting "not bankrupt." This is the number the model needs to beat meaningfully, since raw accuracy is a weak metric on an imbalanced dataset like this one.
- Confusion matrix and feature-importance plots are generated automatically by `train.py` and saved to `models/confusion_matrix.png` and `models/feature_importance.png`.
- Best hyperparameters found by the grid search are logged at the end of the `train.py` run, and saved in `models/inference_config.json` alongside both the train and test metrics.

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3 |
| Data handling | pandas, numpy |
| Modeling | scikit-learn (`RandomForestClassifier`, `GridSearchCV`) |
| Model persistence | joblib |
| Visualization | matplotlib, seaborn |
| App / deployment | Streamlit, Streamlit Community Cloud |
| Data source | Kaggle API (`kaggle` package) |
| Testing | pytest-style unit tests (`tests/`) |

## Project Structure

```
Company-Bankruptcy-Prediction/
│
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── main.py                          # Streamlit app entry point
│
├── assets/                          # Model/medians bundled for the deployed app
│   └── ...
│
├── data/                            # Downloaded dataset (gitignored)
│   └── .gitkeep
│
├── models/                          # Trained model + exported artifacts (gitignored)
│   └── .gitkeep
│
├── notebooks/
│   └── Company_Bankruptcy_Prediction.ipynb   # Original exploratory analysis + modeling
│
├── src/                              # Modular pipeline (see below)
│   ├── __init__.py
│   ├── config.py                     # Central paths, hyperparameters, thresholds
│   ├── data_loader.py                # Kaggle download + load + schema validation
│   ├── preprocessing.py              # Split, train/test split, feature medians
│   ├── train.py                      # Tune/fit, evaluate, export artifacts
│   ├── evaluate.py                   # Metrics + plotting utilities
│   └── predict.py                    # Load artifacts, score new records
│
└── tests/                            # Unit tests for the src/ modules
    ├── __init__.py
    ├── test_data_loader.py
    ├── test_preprocessing.py
    ├── test_evaluate.py
    └── test_predict.py
```

## Installation

```bash
git clone https://github.com/Artneth/Company-Bankruptcy-Prediction.git
cd Company-Bankruptcy-Prediction
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

All modular scripts live in `src/` and are designed to be run as modules (`python -m src.<script>`) from the project root, so relative imports (`from src import config`) resolve correctly.

### 1. Get the data

The dataset comes from Kaggle. Generate an API token at [kaggle.com/settings](https://www.kaggle.com/settings) → *Create New Token*, then either drop the downloaded `kaggle.json` into `~/.kaggle/kaggle.json`, or export it as environment variables:

```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_key
```

Then download + validate:

```bash
python -m src.data_loader           # skips download if data/data.csv already exists
python -m src.data_loader --force   # force a re-download
```

This fetches the dataset into `data/data.csv`, strips whitespace from column names, and runs sanity checks (missing target values, unexpected labels, duplicate rows, class balance).

### 2. Train the model

```bash
python -m src.train                     # full grid search (slower, thorough)
python -m src.train --no-tune           # skip the search, use config.FIXED_BEST_PARAMS (fast)
python -m src.train --threshold 0.3     # evaluate/export at a custom decision threshold
```

This loads the data, does the stratified split, tunes (or fits fixed) hyperparameters, evaluates on train/test, saves the confusion-matrix and feature-importance plots, and exports everything `predict.py` and the Streamlit app need:

- `models/rf_model.pkl` — the trained model
- `models/feature_medians.pkl` / `.json` — per-feature medians (training split only, to avoid leakage) used to impute unsupplied features at inference time
- `models/inference_config.json` — feature order, decision threshold, model hyperparameters, and train/test metrics, all in one place

### 3. Evaluate

`evaluate.py` isn't a standalone CLI — it's a set of reusable functions (`evaluate_model`, `plot_confusion_matrix`, `plot_feature_importance`) that `train.py` calls internally. Import it directly if you want to evaluate a model at a different threshold or re-plot results, e.g. back in a notebook:

```python
from src import evaluate
metrics = evaluate.evaluate_model(model, X_test, y_test, threshold=0.4)
```

### 4. Predict on new records

```bash
python -m src.predict --input sample_records.json
```

Input is a single JSON object or a list of objects, with any subset of the 95 features — anything not supplied is filled in with its training-set median:

```json
{"ROA(C) before interest and depreciation before interest": 0.37, "Debt ratio %": 0.12}
```

Output is a list of `{"bankrupt_probability": ..., "bankrupt_prediction": ...}` per record, evaluated at the threshold saved in `models/inference_config.json`.

### 5. Run the Streamlit app

```bash
streamlit run main.py
```

Lets you score **one company at a time**, either by manually searching/entering individual features or by uploading a JSON file, with the classification threshold adjustable live from the sidebar. A hosted version is live here: https://bankruptcyprevention-btrwedmajcnqrdqjsmsrnp.streamlit.app/

## Running the tests

```bash
pytest tests/
```

The tests cover the modular `src/` pipeline — data loading/validation, preprocessing, evaluation utilities, and prediction — not the exploratory notebook.

## A note on the `src/` scripts

The modular pipeline in `src/` (and its tests in `tests/`) was AI-generated — refactored from the logic originally developed in `notebooks/Company_Bankruptcy_Prediction.ipynb` — and then reviewed, tested, and finalized by me before being committed.

## Data source

[Company Bankruptcy Prediction](https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction) dataset on Kaggle, containing financial ratios for companies listed on the Taiwan Stock Exchange, labeled by whether the company went bankrupt.

## License

See [LICENSE](./LICENSE).
