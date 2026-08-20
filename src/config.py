"""
Central configuration for the bankruptcy prediction pipeline.

Every other module reads paths / constants from here instead of hardcoding
them, so there is exactly one place to change e.g. the CV fold count or the
decision threshold.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "data.csv"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "rf_model.pkl"
FEATURE_MEDIANS_PKL_PATH = MODELS_DIR / "feature_medians.pkl"
FEATURE_MEDIANS_JSON_PATH = MODELS_DIR / "feature_medians.json"
INFERENCE_CONFIG_PATH = MODELS_DIR / "inference_config.json"
CONFUSION_MATRIX_PLOT_PATH = MODELS_DIR / "confusion_matrix.png"
FEATURE_IMPORTANCE_PLOT_PATH = MODELS_DIR / "feature_importance.png"

# --------------------------------------------------------------------------
# Data source
# --------------------------------------------------------------------------
# https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction
KAGGLE_DATASET = "fedesoriano/company-bankruptcy-prediction"
TARGET_COL = "Bankrupt?"

# --------------------------------------------------------------------------
# Train / test split
# --------------------------------------------------------------------------
TEST_SIZE = 0.2
RANDOM_STATE = 42  # used everywhere a seed is needed, so a full run is reproducible

# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
CLASS_WEIGHT = "balanced"

# Grid searched by `train.py --tune` (the default). Recall is the scoring
# metric because a false negative (predicting "not bankrupt" for a company
# that goes bankrupt) is the costlier error for this problem.
PARAM_GRID = {
    "n_estimators": [125, 175, 200],
    "max_depth": [2, 3, 5],
    "min_samples_split": [225, 275, 325],
    "min_samples_leaf": [90, 110, 130],
}
CV_FOLDS = 5
SCORING = "recall"

# Fallback hyperparameters used by `train.py --no-tune` to skip the grid
# search entirely (useful for quick smoke tests / CI). These were the best
# params found during the original exploratory notebook run; re-run with
# --tune periodically to confirm they're still the best choice as the data
# or grid changes.
FIXED_BEST_PARAMS = {   
    "n_estimators": 125,
    "max_depth": 3,
    "min_samples_split": 275,
    "min_samples_leaf": 110,
}

# Probability threshold for the positive ("Bankrupt") class. 0.25 (rather
# than the default 0.5) trades some precision for higher recall, which is
# the right tradeoff for this problem -- see README for the reasoning.
DECISION_THRESHOLD = 0.25
