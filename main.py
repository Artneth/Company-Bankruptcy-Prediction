"""
Company Bankruptcy Prediction — Streamlit App
================================================
Loads a trained RandomForestClassifier (rf_model.pkl) and per-feature
medians (feature_medians.pkl) to predict bankruptcy risk for ONE company
at a time. There are two ways to provide that company's data:
  1. Manually select features (searchable picker) and enter values.
  2. Upload a JSON file with the company's known feature values.
Any of the 95 features not supplied falls back to its median value from
the training data.

Classification rule: a company is flagged "Bankrupt" if
P(bankrupt) >= threshold, where the threshold is adjustable (0-1) from
the sidebar. It defaults to 25% rather than the usual 50%, since the
model was tuned to prioritize recall (catching bankruptcies) over
precision — but you can move it to whatever cutoff suits your use case.

Run with:
    streamlit run main.py

Expects rf_model.pkl and feature_medians.pkl in the same folder as this
file (a sidebar uploader is offered as a fallback if they're missing).
"""

from pathlib import Path

import joblib
import json
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
DEFAULT_THRESHOLD = 0.25
MODEL_PATH = Path("./assets/model/rf_model.pkl")
MEDIANS_PATH = Path("./assets/fallback_values/feature_medians.pkl")

st.set_page_config(page_title="Company Bankruptcy Prediction", layout="wide")


# --------------------------------------------------------------------------
# Load model + medians (with a sidebar fallback uploader if not found)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_artifacts(model_src, medians_src):
    model = joblib.load(model_src)
    medians = joblib.load(medians_src)

    # Prefer the exact column order sklearn recorded when the model was
    # fit on a DataFrame — this is guaranteed to match what predict()
    # expects. Fall back to the medians dict order if unavailable.
    if hasattr(model, "feature_names_in_"):
        feature_order = list(model.feature_names_in_)
    else:
        feature_order = list(medians.keys())

    medians = {k: float(v) for k, v in medians.items()}
    return model, medians, feature_order


model_src, medians_src = MODEL_PATH, MEDIANS_PATH
if not MODEL_PATH.exists() or not MEDIANS_PATH.exists():
    st.sidebar.warning(
        "Couldn't find `rf_model.pkl` / `feature_medians.pkl` next to main.py. "
        "Upload them below."
    )
    up_model = st.sidebar.file_uploader("rf_model.pkl", type="pkl")
    up_medians = st.sidebar.file_uploader("feature_medians.pkl", type="pkl")
    if up_model is not None and up_medians is not None:
        model_src, medians_src = up_model, up_medians
    else:
        st.info("Waiting for `rf_model.pkl` and `feature_medians.pkl` to be uploaded in the sidebar.")
        st.stop()

model, MEDIANS, FEATURES = load_artifacts(model_src, medians_src)
N_FEATURES = len(FEATURES)


# --------------------------------------------------------------------------
# Threshold control (sidebar) — applies to both input methods
# --------------------------------------------------------------------------
st.sidebar.subheader("Classification threshold")
threshold = st.sidebar.slider(
    "Classify as Bankrupt when P(bankrupt) ≥",
    min_value=0.0,
    max_value=1.0,
    value=DEFAULT_THRESHOLD,
    step=0.01,
    help="Lower this to flag more companies as at-risk (higher recall, more false "
    "alarms). Raise it to only flag the clearest cases (higher precision, more misses).",
)
st.sidebar.caption(f"Current threshold: **{threshold:.0%}**")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _normalized_lookup(feature_list):
    """feature.strip().lower() -> canonical feature name (handles the
    dataset's quirky leading/trailing whitespace in column names)."""
    return {f.strip().lower(): f for f in feature_list}


NORMALIZED_FEATURES = _normalized_lookup(FEATURES)


def normalize_record(rec: dict) -> dict:
    """Map a user-supplied record's keys onto the model's exact feature
    names, tolerating extra/missing whitespace. Keys that still don't
    match anything are left as-is (surfaced later as 'unknown keys')."""
    out = {}
    for k, v in rec.items():
        canonical = NORMALIZED_FEATURES.get(str(k).strip().lower())
        out[canonical if canonical else k] = v
    return out


def build_row(user_values: dict) -> pd.DataFrame:
    """One-row DataFrame: user-supplied values, medians for everything else."""
    row = {feat: user_values.get(feat, MEDIANS[feat]) for feat in FEATURES}
    return pd.DataFrame([row], columns=FEATURES)


def predict_proba_bankrupt(df: pd.DataFrame) -> float:
    return float(model.predict_proba(df)[:, 1][0])


def run_prediction(values_dict: dict, state_prefix: str):
    """Shared Predict button + result banner, used by both the
    manual-entry and JSON-upload input methods. The classification label
    is derived from the sidebar threshold every rerun, so moving the
    slider updates the verdict instantly without re-running the model."""
    if st.button("Run Prediction", type="primary", key=f"{state_prefix}_predict_btn"):
        input_df = build_row(values_dict)
        st.session_state[f"{state_prefix}_proba"] = predict_proba_bankrupt(input_df)

    proba_key = f"{state_prefix}_proba"
    if proba_key in st.session_state:
        proba = st.session_state[proba_key]
        if proba >= threshold:
            st.error(f"⚠️ Predicted: **Bankrupt** (threshold {threshold:.0%})")
        else:
            st.success(f"✅ Predicted: **Not Bankrupt** (threshold {threshold:.0%})")


st.title("Find out if a company will go Bankrupt")
st.markdown(
    f"""
    <p style="font-size: 18px;">
        Identifies 94% of companies at risk of bankruptcy ·
        Random Forest model · {N_FEATURES} features ·
        Set the classification threshold in the sidebar.
    </p>
    """,
    unsafe_allow_html=True
)

tab_manual, tab_json = st.tabs(["Manually Select Features", "Upload JSON"])

# ==========================================================================
# OPTION 1: MANUAL FEATURE ENTRY
# ==========================================================================
with tab_manual:
    if "manual_values" not in st.session_state:
        st.session_state.manual_values = {}

    st.subheader("1. Enter known Financial Health Indicators")
    st.caption(
        "Search for a feature below, set its value, and add it. Any feature you "
        "don't set will use its dataset median automatically."
    )

    with st.form("add_feature_form", clear_on_submit=True):
        col1, col2, col3 = st.columns([3, 2, 1])
        with col1:
            feature_choice = st.selectbox(
                "Search feature",
                options=FEATURES,
                index=None,
                placeholder="Type to search a feature name...",
            )
        with col2:
            default_val = float(MEDIANS[feature_choice]) if feature_choice else 0.0
            value = st.number_input(
                "Value",
                value=default_val,
                format="%.6f",
                key=f"value_input_{feature_choice}",
            )
        with col3:
            st.write("")
            st.write("")
            submitted = st.form_submit_button("Add / Update")

        if submitted:
            if feature_choice:
                st.session_state.manual_values[feature_choice] = value
            else:
                st.warning("Select a feature first.")

    if st.session_state.manual_values:
        st.write(f"**Features set manually ({len(st.session_state.manual_values)} / {N_FEATURES}):**")
        manual_df = pd.DataFrame(
            [{"Feature": k, "Value": v} for k, v in st.session_state.manual_values.items()]
        )
        st.dataframe(manual_df, use_container_width=True, hide_index=True)

        remove_choices = st.multiselect(
            "Remove feature(s)", options=list(st.session_state.manual_values.keys())
        )
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Remove selected"):
                for f in remove_choices:
                    st.session_state.manual_values.pop(f, None)
                st.rerun()
        with col_b:
            if st.button("Clear all"):
                st.session_state.manual_values = {}
                st.rerun()
    else:
        st.info("No features set yet — all 95 features will use their median values.")

    st.subheader("2. Predict")
    run_prediction(st.session_state.manual_values, state_prefix="manual")

# ==========================================================================
# OPTION 2: JSON UPLOAD (single company)
# ==========================================================================
with tab_json:
    st.subheader("1. Upload Financial Health Indicators (JSON)")
    st.caption(
        "Upload a JSON file for a **single company**: either a flat "
        '`{"feature": value, ...}` object, or `{"Company Name": {"feature": value, ...}}`. '
        "Any feature you don't include will use its dataset median."
    )

    example_features = FEATURES[:3]
    example_record = {f: MEDIANS[f] for f in example_features}
    template = {"Company A": example_record}
    st.download_button(
        "Download example JSON template",
        data=json.dumps(template, indent=2),
        file_name="company_template.json",
        mime="application/json",
    )

    uploaded = st.file_uploader("Upload JSON file", type=["json"], key="json_uploader")

    if uploaded is not None:
        try:
            payload = json.load(uploaded)
        except Exception as e:
            st.error(f"Could not parse JSON: {e}")
            payload = None

        company_name = None
        record = None

        if isinstance(payload, list):
            st.error(
                "This app predicts one company at a time — please upload a single "
                "JSON object, not a list of multiple entries."
            )
        elif isinstance(payload, dict):
            values = list(payload.values())
            if values and all(isinstance(v, dict) for v in values):
                # Looks like {"Company Name": {feature: value}}
                if len(payload) == 1:
                    company_name, record = next(iter(payload.items()))
                else:
                    st.error(
                        f"Found {len(payload)} companies in this file, but only one "
                        "company can be predicted at a time. Please upload a file "
                        "with a single company."
                    )
            else:
                # Flat {feature: value}
                record = payload
        elif payload is not None:
            st.error("JSON must be an object — either a feature dictionary, or one company name mapped to its feature dictionary.")

        if record is not None:
            normalized_record = normalize_record(record)
            unknown_keys = set(normalized_record.keys()) - set(FEATURES)
            if unknown_keys:
                st.warning(
                    f"Ignored {len(unknown_keys)} key(s) not among the model's 95 features: "
                    + ", ".join(sorted(str(k) for k in unknown_keys))[:300]
                )
            st.session_state.json_record = normalized_record
            st.session_state.json_company_name = company_name

    if "json_record" in st.session_state:
        label = st.session_state.json_company_name or "this company"
        n_provided = len(st.session_state.json_record)
        st.success(f"Loaded data for **{label}** — {n_provided} of {N_FEATURES} feature(s) provided.")

        st.subheader("2. Predict")
        run_prediction(st.session_state.json_record, state_prefix="json")
