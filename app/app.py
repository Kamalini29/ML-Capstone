"""
23CSE301 ML Capstone — Prediction GUI (all models, both tracks)
=================================================================
Run with:  streamlit run app.py   (from inside the app/ folder, so ../models
                                    resolves correctly — adjust MODELS_DIR if
                                    your layout differs)

Loads every saved model from both notebooks (not just the "best" one) and lets
the user pick a track, pick ANY algorithm from that track, enter feature values,
and see that specific model's prediction — plus, optionally, a side-by-side
comparison of every model's prediction on the same input at once.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="ML Capstone — Predictions", layout="wide")
st.title("ML Capstone — Predict with Any Model")

MODELS_DIR = Path("../models")

# ------------------------------------------------------------------
# Load everything once
# ------------------------------------------------------------------
@st.cache_resource
def load_regression_assets():
    models = joblib.load(MODELS_DIR / "regression_all_models.pkl")
    scaler = joblib.load(MODELS_DIR / "regression_scaler.pkl")
    feature_names = joblib.load(MODELS_DIR / "regression_feature_names.pkl")
    scores = pd.read_csv(MODELS_DIR / "regression_model_scores.csv")
    return models, scaler, feature_names, scores

@st.cache_resource
def load_classification_assets():
    models = joblib.load(MODELS_DIR / "classification_all_models.pkl")
    scaler = joblib.load(MODELS_DIR / "classification_scaler.pkl")
    feature_names = joblib.load(MODELS_DIR / "classification_feature_names.pkl")
    label_encoder = joblib.load(MODELS_DIR / "classification_label_encoder.pkl")
    scores = pd.read_csv(MODELS_DIR / "classification_model_scores.csv")
    return models, scaler, feature_names, label_encoder, scores

track = st.sidebar.radio("Track", ["Regression (Oil Well)", "Classification (Play Store Apps)"])

# ------------------------------------------------------------------
# REGRESSION
# ------------------------------------------------------------------
if track.startswith("Regression"):
    try:
        models, scaler, feature_names, scores = load_regression_assets()
    except FileNotFoundError as e:
        st.error(f"Missing saved file: {e}. Run every cell in notebooks/regression.ipynb first, "
                 f"including the final 'save all models' cell.")
        st.stop()

    st.subheader("Regression — pick a model")
    st.dataframe(scores.sort_values("R2", ascending=False).reset_index(drop=True))

    model_name = st.selectbox("Algorithm", list(models.keys()))

    st.write("**Enter feature values**")
    user_input = {}
    cols = st.columns(3)
    for i, feat in enumerate(feature_names):
        with cols[i % 3]:
            user_input[feat] = st.number_input(feat, value=0.0, key=f"reg_{feat}")

    compare_all = st.checkbox("Also show every model's prediction on this input")

    if st.button("Predict", key="reg_predict"):
        input_df = pd.DataFrame([user_input])[feature_names]
        input_scaled = pd.DataFrame(scaler.transform(input_df), columns=feature_names)

        if model_name.startswith("Polynomial"):
            st.caption("Note: polynomial models extrapolate poorly for inputs far outside the "
                       "training data's range — treat extreme predictions from this model with caution.")

        pred = models[model_name].predict(input_scaled)[0]
        st.success(f"**{model_name}** predicts: **{pred:.3f}**")

        if compare_all:
            rows = []
            for name, m in models.items():
                p = m.predict(input_scaled)[0]
                rows.append({"Model": name, "Prediction": p})
            st.write("**All models on this same input:**")
            st.dataframe(pd.DataFrame(rows).sort_values("Prediction").reset_index(drop=True))
            st.caption("Polynomial models can swing wildly if the input is far from the training "
                       "data's range — that's expected behavior for that algorithm, not a bug.")

# ------------------------------------------------------------------
# CLASSIFICATION
# ------------------------------------------------------------------
else:
    try:
        models, scaler, feature_names, label_encoder, scores = load_classification_assets()
    except FileNotFoundError as e:
        st.error(f"Missing saved file: {e}. Run every cell in notebooks/classification.ipynb first, "
                 f"including the final 'save all models' cell.")
        st.stop()

    st.subheader("Classification — pick a model")
    st.dataframe(scores.sort_values("F1 (weighted)", ascending=False).reset_index(drop=True))

    model_name = st.selectbox("Algorithm", list(models.keys()))

    st.write("**Enter feature values**")
    user_input = {}
    cols = st.columns(3)
    for i, feat in enumerate(feature_names):
        with cols[i % 3]:
            user_input[feat] = st.number_input(feat, value=0.0, key=f"clf_{feat}")

    compare_all = st.checkbox("Also show every model's prediction on this input")

    if st.button("Predict", key="clf_predict"):
        input_df = pd.DataFrame([user_input])[feature_names]
        input_scaled = pd.DataFrame(scaler.transform(input_df), columns=feature_names)

        pred_encoded = models[model_name].predict(input_scaled)[0]
        pred_label = label_encoder.inverse_transform([int(pred_encoded)])[0]
        st.success(f"**{model_name}** predicts class: **{pred_label}**")

        if compare_all:
            rows = []
            for name, m in models.items():
                p_enc = m.predict(input_scaled)[0]
                p_label = label_encoder.inverse_transform([int(p_enc)])[0]
                rows.append({"Model": name, "Predicted class": p_label})
            st.write("**All models on this same input:**")
            st.dataframe(pd.DataFrame(rows).reset_index(drop=True))
