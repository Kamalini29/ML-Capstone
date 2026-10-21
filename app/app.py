"""
23CSE301 ML Capstone — Prediction GUI (all models, all 3 tracks)
================================================================
Run:  streamlit run app.py        (works from any folder)

Tracks: Regression (Oil Well) | Classification (Play Store Apps) | Clustering (Gene Expression)

Reads what the notebooks' final "save models" cell writes:
  models/<track>/manifest.json, <model>.joblib (one file per model), features.json, scores.csv,
  scaler.joblib (regression, classification), label_encoder.joblib (classification),
  demo_samples.csv (clustering)

One broken/unloadable model never takes the others down: each file is loaded separately and any
failure is listed on screen. A scikit-learn version mismatch is flagged at the top.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn
import streamlit as st

st.set_page_config(page_title="ML Capstone — Predictions", layout="wide")
st.title("ML Capstone — Predict with Any Model")

# Works whether models/ sits next to app/ (../models) or next to app.py (./models)
_here = Path(__file__).resolve().parent
MODELS_DIR = next((p for p in (_here.parent / "models", _here / "models") if p.exists()),
                  _here.parent / "models")


# scikit-learn version your OLD-format notebooks (regression / classification) saved their .pkl files with
# (printed by the `sklearn.__version__` cell at the end of those notebooks).
LEGACY_SKLEARN = "1.7.2"


def load_legacy(track):
    """Old layout: models/<track>_all_models.pkl, <track>_scaler.pkl, <track>_feature_names.pkl, <track>_model_scores.csv
    (+ classification_label_encoder.pkl). Everything lives in ONE dict pickle, so it is all-or-nothing."""
    r = MODELS_DIR
    A = {
        "manifest": {"sklearn": None}, "failed": {}, "demo": None, "extras": {},
        "models": joblib.load(r / f"{track}_all_models.pkl"),
        "scaler": joblib.load(r / f"{track}_scaler.pkl"),
        "features": list(joblib.load(r / f"{track}_feature_names.pkl")),
        "scores": pd.read_csv(r / f"{track}_model_scores.csv"),
    }
    if track == "classification":
        A["extras"]["label_encoder"] = joblib.load(r / "classification_label_encoder.pkl")
    return A


@st.cache_resource
def load_track(track):
    d = MODELS_DIR / track
    if not (d / "manifest.json").exists():          # no new-format folder -> use the old .pkl files
        return load_legacy(track)
    manifest = json.loads((d / "manifest.json").read_text())
    models, failed = {}, {}
    for name, fn in manifest["models"].items():
        try:
            models[name] = joblib.load(d / fn)
        except Exception as e:                     # e.g. pickle made under another sklearn version
            failed[name] = f"{type(e).__name__}: {str(e)[:120]}"
    A = {
        "manifest": manifest, "models": models, "failed": failed,
        "features": json.loads((d / "features.json").read_text()),
        "scores": pd.read_csv(d / "scores.csv"),
        "scaler": joblib.load(d / manifest["scaler"]) if "scaler" in manifest else None,
        "extras": {k: joblib.load(d / fn) for k, fn in manifest.get("extras", {}).items()},
        "demo": pd.read_csv(d / "demo_samples.csv", index_col=0) if (d / "demo_samples.csv").exists() else None,
    }
    return A


def feature_form(features, prefix):
    """Number inputs in 3 columns -> one-row DataFrame in training column order."""
    st.write("**Enter feature values**")
    vals, cols = {}, st.columns(3)
    for i, f in enumerate(features):
        with cols[i % 3]:
            vals[f] = st.number_input(f, value=0.0, key=f"{prefix}_{f}")
    return pd.DataFrame([vals])[features]


def scale(scaler, df, features):
    return pd.DataFrame(scaler.transform(df), columns=features)


TRACKS = {
    "Regression (Oil Well)": "regression",
    "Classification (Play Store Apps)": "classification",
    "Clustering (Gene Expression)": "clustering",
}
key = TRACKS[st.sidebar.radio("Track", list(TRACKS))]

try:
    A = load_track(key)
except FileNotFoundError as e:
    st.error(f"Missing saved file: {e}\n\nRun every cell in notebooks/{key}.ipynb (including the last "
             f"'save models' cell) and make sure the output folder is {MODELS_DIR / key}.")
    st.stop()

models, features, scores, scaler = A["models"], A["features"], A["scores"], A["scaler"]
saved_ver = A["manifest"].get("sklearn")
if saved_ver and saved_ver != sklearn.__version__:
    st.warning(f"These models were saved with scikit-learn {saved_ver}, but this app runs "
               f"{sklearn.__version__}. Pin `scikit-learn=={saved_ver}` in requirements.txt "
               f"(or re-run the notebook in this environment).")
if A["failed"]:
    st.error("These models could not be loaded and were skipped:\n\n" +
             "\n".join(f"- **{n}** — {why}" for n, why in A["failed"].items()))
if not models:
    st.stop()

# ------------------------------------------------------------------
# REGRESSION
# ------------------------------------------------------------------
if key == "regression":
    st.subheader("Regression — pick a model")
    st.dataframe(scores.sort_values("R2", ascending=False).reset_index(drop=True))
    model_name = st.selectbox("Algorithm", list(models))
    user_df = feature_form(features, "reg")
    compare_all = st.checkbox("Also show every model's prediction on this input")

    if st.button("Predict", key="reg_predict"):
        x = scale(scaler, user_df, features)
        if model_name.startswith("Polynomial"):
            st.caption("Note: polynomial models extrapolate poorly for inputs far outside the "
                       "training data's range — treat extreme predictions with caution.")
        st.success(f"**{model_name}** predicts: **{models[model_name].predict(x)[0]:.3f}**")
        if compare_all:
            rows = [{"Model": n, "Prediction": m.predict(x)[0]} for n, m in models.items()]
            st.write("**All models on this same input:**")
            st.dataframe(pd.DataFrame(rows).sort_values("Prediction").reset_index(drop=True))

# ------------------------------------------------------------------
# CLASSIFICATION
# ------------------------------------------------------------------
elif key == "classification":
    le = A["extras"]["label_encoder"]
    st.subheader("Classification — pick a model")
    st.dataframe(scores.sort_values("F1 (weighted)", ascending=False).reset_index(drop=True))
    model_name = st.selectbox("Algorithm", list(models))
    user_df = feature_form(features, "clf")
    compare_all = st.checkbox("Also show every model's prediction on this input")

    if st.button("Predict", key="clf_predict"):
        x = scale(scaler, user_df, features)
        label = le.inverse_transform([int(models[model_name].predict(x)[0])])[0]
        st.success(f"**{model_name}** predicts class: **{label}**")
        if compare_all:
            rows = [{"Model": n, "Predicted class": le.inverse_transform([int(m.predict(x)[0])])[0]}
                    for n, m in models.items()]
            st.write("**All models on this same input:**")
            st.dataframe(pd.DataFrame(rows).reset_index(drop=True))

# ------------------------------------------------------------------
# CLUSTERING  (each saved model is a pipeline: raw genes -> scaler -> PCA -> clusterer)
# ------------------------------------------------------------------
else:
    st.subheader("Clustering — assign tumour samples to clusters")
    st.dataframe(scores.reset_index(drop=True))
    st.caption("Agglomerative clustering cannot label new points by itself, so it is applied as a "
               "nearest-centroid assignment trained on its own cluster labels.")

    model_name = st.selectbox("Algorithm", list(models))
    demo = A["demo"]
    source = st.radio("Input", ["Demo samples from the dataset", "Upload CSV (rows = samples, columns = genes)"],
                      horizontal=True)

    rows, truth = None, None
    if source.startswith("Demo"):
        if demo is None:
            st.info("No demo_samples.csv found — upload a CSV instead.")
        else:
            picked = st.multiselect("Samples", list(demo.index), default=list(demo.index[:5]))
            rows = demo.loc[picked, features] if picked else None
            if rows is not None and "true_class" in demo.columns:
                truth = demo.loc[picked, "true_class"]
    else:
        up = st.file_uploader("CSV with a sample-id first column and one column per gene", type="csv")
        if up is not None:
            up_df = pd.read_csv(up, index_col=0)
            missing = [g for g in features if g not in up_df.columns]
            if missing:
                st.error(f"{len(missing)} of the {len(features)} genes the model needs are missing from "
                         f"your file (e.g. {missing[:3]}).")
            else:
                rows = up_df[features]

    compare_all = st.checkbox("Also show every model's cluster for these samples")
    if st.button("Assign cluster", key="clu_predict") and rows is not None and len(rows):
        out = pd.DataFrame({"Sample": rows.index, model_name: models[model_name].predict(rows)})
        if truth is not None:
            out["Known class"] = truth.values
        if compare_all:
            for n, m in models.items():
                if n != model_name:
                    out[n] = m.predict(rows)
        st.dataframe(out.reset_index(drop=True))
        st.caption("Cluster numbers are arbitrary per algorithm — cluster 0 in one model is not "
                   "the same group as cluster 0 in another.")