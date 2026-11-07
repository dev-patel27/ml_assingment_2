"""
app.py
======
M.Tech AIML – Machine Learning Assignment 2
Breast Cancer Wisconsin Diagnostic – Streamlit Application

Loads pre-trained model pipelines (saved by train.py) and lets the user:
  • Upload a CSV test file
  • Select a classifier
  • View evaluation metrics, confusion matrix, classification report
  • Download predictions

Run:  streamlit run app.py
"""

# ─────────────────────────── Imports ────────────────────────────────────────
import os
import io
import warnings
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

warnings.filterwarnings("ignore")


# ─────────────────────────── Page config ────────────────────────────────────
st.set_page_config(
    page_title="🩺 Breast Cancer Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────── Paths ──────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

MODEL_FILES = {
    "Logistic Regression" : "logistic_regression.joblib",
    "Decision Tree"       : "decision_tree.joblib",
    "K-Nearest Neighbors" : "knn.joblib",
    "Naive Bayes"         : "naive_bayes.joblib",
    "Support Vector Machine (SVM)" : "svm.joblib",
}


# ─────────────────────────── Helper: load model ─────────────────────────────
@st.cache_resource
def load_model(model_name: str):
    """Load a pre-trained pipeline from disk. Cached so it loads only once."""
    path = os.path.join(MODEL_DIR, MODEL_FILES[model_name])
    return joblib.load(path)


@st.cache_resource
def load_artifacts():
    """Load the LabelEncoder and expected feature names."""
    le   = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))
    feat = joblib.load(os.path.join(MODEL_DIR, "feature_names.joblib"))
    return le, feat


# ─────────────────────────── Helper: plot confusion matrix ──────────────────
def plot_confusion_matrix(cm: np.ndarray, labels=("Benign", "Malignant")):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, ax=ax
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


# ─────────────────────────── Helper: metric card ────────────────────────────
def metric_card(label: str, value: float, delta_color: str = "normal"):
    """Render a single metric tile."""
    st.metric(label=label, value=f"{value:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9a/"
        "Visual_Studio_Code_1.35_icon.svg/512px-Visual_Studio_Code_1.35_icon.svg.png",
        width=60,
    )
    st.title("🔬 Configuration")
    st.markdown("---")

    st.subheader("1️⃣  Select Classifier")
    selected_model_name = st.selectbox(
        "Choose a model:",
        list(MODEL_FILES.keys()),
        help="Select one of the five pre-trained classifiers."
    )

    st.markdown("---")
    st.subheader("2️⃣  Upload Test Data")
    uploaded_file = st.file_uploader(
        "Upload a CSV file (must contain the 30 feature columns + `diagnosis`)",
        type=["csv"],
        help=(
            "Use the generated `test_data.csv` or any file with the same "
            "30 feature columns. The model is never retrained on uploaded data."
        ),
    )

    st.markdown("---")
    st.info(
        "ℹ️ **No retraining** — uploaded data is used **only** for evaluation "
        "against pre-trained model weights.",
        icon="🔒",
    )
    st.markdown("---")
    st.caption("M.Tech AIML • ML Assignment 2 • Breast Cancer Wisconsin Diagnostic")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN HEADER & INTRODUCTION
# ═══════════════════════════════════════════════════════════════════════════
st.title("🩺 Breast Cancer Wisconsin Diagnostic Classifier")
st.markdown(
    """
    ### M.Tech AIML — Machine Learning Assignment 2

    This application evaluates **five pre-trained machine learning classifiers**
    on the Breast Cancer Wisconsin Diagnostic dataset.

    | | |
    |---|---|
    | **Dataset** | Breast Cancer Wisconsin Diagnostic (WDBC) |
    | **Records** | 569 (357 Benign · 212 Malignant) |
    | **Features** | 30 numeric features (mean, SE, worst for 10 nucleus measurements) |
    | **Target** | `diagnosis` — B = Benign (0) · M = Malignant (1) |
    | **Task** | Binary Classification |

    **Models available:** Logistic Regression · Decision Tree · KNN · Naive Bayes · SVM

    ---
    **How to use:**
    1. Select a classifier from the sidebar.
    2. Upload `test_data.csv` (generated by `train.py`).
    3. Explore metrics, confusion matrix, and predictions below.
    """
)

# ═══════════════════════════════════════════════════════════════════════════
# LOAD ARTIFACTS
# ═══════════════════════════════════════════════════════════════════════════
try:
    le, FEATURE_NAMES = load_artifacts()
except Exception as e:
    st.error(
        f"❌ Could not load model artefacts from `model/`.\n\n"
        f"Please run `python train.py` first to generate the saved models.\n\n"
        f"Error: {e}"
    )
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════
# PROCESS UPLOADED FILE
# ═══════════════════════════════════════════════════════════════════════════
if uploaded_file is None:
    st.info(
        "⬆️  Please upload a CSV file using the sidebar to begin evaluation.",
        icon="📂",
    )
    st.stop()

# ── Read the uploaded CSV ────────────────────────────────────────────────────
try:
    test_df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"❌ Could not read the uploaded file: {e}")
    st.stop()

# ── Validate that it has the diagnosis column ────────────────────────────────
if "diagnosis" not in test_df.columns:
    st.error(
        "❌ The uploaded CSV must contain a `diagnosis` column (B or M) "
        "so we can compare predictions against ground truth."
    )
    st.stop()

# ── Validate feature columns ─────────────────────────────────────────────────
missing_cols = [c for c in FEATURE_NAMES if c not in test_df.columns]
if missing_cols:
    st.error(
        f"❌ The uploaded CSV is missing {len(missing_cols)} required feature "
        f"column(s):\n\n`{missing_cols[:10]}`"
    )
    st.stop()

# ── Extract X and y ──────────────────────────────────────────────────────────
X_upload = test_df[FEATURE_NAMES]
y_true_str = test_df["diagnosis"].str.strip().str.upper()

# Encode labels (B→0, M→1)
try:
    y_true = le.transform(y_true_str)
except Exception:
    st.error(
        "❌ The `diagnosis` column contains unexpected values. "
        "Expected only 'B' (Benign) or 'M' (Malignant)."
    )
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════
# LOAD SELECTED MODEL & PREDICT
# ═══════════════════════════════════════════════════════════════════════════
try:
    pipeline = load_model(selected_model_name)
except Exception as e:
    st.error(
        f"❌ Could not load `{selected_model_name}` model.\n\n"
        f"Make sure `python train.py` has been executed.\n\nError: {e}"
    )
    st.stop()

y_pred      = pipeline.predict(X_upload)
y_pred_prob = pipeline.predict_proba(X_upload)[:, 1]

# ── Compute metrics ───────────────────────────────────────────────────────────
acc  = accuracy_score(y_true, y_pred)
auc  = roc_auc_score(y_true, y_pred_prob)
prec = precision_score(y_true, y_pred)
rec  = recall_score(y_true, y_pred)
f1   = f1_score(y_true, y_pred)
mcc  = matthews_corrcoef(y_true, y_pred)
cm   = confusion_matrix(y_true, y_pred)
cr   = classification_report(y_true, y_pred, target_names=["Benign", "Malignant"])

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 – MODEL & DATA SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.header(f"📊 Evaluation Results — {selected_model_name}")

col_info1, col_info2, col_info3 = st.columns(3)
with col_info1:
    st.metric("Model", selected_model_name)
with col_info2:
    st.metric("Uploaded Samples", len(test_df))
with col_info3:
    benign_n    = int((y_true == 0).sum())
    malignant_n = int((y_true == 1).sum())
    st.metric("Benign / Malignant", f"{benign_n} / {malignant_n}")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 – EVALUATION METRICS (6 tiles)
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("### 📈 Evaluation Metrics")

m1, m2, m3, m4, m5, m6 = st.columns(6)
with m1: metric_card("Accuracy",  acc)
with m2: metric_card("AUC Score", auc)
with m3: metric_card("Precision", prec)
with m4: metric_card("Recall",    rec)
with m5: metric_card("F1 Score",  f1)
with m6: metric_card("MCC",       mcc)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 – CONFUSION MATRIX
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("### 🔲 Confusion Matrix")

col_cm, col_cm_info = st.columns([1, 1])

with col_cm:
    fig_cm = plot_confusion_matrix(cm)
    st.pyplot(fig_cm)

with col_cm_info:
    tn, fp, fn, tp = cm.ravel()
    st.markdown(
        f"""
        | Metric | Value |
        |---|---|
        | **True Negatives (TN)** — Correctly predicted Benign | **{tn}** |
        | **True Positives (TP)** — Correctly predicted Malignant | **{tp}** |
        | **False Positives (FP)** — Benign predicted as Malignant | **{fp}** |
        | **False Negatives (FN)** — Malignant predicted as Benign | **{fn}** |
        """
    )
    st.markdown(
        "> ⚠️ In cancer screening, **False Negatives (FN)** are especially costly "
        "(a malignant tumour missed). High **Recall** minimises FN."
    )

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 – CLASSIFICATION REPORT
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("### 📋 Classification Report")
st.code(cr, language="text")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 – ACTUAL vs PREDICTED TABLE
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("### 🔍 Actual vs Predicted Results")

pred_labels = le.inverse_transform(y_pred)   # 0/1 → B/M
actual_df   = test_df[FEATURE_NAMES].copy()
actual_df.insert(0, "Actual Diagnosis",    y_true_str.values)
actual_df.insert(1, "Predicted Diagnosis", pred_labels)
actual_df.insert(2, "Probability (M)",     np.round(y_pred_prob, 4))
actual_df.insert(3, "Correct?",
                 actual_df["Actual Diagnosis"] == actual_df["Predicted Diagnosis"])

# Colour correct / incorrect rows
def highlight_correct(row):
    colour = "#d4edda" if row["Correct?"] else "#f8d7da"
    return [f"background-color: {colour}"] * len(row)

n_show = st.slider("Rows to preview:", 10, len(actual_df), min(50, len(actual_df)))
styled = actual_df.head(n_show).style.apply(highlight_correct, axis=1)
st.dataframe(styled, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6 – DOWNLOAD PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("### ⬇️ Download Predictions")

csv_buffer = io.StringIO()
actual_df.to_csv(csv_buffer, index=False)
csv_bytes = csv_buffer.getvalue().encode()

st.download_button(
    label="📥  Download predictions.csv",
    data=csv_bytes,
    file_name="predictions.csv",
    mime="text/csv",
    help="Download the full table of actual vs predicted results.",
)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7 – MODEL COMPARISON TABLE (from saved CSV)
# ═══════════════════════════════════════════════════════════════════════════
comparison_path = os.path.join(MODEL_DIR, "model_comparison.csv")
if os.path.exists(comparison_path):
    st.markdown("---")
    st.header("🏆 All Models Comparison (from Training Run)")
    comp_df = pd.read_csv(comparison_path)

    # Highlight best values in each metric column
    metric_cols = ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"]

    def highlight_max(s):
        is_max = s == s.max()
        return ["background-color: #d4edda; font-weight: bold" if v else "" for v in is_max]

    styled_comp = comp_df.style.apply(highlight_max, subset=metric_cols)
    st.dataframe(styled_comp, use_container_width=True)

    best_row = comp_df.loc[comp_df["Accuracy"].idxmax()]
    st.success(
        f"🏆 **Best model by Accuracy:** {best_row['Model']}  "
        f"(Accuracy={best_row['Accuracy']:.4f}, AUC={best_row['AUC']:.4f})"
    )

st.markdown("---")
st.caption(
    "M.Tech AIML · Machine Learning Assignment 2 · "
    "Breast Cancer Wisconsin Diagnostic · Built with Streamlit & Scikit-learn"
)

