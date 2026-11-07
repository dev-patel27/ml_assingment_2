"""
app.py
======
M.Tech AIML – Machine Learning Assignment 2
Breast Cancer Wisconsin Diagnostic – Streamlit Application

UI matches the Model Evaluation Dashboard reference design:
  • Sidebar: 1·Test data  2·Model  3·Display
  • Main: Performance metrics → Tabs (Performance / Confusion matrix /
          Classification report / Compare models)
  • Data-preview expander with row count

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
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report,
    roc_curve,
)

warnings.filterwarnings("ignore")

# ─────────────────────────── Page config ────────────────────────────────────
st.set_page_config(
    page_title="Breast Cancer Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────── Custom CSS ─────────────────────────────────────
st.markdown(
    """
    <style>
    /* metric tile cards */
    div[data-testid="metric-container"] {
        background-color: #1a1f2e;
        border: 1px solid #2d3748;
        border-radius: 8px;
        padding: 16px 20px 12px 20px;
    }
    div[data-testid="metric-container"] label {
        font-size: 0.75rem !important;
        color: #a0aec0 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        font-size: 1.9rem !important;
        font-weight: 700 !important;
        color: #f7fafc !important;
    }
    /* tab styling */
    button[data-baseweb="tab"] {
        font-size: 0.85rem !important;
        padding: 6px 16px !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #e53e3e !important;
        border-bottom: 2px solid #e53e3e !important;
    }
    /* info bar */
    .info-bar {
        font-size: 0.82rem;
        color: #718096;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────── Paths ──────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR     = os.path.join(BASE_DIR, "model")
DATA_DIR      = os.path.join(BASE_DIR, "data")
DATASET_PATH  = os.path.join(DATA_DIR,  "dataset.csv")
TEST_CSV_PATH = os.path.join(BASE_DIR,  "test_data.csv")

MODEL_FILES = {
    "Logistic Regression"          : "logistic_regression.joblib",
    "Decision Tree"                : "decision_tree.joblib",
    "K-Nearest Neighbors"          : "knn.joblib",
    "Naive Bayes"                  : "naive_bayes.joblib",
    "Support Vector Machine (SVM)" : "svm.joblib",
}


# ─────────────────────────── Cached loaders ─────────────────────────────────
@st.cache_resource
def load_model(name: str):
    return joblib.load(os.path.join(MODEL_DIR, MODEL_FILES[name]))


@st.cache_resource
def load_artifacts():
    le   = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))
    feat = joblib.load(os.path.join(MODEL_DIR, "feature_names.joblib"))
    return le, feat


@st.cache_data
def read_test_csv():
    if os.path.exists(TEST_CSV_PATH):
        return pd.read_csv(TEST_CSV_PATH)
    return None


@st.cache_data
def read_dataset_bytes():
    if os.path.exists(DATASET_PATH):
        with open(DATASET_PATH, "rb") as f:
            return f.read()
    return None


@st.cache_data
def read_test_csv_bytes():
    if os.path.exists(TEST_CSV_PATH):
        with open(TEST_CSV_PATH, "rb") as f:
            return f.read()
    return None


# ─────────────────────────── Plot helpers ───────────────────────────────────
def plot_confusion_matrix(cm, labels=("Benign", "Malignant")):
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, ax=ax,
        annot_kws={"size": 14, "weight": "bold"},
    )
    ax.set_xlabel("Predicted Label", color="#a0aec0", fontsize=11)
    ax.set_ylabel("True Label",      color="#a0aec0", fontsize=11)
    ax.tick_params(colors="#a0aec0")
    ax.set_title("Confusion Matrix", color="#f7fafc", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_roc(y_true, y_prob, model_name, auc_val):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#151921")
    ax.plot(fpr, tpr, color="#3182ce", lw=2.5,
            label=f"Model (AUC {auc_val:.4f})")
    ax.plot([0, 1], [0, 1], color="#718096", linestyle="--", lw=1.2,
            label="Random baseline")
    ax.fill_between(fpr, tpr, alpha=0.10, color="#3182ce")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False positive rate", color="#a0aec0", fontsize=11)
    ax.set_ylabel("True positive rate",  color="#a0aec0", fontsize=11)
    ax.set_title("Receiver Operating Characteristic", color="#f7fafc",
                 fontsize=13, fontweight="bold")
    ax.tick_params(colors="#718096")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2d3748")
    legend = ax.legend(loc="lower right", fontsize=10,
                       facecolor="#1a1f2e", edgecolor="#2d3748",
                       labelcolor="#f7fafc")
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🔬 Configuration")
    st.markdown("---")

    # ── 1 · Test data ────────────────────────────────────────────────────────
    st.markdown("### 1 · Test data")
    st.caption("Upload a labelled CSV with the required model features and target column.")

    uploaded_file = st.file_uploader(
        "Upload test CSV",
        type=["csv"],
        label_visibility="collapsed",
    )

    # Toggle: use bundled test_data.csv
    use_bundled = st.toggle(
        "No file? Use the bundled `test_data.csv`",
        value=(uploaded_file is None),
        disabled=(uploaded_file is not None),
    )

    # Download test_data.csv
    test_bytes = read_test_csv_bytes()
    if test_bytes:
        st.download_button(
            label="⬇️  Download test_data.csv",
            data=test_bytes,
            file_name="test_data.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("---")

    # ── 2 · Model ────────────────────────────────────────────────────────────
    st.markdown("### 2 · Model")
    selected_model = st.selectbox(
        "Choose a classifier",
        list(MODEL_FILES.keys()),
    )

    st.markdown("---")

    # ── 3 · Display ──────────────────────────────────────────────────────────
    st.markdown("### 3 · Display")
    show_roc     = st.checkbox("Show ROC curve",    value=True)
    show_preview = st.checkbox("Show data preview", value=True)

    preview_rows = 10
    if show_preview:
        preview_rows = st.slider("Preview rows", 5, 114, 10, step=5)

    with st.expander("Expected columns"):
        try:
            _le, _feat = load_artifacts()
            st.markdown(
                "The `diagnosis` column (`B` / `M`) is required. "
                "An `id` column is accepted but ignored (leakage prevention)."
            )
            st.code(", ".join(_feat), language="text")
        except Exception:
            st.warning("Run `python train.py` first.")

    st.markdown("---")
    st.caption("M.Tech AIML • ML Assignment 2 • Breast Cancer Wisconsin Diagnostic")


# ═══════════════════════════════════════════════════════════════════════════
# LOAD ARTIFACTS
# ═══════════════════════════════════════════════════════════════════════════
try:
    le, FEATURE_NAMES = load_artifacts()
except Exception as e:
    st.error(f"❌ Could not load model artefacts — run `python train.py` first.\n\n{e}")
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════
# RESOLVE DATA SOURCE
# ═══════════════════════════════════════════════════════════════════════════
source_label = ""

if uploaded_file is not None:
    try:
        test_df      = pd.read_csv(uploaded_file)
        source_label = uploaded_file.name
    except Exception as e:
        st.error(f"❌ Could not read uploaded file: {e}")
        st.stop()
elif use_bundled:
    bundled = read_test_csv()
    if bundled is None:
        st.error("❌ Bundled `test_data.csv` not found. Run `python train.py` first.")
        st.stop()
    test_df      = bundled
    source_label = "test_data.csv"
else:
    st.info("⬆️  Upload a CSV file or toggle **'Use the bundled test_data.csv'** to begin.", icon="📂")
    st.stop()

# ── Validate columns ─────────────────────────────────────────────────────────
if "diagnosis" not in test_df.columns:
    st.error("❌ Missing `diagnosis` column (B / M).")
    st.stop()

missing_cols = [c for c in FEATURE_NAMES if c not in test_df.columns]
if missing_cols:
    st.error(f"❌ Missing {len(missing_cols)} feature column(s): `{missing_cols[:5]}`…")
    st.stop()

# ── Extract X and y ──────────────────────────────────────────────────────────
X         = test_df[FEATURE_NAMES]
y_str     = test_df["diagnosis"].str.strip().str.upper()
try:
    y_true = le.transform(y_str)
except Exception:
    st.error("❌ `diagnosis` column must contain only 'B' or 'M'.")
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════
# LOAD MODEL & PREDICT
# ═══════════════════════════════════════════════════════════════════════════
try:
    pipeline = load_model(selected_model)
except Exception as e:
    st.error(f"❌ Could not load model — run `python train.py` first.\n\n{e}")
    st.stop()

y_pred      = pipeline.predict(X)
y_pred_prob = pipeline.predict_proba(X)[:, 1]

# ── Metrics ───────────────────────────────────────────────────────────────────
acc  = accuracy_score(y_true, y_pred)
auc  = roc_auc_score(y_true, y_pred_prob)
prec = precision_score(y_true, y_pred, zero_division=0)
rec  = recall_score(y_true, y_pred, zero_division=0)
f1   = f1_score(y_true, y_pred, zero_division=0)
mcc  = matthews_corrcoef(y_true, y_pred)
cm   = confusion_matrix(y_true, y_pred)
cr   = classification_report(y_true, y_pred, target_names=["Benign", "Malignant"])

n_total      = len(y_true)
n_malignant  = int(y_pred.sum())
pct_maligant = f"{n_malignant / n_total * 100:.1f}%"


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — HEADER
# ═══════════════════════════════════════════════════════════════════════════
st.title("Model Evaluation Dashboard")
st.markdown(
    "Evaluate saved classifiers on labelled test data — "
    "without retraining or fitting anything to your upload."
)
st.markdown(
    f'<p class="info-bar">Data source: <b>{source_label}</b> &nbsp;·&nbsp; '
    f'Rows: <b>{n_total:,}</b> &nbsp;·&nbsp; '
    f'Model: <b>{selected_model}</b></p>',
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════
# DATA PREVIEW EXPANDER
# ═══════════════════════════════════════════════════════════════════════════
if show_preview:
    with st.expander(
        f"Data preview · first {preview_rows} of {n_total:,} rows",
        expanded=True,
    ):
        st.dataframe(test_df.head(preview_rows), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PERFORMANCE METRIC TILES  (row 1: 6 metrics | row 2: 3 summary stats)
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("#### Performance metrics")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Accuracy",  f"{acc:.4f}")
c2.metric("AUC",       f"{auc:.4f}")
c3.metric("Precision", f"{prec:.4f}")
c4.metric("Recall",    f"{rec:.4f}")
c5.metric("F1",        f"{f1:.4f}")
c6.metric("MCC",       f"{mcc:.4f}")

s1, s2, s3 = st.columns(3)
s1.metric("Test records",           f"{n_total:,}")
s2.metric("Predicted Malignant",    f"{n_malignant:,}")
s3.metric("Predicted Malignant rate", pct_maligant)


# ═══════════════════════════════════════════════════════════════════════════
# TABS  —  Performance | Confusion matrix | Classification report | Compare models
# ═══════════════════════════════════════════════════════════════════════════
tab_perf, tab_cm, tab_cr, tab_cmp = st.tabs([
    "Performance",
    "Confusion matrix",
    "Classification report",
    "Compare models",
])


# ── Tab 1: Performance (ROC curve) ───────────────────────────────────────────
with tab_perf:
    if show_roc:
        col_roc, col_roc_desc = st.columns([3, 2])
        with col_roc:
            st.pyplot(plot_roc(y_true, y_pred_prob, selected_model, auc))
        with col_roc_desc:
            st.markdown("#### About the ROC curve")
            st.markdown(
                f"""
                The **Receiver Operating Characteristic (ROC)** curve plots
                **True Positive Rate** (Recall) vs **False Positive Rate**
                at every classification threshold.

                | | |
                |---|---|
                | **AUC = 1.00** | Perfect classifier |
                | **AUC = 0.50** | Random guessing |
                | **This model** | **{auc:.4f}** |

                A higher AUC means the model separates Malignant from Benign
                across all decision thresholds — not only at the default 0.5 cutoff.

                > ⚠️ In cancer screening, **False Negatives** (Malignant missed)
                are especially costly. High **Recall** minimises them.
                """
            )
    else:
        st.info("Enable **Show ROC curve** in the sidebar → Display section.", icon="📉")

    st.markdown("---")
    st.markdown("#### Actual vs Predicted results")

    pred_labels = le.inverse_transform(y_pred)
    result_df   = pd.DataFrame({
        "Actual Diagnosis"    : y_str.values,
        "Predicted Diagnosis" : pred_labels,
        "Probability (M)"    : np.round(y_pred_prob, 4),
        "Correct?"           : y_str.values == pred_labels,
    })

    def highlight_correct(row):
        c = "#1a3a2a" if row["Correct?"] else "#3a1a1a"
        return [f"background-color: {c}"] * len(row)

    n_show = min(preview_rows, len(result_df)) if show_preview else len(result_df)
    styled = result_df.head(n_show).style.apply(highlight_correct, axis=1)
    st.dataframe(styled, use_container_width=True)

    # Download predictions
    buf = io.StringIO()
    result_df.to_csv(buf, index=False)
    st.download_button(
        label="⬇️  Download predictions.csv",
        data=buf.getvalue().encode(),
        file_name="predictions.csv",
        mime="text/csv",
    )


# ── Tab 2: Confusion matrix ───────────────────────────────────────────────────
with tab_cm:
    col_fig, col_tbl = st.columns([1, 1])
    tn, fp, fn, tp = cm.ravel()

    with col_fig:
        st.pyplot(plot_confusion_matrix(cm))

    with col_tbl:
        st.markdown("#### Confusion matrix breakdown")
        st.markdown(
            f"""
            | | Value | Meaning |
            |---|---|---|
            | **True Negative (TN)** | **{tn}** | Correctly predicted Benign |
            | **True Positive (TP)** | **{tp}** | Correctly predicted Malignant |
            | **False Positive (FP)** | **{fp}** | Benign predicted as Malignant |
            | **False Negative (FN)** | **{fn}** | Malignant predicted as Benign ⚠️ |
            """
        )
        st.markdown(
            "> **False Negatives are most dangerous** in oncology — a missed "
            "malignant diagnosis delays treatment. Prioritise high **Recall** to minimise FN."
        )
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        st.markdown(
            f"**Sensitivity (Recall):** {sensitivity:.4f} &nbsp;|&nbsp; "
            f"**Specificity:** {specificity:.4f}",
            unsafe_allow_html=True,
        )


# ── Tab 3: Classification report ─────────────────────────────────────────────
with tab_cr:
    st.markdown("#### Classification report")
    st.code(cr, language="text")
    st.caption(
        "Macro avg = unweighted mean across classes. "
        "Weighted avg = mean weighted by class support (sample count)."
    )


# ── Tab 4: Compare models ────────────────────────────────────────────────────
with tab_cmp:
    comparison_path = os.path.join(MODEL_DIR, "model_comparison.csv")
    if os.path.exists(comparison_path):
        comp_df     = pd.read_csv(comparison_path)
        metric_cols = ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"]

        st.markdown("#### All models — test-set comparison")
        st.caption("Green highlight = best value in each metric column.")

        def highlight_max(s):
            return [
                "background-color: #1a3a2a; font-weight: bold" if v == s.max() else ""
                for v in s
            ]

        # Also highlight the currently selected model row
        def highlight_selected(row):
            model_col = row["Model"].lower().replace(" ", "_")
            sel_key   = selected_model.lower().replace(" (svm)", "").replace("-", "_").replace(" ", "_")
            if model_col == sel_key or sel_key in model_col:
                return ["border-left: 3px solid #e53e3e"] * len(row)
            return [""] * len(row)

        styled_comp = (
            comp_df.style
            .apply(highlight_max, subset=metric_cols)
        )
        st.dataframe(styled_comp, use_container_width=True)

        best = comp_df.loc[comp_df["Accuracy"].idxmax()]
        st.success(
            f"🏆 **Best overall:** {best['Model']} — "
            f"Accuracy {best['Accuracy']:.4f} · AUC {best['AUC']:.4f} · "
            f"F1 {best['F1 Score']:.4f} · MCC {best['MCC']:.4f}"
        )

        # Bar chart comparison
        st.markdown("#### Metric comparison chart")
        fig_bar, ax_bar = plt.subplots(figsize=(9, 4))
        fig_bar.patch.set_facecolor("#0e1117")
        ax_bar.set_facecolor("#151921")

        x     = np.arange(len(comp_df))
        width = 0.13
        colors = ["#3182ce", "#38a169", "#d69e2e", "#e53e3e", "#805ad5", "#319795"]

        for i, col in enumerate(metric_cols):
            bars = ax_bar.bar(
                x + i * width, comp_df[col], width,
                label=col, color=colors[i], alpha=0.85
            )

        ax_bar.set_xticks(x + width * 2.5)
        ax_bar.set_xticklabels(comp_df["Model"], color="#a0aec0", fontsize=9, rotation=10)
        ax_bar.set_ylim(0, 1.09)
        ax_bar.set_ylabel("Score", color="#a0aec0", fontsize=10)
        ax_bar.set_title("Model Metrics Comparison", color="#f7fafc",
                         fontsize=12, fontweight="bold")
        ax_bar.tick_params(axis="y", colors="#718096")
        for spine in ax_bar.spines.values():
            spine.set_edgecolor("#2d3748")
        ax_bar.legend(
            fontsize=8, facecolor="#1a1f2e",
            edgecolor="#2d3748", labelcolor="#f7fafc",
            ncol=3, loc="upper right",
        )
        plt.tight_layout()
        st.pyplot(fig_bar)

    else:
        st.warning("No `model/model_comparison.csv` found — run `python train.py` first.")


# ─────────────────────────── Footer ─────────────────────────────────────────
st.markdown("---")
st.caption(
    "M.Tech AIML · Machine Learning Assignment 2 · "
    "Breast Cancer Wisconsin Diagnostic · Built with Streamlit & Scikit-learn"
)
