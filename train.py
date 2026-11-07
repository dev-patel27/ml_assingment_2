"""
train.py
========
M.Tech AIML - Machine Learning Assignment 2
Breast Cancer Wisconsin Diagnostic Dataset

This script:
  1. Loads and preprocesses the dataset
  2. Trains five classification models
  3. Evaluates each model on held-out test data
  4. Saves trained pipelines to the model/ directory
  5. Saves test_data.csv for use in the Streamlit app

Run:  python train.py
"""

# ─────────────────────────── Imports ────────────────────────────────────────
import os
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

warnings.filterwarnings("ignore")

# ─────────────────────────── Paths ──────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "data", "dataset.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────── Feature column names ───────────────────────────
# The raw wdbc.data file has no header row.
# Column 0 = ID, Column 1 = diagnosis, Columns 2-31 = 30 numeric features.
FEATURE_NAMES = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean",
    "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se",
    "concave_points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
    "smoothness_worst", "compactness_worst", "concavity_worst",
    "concave_points_worst", "symmetry_worst", "fractal_dimension_worst",
]
COLUMNS = ["id", "diagnosis"] + FEATURE_NAMES


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 – LOAD & INSPECT THE DATASET
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("STEP 1 – LOADING AND INSPECTING THE DATASET")
print("=" * 65)

df = pd.read_csv(DATA_PATH, header=None, names=COLUMNS)

print(f"\nDataset shape : {df.shape}")
print(f"Columns       : {list(df.columns)}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nDuplicate rows: {df.duplicated().sum()}")
print(f"\nClass distribution (diagnosis):\n{df['diagnosis'].value_counts()}")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 – DATA PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 2 – DATA PREPROCESSING")
print("=" * 65)

# 2a – Drop the ID column (not a predictive feature)
df.drop(columns=["id"], inplace=True)
print("\n✔ Dropped 'id' column.")

# 2b – Handle missing values (none expected, but defensive)
if df.isnull().sum().sum() > 0:
    df.dropna(inplace=True)
    print("✔ Dropped rows with missing values.")
else:
    print("✔ No missing values found.")

# 2c – Handle duplicates
n_dup = df.duplicated().sum()
if n_dup > 0:
    df.drop_duplicates(inplace=True)
    print(f"✔ Removed {n_dup} duplicate rows.")
else:
    print("✔ No duplicate rows found.")

# 2d – Encode the target label  (M → 1, B → 0)
le = LabelEncoder()
df["diagnosis"] = le.fit_transform(df["diagnosis"])   # B=0, M=1
print(f"\n✔ Label encoded: {dict(zip(le.classes_, le.transform(le.classes_)))}")
print("   (B=0 Benign, M=1 Malignant)")

# 2e – Separate features (X) and target (y)
X = df[FEATURE_NAMES]
y = df["diagnosis"]
print(f"\n✔ Features shape : {X.shape}")
print(f"✔ Target shape   : {y.shape}")

# 2f – Stratified train / test split  (80 / 20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=42
)
print(f"\n✔ Training set  : {X_train.shape[0]} samples")
print(f"✔ Test set      : {X_test.shape[0]} samples")
print(f"  Train class balance: {y_train.value_counts().to_dict()}")
print(f"  Test  class balance: {y_test.value_counts().to_dict()}")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 – DEFINE MODELS WITH PREPROCESSING PIPELINES
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 3 – DEFINING PIPELINES")
print("=" * 65)

# Each pipeline:  StandardScaler → Classifier
# StandardScaler is fitted only on X_train (inside the pipeline)
# → prevents data leakage into X_test.

pipelines = {
    "logistic_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ]),
    "decision_tree": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", DecisionTreeClassifier(random_state=42))
    ]),
    "knn": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=5))
    ]),
    "naive_bayes": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GaussianNB())
    ]),
    "svm": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True, random_state=42))
    ]),
}

print("\n✔ Pipelines defined (StandardScaler + Classifier) for:")
for name in pipelines:
    print(f"   • {name}")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4 – TRAIN, EVALUATE AND SAVE EACH MODEL
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 4 – TRAINING, EVALUATION AND SAVING")
print("=" * 65)

results = {}   # will store metrics for each model

for model_name, pipeline in pipelines.items():
    print(f"\n{'─'*55}")
    print(f" Model : {model_name.upper()}")
    print(f"{'─'*55}")

    # ── Train ────────────────────────────────────────────────────────────
    pipeline.fit(X_train, y_train)

    # ── Predict ──────────────────────────────────────────────────────────
    y_pred      = pipeline.predict(X_test)
    y_pred_prob = pipeline.predict_proba(X_test)[:, 1]   # prob of class 1 (M)

    # ── Metrics ──────────────────────────────────────────────────────────
    acc  = accuracy_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_pred_prob)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    mcc  = matthews_corrcoef(y_test, y_pred)
    cm   = confusion_matrix(y_test, y_pred)
    cr   = classification_report(y_test, y_pred, target_names=["Benign", "Malignant"])

    results[model_name] = {
        "Accuracy"  : round(acc,  4),
        "AUC"       : round(auc,  4),
        "Precision" : round(prec, 4),
        "Recall"    : round(rec,  4),
        "F1 Score"  : round(f1,   4),
        "MCC"       : round(mcc,  4),
    }

    print(f" Accuracy  : {acc:.4f}")
    print(f" AUC       : {auc:.4f}")
    print(f" Precision : {prec:.4f}")
    print(f" Recall    : {rec:.4f}")
    print(f" F1 Score  : {f1:.4f}")
    print(f" MCC       : {mcc:.4f}")
    print(f"\n Confusion Matrix:\n{cm}")
    print(f"\n Classification Report:\n{cr}")

    # ── Save pipeline ─────────────────────────────────────────────────────
    model_path = os.path.join(MODEL_DIR, f"{model_name}.joblib")
    joblib.dump(pipeline, model_path)
    print(f" ✔ Saved → {model_path}")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5 – RESULTS COMPARISON TABLE
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 5 – MODEL COMPARISON TABLE")
print("=" * 65)

results_df = pd.DataFrame(results).T.reset_index()
results_df.rename(columns={"index": "Model"}, inplace=True)
results_df["Model"] = results_df["Model"].str.replace("_", " ").str.title()

print("\n" + results_df.to_string(index=False))

best_model_name = results_df.loc[results_df["Accuracy"].idxmax(), "Model"]
best_model_auc  = results_df.loc[results_df["Accuracy"].idxmax(), "AUC"]
print(f"\n🏆 Best model by Accuracy : {best_model_name}  (AUC={best_model_auc})")

# Save comparison table as CSV alongside the models
comparison_path = os.path.join(MODEL_DIR, "model_comparison.csv")
results_df.to_csv(comparison_path, index=False)
print(f"\n✔ Comparison table saved → {comparison_path}")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 6 – SAVE TEST DATA  (for the Streamlit app)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 6 – SAVING TEST DATA")
print("=" * 65)

test_df = X_test.copy()
# Store original string labels (B / M) so the app can display them
test_df["diagnosis"] = le.inverse_transform(y_test.values)
test_data_path = os.path.join(BASE_DIR, "test_data.csv")
test_df.to_csv(test_data_path, index=False)
print(f"\n✔ test_data.csv saved → {test_data_path}")
print(f"  Shape : {test_df.shape}")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 7 – SAVE LABEL ENCODER  (needed in Streamlit app)
# ═══════════════════════════════════════════════════════════════════════════
le_path = os.path.join(MODEL_DIR, "label_encoder.joblib")
joblib.dump(le, le_path)
print(f"\n✔ LabelEncoder saved → {le_path}")

# Also save feature names list for validation in the app
feature_names_path = os.path.join(MODEL_DIR, "feature_names.joblib")
joblib.dump(FEATURE_NAMES, feature_names_path)
print(f"✔ Feature names saved → {feature_names_path}")

print("\n" + "=" * 65)
print("✅  TRAINING COMPLETE — all artefacts saved to model/")
print("=" * 65)

