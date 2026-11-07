# Breast Cancer Wisconsin Diagnostic — ML Assignment 2

## M.Tech AIML | Machine Learning | Assignment 2

---

## Problem Statement

Breast cancer is one of the most prevalent cancers worldwide. Early and accurate diagnosis is critical for patient survival. This project builds a **binary classification system** to predict whether a breast tumour is **Malignant (M)** or **Benign (B)** using 30 numerical features extracted from digitised images of fine-needle aspirates (FNA) of breast masses.

The objective is to train, evaluate, and compare **five classical machine learning classifiers** using the Breast Cancer Wisconsin Diagnostic dataset, deploy an interactive Streamlit application, and document all findings for academic submission.

---

## Dataset Description

| Property           | Value                                 |
| ------------------ | ------------------------------------- |
| **Name**           | Breast Cancer Wisconsin (Diagnostic)  |
| **Source**         | UCI Machine Learning Repository       |
| **Records**        | 569                                   |
| **Features**       | 30 real-valued numeric features       |
| **Target Column**  | `diagnosis`                           |
| **Classes**        | M = Malignant (212), B = Benign (357) |
| **Missing Values** | None                                  |
| **Duplicates**     | None                                  |

**Features:** Ten nuclear measurements (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension) are computed for each cell nucleus across three statistics — **mean**, **standard error (SE)**, and **worst** (mean of the three largest values) — yielding 30 features total.

---

## GitHub Repository

> 🔗 **GitHub Repository URL:** `https://github.com/dev-patel27/ml_assingment_2`

---

## Project Structure

```
ml_assignment_2/
├── app.py                  # Streamlit web application
├── train.py                # Training script (runs all 5 models)
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── test_data.csv           # Held-out test set (114 samples)
├── data/
│   └── dataset.csv         # Breast Cancer Wisconsin dataset (569 records)
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    ├── label_encoder.joblib
    ├── feature_names.joblib
    └── model_comparison.csv
```

---

## Five Machine Learning Models

| #   | Model                      | Library Class            | Key Hyperparameters                               |
| --- | -------------------------- | ------------------------ | ------------------------------------------------- |
| 1   | **Logistic Regression**    | `LogisticRegression`     | `max_iter=1000, random_state=42`                  |
| 2   | **Decision Tree**          | `DecisionTreeClassifier` | `random_state=42`                                 |
| 3   | **K-Nearest Neighbors**    | `KNeighborsClassifier`   | `n_neighbors=5`                                   |
| 4   | **Gaussian Naive Bayes**   | `GaussianNB`             | default                                           |
| 5   | **Support Vector Machine** | `SVC`                    | `kernel='rbf', probability=True, random_state=42` |

Each model is wrapped in a **scikit-learn Pipeline** with `StandardScaler` to prevent data leakage. The scaler is fit only on training data and applied to test data during evaluation.

---

## Data Preprocessing Steps

1. **Load** CSV with named columns (ID, diagnosis, 30 features)
2. **Drop** `id` column (not a predictive feature)
3. **Check** missing values → 0 found
4. **Check** duplicates → 0 found
5. **Encode** target: `B → 0` (Benign), `M → 1` (Malignant)
6. **Split** features (X) and target (y)
7. **Stratified train/test split**: 80% train (455), 20% test (114), `random_state=42`
8. **Pipeline** with `StandardScaler` applied inside `Pipeline.fit()` on training data only → no leakage

---

## Evaluation Metrics — Model Comparison Table

> **Test Set:** 114 samples | 72 Benign · 42 Malignant | `random_state=42`

| Model                | Accuracy   | AUC        | Precision  | Recall | F1 Score   | MCC        |
| -------------------- | ---------- | ---------- | ---------- | ------ | ---------- | ---------- |
| Logistic Regression  | **0.9649** | **0.9960** | 0.9750     | 0.9286 | 0.9512     | 0.9245     |
| Decision Tree        | 0.9298     | 0.9246     | 0.9048     | 0.9048 | 0.9048     | 0.8492     |
| K-Nearest Neighbors  | 0.9561     | 0.9823     | 0.9744     | 0.9048 | 0.9383     | 0.9058     |
| Naive Bayes          | 0.9211     | 0.9891     | 0.9231     | 0.8571 | 0.8889     | 0.8292     |
| **SVM (RBF kernel)** | **0.9737** | **0.9947** | **1.0000** | 0.9286 | **0.9630** | **0.9442** |

> ✅ All metrics are computed on the held-out test set only. No data leakage. Models are never retrained on test data.

---

## Observations Per Model

### 1. Logistic Regression

- **Accuracy: 96.49%** — Second-highest accuracy, strong linear separator for this dataset.
- **AUC: 0.9960** — Highest AUC, indicating excellent discriminative ability across all thresholds.
- **Precision: 0.9750** — Very few false positives; reliable when predicting Malignant.
- **MCC: 0.9245** — Strong balanced performance even with class imbalance.
- **Strength:** Interpretable coefficients, fast training, generalises well due to regularisation.
- **Weakness:** Assumes linearity — slightly lower than SVM in accuracy.

### 2. Decision Tree

- **Accuracy: 92.98%** — Lowest accuracy among the five models.
- **AUC: 0.9246** — Lowest AUC; probability estimates from DT can be unreliable.
- **Precision = Recall = 0.9048** — Balanced but lower performance.
- **MCC: 0.8492** — Acceptable but weakest correlation coefficient.
- **Strength:** Fully interpretable (visualisable tree), no need for feature scaling.
- **Weakness:** Prone to overfitting; axis-aligned splits may miss curved boundaries.

### 3. K-Nearest Neighbors (KNN)

- **Accuracy: 95.61%** — Third-highest accuracy, solid non-parametric performance.
- **AUC: 0.9823** — Good probabilistic discrimination.
- **Precision: 0.9744** — Very few false positives.
- **MCC: 0.9058** — Good balanced score.
- **Strength:** Simple, no training phase, effective with 5 neighbours on this 30-feature dataset.
- **Weakness:** Slow prediction for large datasets; sensitive to irrelevant features.

### 4. Naive Bayes (Gaussian)

- **Accuracy: 92.11%** — Second-lowest accuracy.
- **AUC: 0.9891** — Surprisingly high AUC despite lower accuracy; probabilistic model works well.
- **Recall: 0.8571** — Lowest recall; misses 6 Malignant cases (False Negatives).
- **MCC: 0.8292** — Lowest correlation coefficient.
- **Strength:** Extremely fast, works with small training sets, handles continuous features well.
- **Weakness:** Independence assumption often violated in real data; FNs are costly in cancer diagnosis.

### 5. Support Vector Machine (RBF kernel)

- **Accuracy: 97.37%** — **Highest accuracy** on the test set.
- **AUC: 0.9947** — Near-perfect discriminative performance.
- **Precision: 1.0000** — **Zero false positives**: every patient predicted Malignant truly is Malignant.
- **F1 Score: 0.9630** — Best F1 among all five models.
- **MCC: 0.9442** — Highest balanced correlation.
- **Strength:** Excellent with high-dimensional data; RBF kernel captures non-linear boundaries; maximum-margin classifier.
- **Weakness:** Slower to train on large datasets; less interpretable than LR or DT; hyperparameter tuning needed.

---

## 🏆 Best Performing Model

> **Support Vector Machine (SVM with RBF kernel)**

SVM achieves the highest test accuracy (97.37%), the best F1 Score (0.9630), and the highest MCC (0.9442). Crucially, it achieves **Precision = 1.00**, meaning it makes **zero false positive predictions** — no healthy patient is incorrectly told they have cancer. While Logistic Regression achieves a slightly higher AUC (0.9960 vs 0.9947), SVM's superior accuracy, F1 and MCC make it the best overall model for clinical deployment.

---

## Installation Instructions

### Prerequisites

- Python 3.9 or higher
- pip

### 1. Clone the repository

```bash
git clone https://github.com/dev-patel27/ml_assingment_2.git
```

### 2. Create a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Execution Instructions

### Step 1 — Train all models

```bash
python train.py
```

This will:

- Load `data/dataset.csv`
- Preprocess the data (drop ID, encode labels, split 80/20)
- Train all five pipelines
- Save `.joblib` files in `model/`
- Save `test_data.csv`
- Print evaluation metrics and the comparison table

### Step 2 — Run the Streamlit application

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`. Upload `test_data.csv` and select a model to evaluate.

---

## Streamlit Deployment

> 🌐 **Live App URL:** `https://breast-cancer-classifier-a75uizyebedwoddz7i6cnc-ml-assingment-2.streamlit.app/`

---

## References

1. W.N. Street, W.H. Wolberg and O.L. Mangasarian. _Nuclear feature extraction for breast tumor diagnosis._ IS&T/SPIE 1993 International Symposium on Electronic Imaging, 1905, 861–870, 1993.
2. Scikit-learn: Machine Learning in Python — [sklearn.org](https://scikit-learn.org)
3. Streamlit Documentation — [docs.streamlit.io](https://docs.streamlit.io)
4. UCI ML Repository — [archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
