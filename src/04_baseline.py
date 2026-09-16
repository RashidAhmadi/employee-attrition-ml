#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 9:14:14 2026

@author: Rashid Ahmadi
"""

# %%

"""
Employee Attrition Prediction
03 - Baseline Models

Portfolio Project
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# %%

# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "employee_attrition_clean.csv"
)

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# %%

# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(PROCESSED_PATH)

print("=" * 60)
print("DATASET")
print("=" * 60)

print("Shape:", df.shape)
print()


# %%
# ============================================================
# 3. DEFINE FEATURES AND TARGET
# ============================================================


X=df.drop(columns=["Attrition"])
y=df["Attrition"].map({
    "No":0,
    "Yes":1
    })


print("\nTarget distribution: ")
print(y.value_counts())

print("\nTarget percentages:")
print(
    y.value_counts(normalize=True)
    .mul(100)
    .round(2)
)
# %%


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================


X_train, X_test, y_train, y_test=train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
    )


print("\nTraining set:", X_train.shape)
print("Test set:", X_test.shape)


# %%

# ============================================================
# 5. Identify numerical and categorical features
# ============================================================

numerical_columns=X_train.select_dtypes(
    include="number"
    ).columns.tolist()

categorical_columns=X_train.select_dtypes(
    include="object"
    ).columns.tolist()


print("\nNumber of numerical features: ", len(numerical_columns))
print("Number of categorical features :", len(categorical_columns))

# %%

# ============================================================
# 6. Preprocessing
# ============================================================

numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(
    handle_unknown="ignore"
    )


preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numerical_transformer,
            numerical_columns
            ),
        (
            "cat",
            categorical_transformer,
            categorical_columns
            )
        ]
    )

# %%

# ============================================================
# 7. Logistic Regression model
# ============================================================

logistic_model=LogisticRegression(
    max_iter=1000,
    random_state=42
    )



# ============================================================
# 8. Complete Pipeline
# ============================================================

baseline_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
            ),
        (
            "classifier",
            logistic_model
            ),
        
        ]
    )
# %%

# ============================================================
# 9. Cross-validation on training data
# ============================================================

cv_scores=cross_val_score(
    baseline_pipeline,
    X_train,
    y_train,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
    )

print("\n" + "=" * 60)
print("CROSS-VALIDATION")
print("=" * 60)

print("ROC-AUC scores:", cv_scores)
print("Mean ROC-AUC:", round(cv_scores.mean(), 4))
print("Std ROC-AUC:", round(cv_scores.std(), 4))

# %%
# ============================================================
# 10. Train final baseline model
# ============================================================

baseline_pipeline.fit(
    X_train,
    y_train
)
# %%

# ============================================================
# 11. Predictions
# ============================================================

y_pred = baseline_pipeline.predict(X_test)

y_proba = baseline_pipeline.predict_proba(X_test)[:, 1]

# %%
# ============================================================
# 12. Evaluation metrics
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    pos_label=1
)

recall = recall_score(
    y_test,
    y_pred,
    pos_label=1
)

f1 = f1_score(
    y_test,
    y_pred,
    pos_label=1
)

roc_auc = roc_auc_score(
    y_test,
    y_proba
)

pr_auc = average_precision_score(
    (y_test == "Yes").astype(int),
    y_proba
)


print("\n" + "=" * 60)
print("LOGISTIC REGRESSION BASELINE")
print("=" * 60)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")
print(f"PR-AUC:    {pr_auc:.4f}")
# %%

# ============================================================
# 13. Confusion Matrix
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

print("\nConfusion Matrix:")
print(cm)


# %%
# ============================================================
#  14. Plot Confusion Matrix
# ============================================================
plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No", "Yes"],
    yticklabels=["No", "Yes"]
)

plt.title("Logistic Regression - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

confusion_matrix_path = (
    FIGURES_DIR
    / "baseline_confusion_matrix.png"
)

plt.savefig(
    confusion_matrix_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()


# %% 
# ============================================================
#  15. Save baseline results
# ============================================================

baseline_results = pd.DataFrame({
    "Model": ["Logistic Regression"],
    "Accuracy": [accuracy],
    "Precision": [precision],
    "Recall": [recall],
    "F1": [f1],
    "ROC_AUC": [roc_auc],
    "PR_AUC": [pr_auc],
    "CV_ROC_AUC_Mean": [cv_scores.mean()],
    "CV_ROC_AUC_STD": [cv_scores.std()],
})


results_path = (
    PROJECT_ROOT
    / "reports" / "results"
    / "baseline_results.csv"
)

baseline_results.to_csv(
    results_path,
    index=False
)


print("\n" + "=" * 60)
print("FILES SAVED")
print("=" * 60)

print("Confusion matrix:")
print(confusion_matrix_path)

print("\nBaseline results:")
print(results_path)








