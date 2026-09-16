#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 13:25:39 2026

@author: Rashid Ahmadi
"""

# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

from sklearn.model_selection import cross_val_predict

# %%
# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "employee_attrition_clean.csv"
)

FIGURES_DIR = (
    PROJECT_ROOT
    / "reports"
    / "figures"
)

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
# %%
# ============================================================
# 2.LOAD THE DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["Attrition"])
y = df["Attrition"].map(
    {
     "No":0,
     "Yes":1
     }
    )

# %%
# ============================================================
# 4. Train test split
# ============================================================

X_train, X_test, y_train, y_test=train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
    )

# %%
# ============================================================
# 5. Preprocessing
# ============================================================

numeric_columns = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_columns = X_train.select_dtypes(
    include=["object"]
).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numeric_columns
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ]
)

# %%
# ============================================================
# 6. Define the tuned models
# ============================================================
# a. Logistic Regression from Grid Search:

logistic_model = LogisticRegression(
    C=100,
    solver="liblinear",
    max_iter=1000,
    random_state=42
)

# b. Random Forest from Grid Search:
    
random_forest_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=10,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

# c. GBoost useing the PyHopper result

xgboost_model = XGBClassifier(
    n_estimators=386,
    max_depth=9,
    learning_rate=0.2,
    subsample=0.6128557157692793,
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1
)

# %%

# ============================================================
# 7. Build the pipelines
# ============================================================

logistic_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", logistic_model)
    ]
)

random_forest_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", random_forest_model)
    ]
)

xgboost_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", xgboost_model)
    ]
)
# %%
# ============================================================
# 7. Train the model
# ============================================================
logistic_pipeline.fit(X_train, y_train)

random_forest_pipeline.fit(X_train, y_train)

xgboost_pipeline.fit(X_train, y_train)

# %% 
# ============================================================
# 7. Generate predictions
# ============================================================
logistic_pred = logistic_pipeline.predict(X_test)
logistic_proba = logistic_pipeline.predict_proba(X_test)[:, 1]

random_forest_pred = random_forest_pipeline.predict(X_test)
random_forest_proba = random_forest_pipeline.predict_proba(X_test)[:, 1]

xgboost_pred = xgboost_pipeline.predict(X_test)
xgboost_proba = xgboost_pipeline.predict_proba(X_test)[:, 1]

# %% 
# ============================================================
# 8. Evaluate all three models
# ============================================================

def evaluate_model(model_name, y_true, y_pred, y_proba):

    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true,
            y_pred,
            pos_label=1
        ),
        "Recall": recall_score(
            y_true,
            y_pred,
            pos_label=1
        ),
        "F1": f1_score(
            y_true,
            y_pred,
            pos_label=1
        ),
        "ROC_AUC": roc_auc_score(
            y_true,
            y_proba
        ),
        "PR_AUC": average_precision_score(
            y_true,
            y_proba
        )
    }

results = []

results.append(
    evaluate_model(
        "Tuned Logistic Regression",
        y_test,
        logistic_pred,
        logistic_proba
    )
)

results.append(
    evaluate_model(
        "Tuned Random Forest",
        y_test,
        random_forest_pred,
        random_forest_proba
    )
)

results.append(
    evaluate_model(
        "Tuned XGBoost",
        y_test,
        xgboost_pred,
        xgboost_proba
    )
)

evaluation_results = pd.DataFrame(results)

print("\nFinal test-set evaluation:")
print(evaluation_results.round(4))

# %% 
# ============================================================
# 8. Save the results
# ============================================================

evaluation_results.to_csv(
    PROJECT_ROOT / "reports" / "results" / "final_model_evaluation.csv",
    index=False
)
# %%
# ============================================================
# 9. Confusion matrices
# ============================================================
def save_confusion_matrix(
    model_name,
    y_true,
    y_pred,
    filename
):

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=["No", "Yes"],
        yticklabels=["No", "Yes"]
    )

    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    
    
save_confusion_matrix(
    "Tuned Logistic Regression",
    y_test,
    logistic_pred,
    "logistic_confusion_matrix.png"
)

save_confusion_matrix(
    "Tuned Random Forest",
    y_test,
    random_forest_pred,
    "random_forest_confusion_matrix.png"
)

save_confusion_matrix(
    "Tuned XGBoost",
    y_test,
    xgboost_pred,
    "xgboost_confusion_matrix.png"
)

# %%

# ============================================================
# 10. ROC Curve
# ============================================================

logistic_fpr, logistic_tpr, _ = roc_curve(
    y_test,
    logistic_proba
)

rf_fpr, rf_tpr, _ = roc_curve(
    y_test,
    random_forest_proba
)

xgb_fpr, xgb_tpr, _ = roc_curve(
    y_test,
    xgboost_proba
)

plt.figure(figsize=(8, 6))

plt.plot(
    logistic_fpr,
    logistic_tpr,
    label="Logistic Regression"
)

plt.plot(
    rf_fpr,
    rf_tpr,
    label="Random Forest"
)

plt.plot(
    xgb_fpr,
    xgb_tpr,
    label="XGBoost"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random classifier"
)

plt.title("ROC Curves - Tuned Models")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "roc_curves.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# %%

# ============================================================
# 11. Precision-Recall Curve
# ============================================================

logistic_precision, logistic_recall, _ = precision_recall_curve(
    y_test,
    logistic_proba
)

rf_precision, rf_recall, _ = precision_recall_curve(
    y_test,
    random_forest_proba
)

xgb_precision, xgb_recall, _ = precision_recall_curve(
    y_test,
    xgboost_proba
)

plt.figure(figsize=(8, 6))

plt.plot(
    logistic_recall,
    logistic_precision,
    label="Logistic Regression"
)

plt.plot(
    rf_recall,
    rf_precision,
    label="Random Forest"
)

plt.plot(
    xgb_recall,
    xgb_precision,
    label="XGBoost"
)

plt.title("Precision-Recall Curves - Tuned Models")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "precision_recall_curves.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# %%

# ============================================================
# 12. Threshold Tuning
# ============================================================

# Generate out-of-fold probabilities for Logistic Regression

logistic_cv_proba = cross_val_predict(
    logistic_pipeline,
    X_train,
    y_train,
    cv=5,
    method="predict_proba",
    n_jobs=-1
)[:, 1]

# %%
# ============================================================
# 13. Test different thresholds
# ============================================================
thresholds = np.arange(
    0.10,
    0.91,
    0.01
)

threshold_results = []

for threshold in thresholds:

    predictions = (
        logistic_cv_proba >= threshold
    ).astype(int)

    threshold_results.append({
        "Threshold": threshold,
        "Precision": precision_score(
            y_train,
            predictions,
            zero_division=0
        ),
        "Recall": recall_score(
            y_train,
            predictions,
            zero_division=0
        ),
        "F1": f1_score(
            y_train,
            predictions,
            zero_division=0
        )
    })

threshold_results = pd.DataFrame(
    threshold_results
)

# %%
# ============================================================
# 14. Find the threshold with the highest F1
# ============================================================

best_threshold_row = threshold_results.loc[
    threshold_results["F1"].idxmax()
]

best_threshold = best_threshold_row["Threshold"]

print("\n" + "=" * 60)
print("THRESHOLD TUNING")
print("=" * 60)

print(
    f"Best threshold: {best_threshold:.2f}"
)

print(
    f"Precision: "
    f"{best_threshold_row['Precision']:.4f}"
)

print(
    f"Recall: "
    f"{best_threshold_row['Recall']:.4f}"
)

print(
    f"F1: "
    f"{best_threshold_row['F1']:.4f}"
)

# %%
# ============================================================
# 15. Save the results
# ============================================================

RESULTS_DIR = PROJECT_ROOT / "reports" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
threshold_results.to_csv(
    RESULTS_DIR / "threshold_tuning.csv",
    index=False
)

# %%
# ============================================================
# 16. Create the threshold figure
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(
    threshold_results["Threshold"],
    threshold_results["Precision"],
    label="Precision"
)

plt.plot(
    threshold_results["Threshold"],
    threshold_results["Recall"],
    label="Recall"
)

plt.plot(
    threshold_results["Threshold"],
    threshold_results["F1"],
    label="F1"
)

plt.axvline(
    best_threshold,
    linestyle="--",
    label=f"Best F1 threshold = {best_threshold:.2f}"
)

plt.title(
    "Logistic Regression - Threshold Tuning"
)

plt.xlabel("Classification Threshold")
plt.ylabel("Score")

plt.legend()
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "threshold_tuning.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
# %%
# ============================================================
# 17. Apply the selected threshold to the test set
# ============================================================

tuned_logistic_pred = (
    logistic_proba >= best_threshold
).astype(int)

tuned_threshold_metrics = {
    "Model": "Tuned Logistic Regression - Threshold",
    "Threshold": best_threshold,
    "Precision": precision_score(
        y_test,
        tuned_logistic_pred,
        zero_division=0
    ),
    "Recall": recall_score(
        y_test,
        tuned_logistic_pred,
        zero_division=0
    ),
    "F1": f1_score(
        y_test,
        tuned_logistic_pred,
        zero_division=0
    )
}

print("\nThreshold-tuned test performance:")
print(
    pd.DataFrame([tuned_threshold_metrics]).round(4)
)

# %%
# ============================================================
# 18. Default vs Threshold-Tuned Logistic Regression
# ============================================================

default_metrics = {
    "Threshold": 0.50,
    "Precision": precision_score(
        y_test,
        logistic_pred,
        zero_division=0
    ),
    "Recall": recall_score(
        y_test,
        logistic_pred,
        zero_division=0
    ),
    "F1": f1_score(
        y_test,
        logistic_pred,
        zero_division=0
    )
}

tuned_metrics = {
    "Threshold": best_threshold,
    "Precision": precision_score(
        y_test,
        tuned_logistic_pred,
        zero_division=0
    ),
    "Recall": recall_score(
        y_test,
        tuned_logistic_pred,
        zero_division=0
    ),
    "F1": f1_score(
        y_test,
        tuned_logistic_pred,
        zero_division=0
    )
}

threshold_comparison = pd.DataFrame([
    default_metrics,
    tuned_metrics
])

print("\n" + "=" * 60)
print("DEFAULT VS THRESHOLD-TUNED LOGISTIC REGRESSION")
print("=" * 60)

print(
    threshold_comparison.round(4)
)

# %%
# ============================================================
# 19. Save this comparison
# ============================================================
threshold_comparison.to_csv(
    RESULTS_DIR / "threshold_comparison.csv",
    index=False
)

# %%
# ============================================================
# 19. Confusion matrix for the threshold-tuned model
# ============================================================
save_confusion_matrix(
    f"Logistic Regression - Threshold {best_threshold:.2f}",
    y_test,
    tuned_logistic_pred,
    "logistic_threshold_confusion_matrix.png"
)

# %%

# ============================================================
# 20. Final Evaluation Summary
# ============================================================

final_summary = evaluation_results.copy()

threshold_row = pd.DataFrame([{
    "Model": "Tuned Logistic Regression - Threshold 0.38",
    "Accuracy": accuracy_score(
        y_test,
        tuned_logistic_pred
    ),
    "Precision": precision_score(
        y_test,
        tuned_logistic_pred,
        zero_division=0
    ),
    "Recall": recall_score(
        y_test,
        tuned_logistic_pred,
        zero_division=0
    ),
    "F1": f1_score(
        y_test,
        tuned_logistic_pred,
        zero_division=0
    ),
    "ROC_AUC": roc_auc_score(
        y_test,
        logistic_proba
    ),
    "PR_AUC": average_precision_score(
        y_test,
        logistic_proba
    )
}])

final_summary = pd.concat(
    [
        final_summary,
        threshold_row
    ],
    ignore_index=True
)

final_summary.to_csv(
    RESULTS_DIR / "final_evaluation_summary.csv",
    index=False
)

print("\n" + "=" * 60)
print("FINAL EVALUATION SUMMARY")
print("=" * 60)

print(
    final_summary.round(4)
)