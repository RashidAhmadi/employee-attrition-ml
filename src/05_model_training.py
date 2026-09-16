#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 8:16:31 2026

@author: Rashid Ahmadi
"""

# %% imports 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

from xgboost import XGBClassifier

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
# 2. LOAD CLEAN DATA
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
y=df["Attrition"].map(
    {
     "No":0,
     "Yes":1
     }
    )

print("\nTarget distribution:")
print(y.value_counts())
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

print("\nTraining samples:", X_train.shape[0])
print("Test samples:", X_test.shape[0])

# %%

# ============================================================
# 5. Identify numerical and categorical features
# ============================================================

numerical_columns=X_train.select_dtypes(include="number").columns.tolist()
categorical_columns=X_train.select_dtypes(include="object").columns.tolist()

print("\nNumerical features:", len(numerical_columns))
print("Categorical features:", len(categorical_columns))


# %%

# ============================================================
# 6. Preprocessing
# ============================================================

preprocessor=ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_columns
        ),
        (
            "cat",
            OneHotEncoder(),
            categorical_columns
            ),
        ]
    )

# %%


# ============================================================
# 7. Define models
# ============================================================

lg=LogisticRegression(
    max_iter=1000,
    random_state=42
    )

rf=RandomForestClassifier(
    random_state=42,
    n_jobs=-1,
    )


xgb=XGBClassifier(
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1
    )


# %%

# ============================================================
# 8. Create pipelines
# ============================================================

lg_pipeline=Pipeline(
    steps=[
        ("preprocessor", preprocessor ),
        ("classifier", lg)
        ]
    )

rf_pipeline=Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", rf)
        
        ]
    )

xgb_pipeline=Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", xgb)
        ]
    )

# %%

# ============================================================
# 9. Train Logistic Regression
# ============================================================

print("\n" + "=" * 60)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 60)

lg_pipeline.fit(
    X_train,
    y_train
    )

lg_pred=lg_pipeline.predict(X_test)
lg_proba=lg_pipeline.predict_proba(X_test)[:,1]

# ============================================================
# 10. Train the Random Forest
# ============================================================

print("\n" + "=" * 60)
print("TRAINING RANDOM FOREST")
print("=" * 60)

rf_pipeline.fit(
    X_train,
    y_train
    )

rf_pred=rf_pipeline.predict(X_test)
rf_proba=rf_pipeline.predict_proba(X_test)[:,1]


# ============================================================
# 11. Train the Random Forest
# ============================================================

print("\n" + "=" * 60)
print("TRAINING XGBOOST")
print("=" * 60)


xgb_pipeline.fit(
    X_train,
    y_train
    
    )


xgb_pred=xgb_pipeline.predict(X_test)
xgb_proba=xgb_pipeline.predict_proba(X_test)[:,1]


# %%

# ============================================================
# 12. Evaluation function
# ============================================================

def evaluate_model(
        model_name,
        y_true,
        y_pred,
        y_proba
        ):
    
    return{
        "Model":model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, pos_label=1),
        "Recall": recall_score(y_true, y_pred, pos_label=1),
        "F1": f1_score(y_true, y_pred, pos_label=1),
        "ROC_AUC": roc_auc_score(y_true, y_proba),
        "PR_AUC" : average_precision_score(y_true, y_proba),
        
        }


# %%

# ============================================================
# 13. Evaluate all models
# ============================================================

results=[]

results.append(
    evaluate_model(
        "Logistic Regression",
        y_test, 
        lg_pred,
        lg_proba)
    )


results.append(
    evaluate_model(
        "Random Forest",
        y_test,
        rf_pred,
        rf_proba)
    )

results.append(
    evaluate_model(
        "XGBoost",
        y_test,
        xgb_pred,
        xgb_proba
        )
    )

results_df = pd.DataFrame(results)

# %%

# ============================================================
# 14. Display results
# ============================================================

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    results_df.round(4).to_string(index=False)
)


# %%
# ============================================================
#  15. Save results
# ============================================================
results_path = (
    PROJECT_ROOT
    / "reports" / "results"
    / "model_training_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)

print("\nResults saved to:")
print(results_path)

# %%

# ============================================================
#  Model comparison figure
# ============================================================

plt.figure(figsize=(10,6))

results_melted = results_df.melt(
    id_vars="Model",
    value_vars=[
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC_AUC",
        "PR_AUC",
        ],
    var_name="Metric",
    value_name="Score"
    )

sns.barplot(
    data=results_melted,
    x="Model",
    y="Score",
    hue="Metric"
    )

plt.title(
    "Employee Attrition - Model Comparison"
    )

plt.xlabel("Model")
plt.ylabel("Score")

plt.ylim(0,1)

plt.legend(
    title="Metric",
    bbox_to_anchor=(1.05, 1),
    loc="upper left"
    )
    

plt.tight_layout()

comparison_path = (
    FIGURES_DIR
    / "model_comparison.png"
)

plt.savefig(
    comparison_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()

print("\nFigure saved to:")
print(comparison_path)












