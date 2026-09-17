#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 13:26:32 2026

@author: Rashid Ahmadi
"""

# %%

# %% imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

# %%

# %% 
# ============================================================
# 1. Project paths
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

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# %% 
# ============================================================
# 1. Load data
# ============================================================

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["Attrition"])
y = df["Attrition"].map(
    {
     "No": 0,
     "Yes": 1
     }
    )

print("\nData shape:")
print(df.shape)

print("\nTarget distribution:")
print(y.value_counts())

# %% 
# ============================================================
# 2. Train/test split
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# %% 
# ============================================================
# 3. Identify numerical and categorical features
# ============================================================
numeric_columns = X_train.select_dtypes(
    include=np.number
).columns.tolist()

categorical_columns = X_train.select_dtypes(
    include="object"
).columns.tolist()

print("\nNumber of numerical features:", len(numeric_columns))
print(
    "Number of categorical features:",
    len(categorical_columns)
)

# %% 
# ============================================================
# 4. Preprocessing
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numeric_columns
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_columns
        )
    ]
)

# %%
# ============================================================
# 5.  XGBoost model
# ============================================================

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
# 6. XGBoost pipeline
# ============================================================
xgboost_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            xgboost_model
        )
    ]
)

# %% 
# ============================================================
# 7. Train model
# ============================================================

xgboost_pipeline.fit(
    X_train,
    y_train
)

print("\nXGBoost model trained successfully.")

# %% 
# ============================================================
# 8. Transform the data for SHAP
# ============================================================

X_train_transformed = (
    xgboost_pipeline
    .named_steps["preprocessor"]
    .transform(X_train)
)

X_test_transformed = (
    xgboost_pipeline
    .named_steps["preprocessor"]
    .transform(X_test)
)

print("\nTransformed training shape:")
print(X_train_transformed.shape)

print("\nTransformed test shape:")
print(X_test_transformed.shape)

# %% 
# ============================================================
# 9. Feature names
# ============================================================
feature_names = (
    xgboost_pipeline
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

clean_feature_names = np.array([
    name.replace("num__", "").replace("cat__", "")
    for name in feature_names
])

print("\nFirst 20 cleaned feature names:")
print(clean_feature_names[:20])


# %% 
# ============================================================
# 10. SHAP explainer
# ============================================================

xgb_model = (
    xgboost_pipeline
    .named_steps["classifier"]
)

explainer = shap.TreeExplainer(
    xgb_model
)

# %% 
# ============================================================
# 11. Calculate SHAP values
# ============================================================

shap_values = explainer.shap_values(
    X_test_transformed
)

print("\nSHAP values shape:")
print(shap_values.shape)


# %%
# ============================================================
# 12.  SHAP beeswarm plot
# ============================================================
plt.figure()

shap.summary_plot(
    shap_values,
    X_test_transformed,
    feature_names=clean_feature_names,
    show=False
)

plt.title(
    "SHAP Feature Impact - XGBoost"
)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR
    / "shap_beeswarm_xgboost.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# %% 
# ============================================================
# 13.  SHAP feature importance
# ============================================================

plt.figure()

shap.summary_plot(
    shap_values,
    X_test_transformed,
    feature_names=clean_feature_names,
    plot_type="bar",
    show=False
)

plt.title(
    "SHAP Feature Importance - XGBoost"
)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR
    / "shap_feature_importance_xgboost.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# %% 
# ============================================================
# 14. Individual prediction
# ============================================================

employee_index = 0

employee_shap = shap_values[employee_index]

print("\nEmployee index:", employee_index)

print(
    "\nPredicted probability of attrition:"
)

print(
    xgboost_pipeline.predict_proba(
        X_test.iloc[[employee_index]]
    )[0, 1]
)

# %% 
# ============================================================
# 15. SHAP waterfall plot
# ============================================================

explanation = shap.Explanation(
    values=shap_values[employee_index],
    base_values=explainer.expected_value,
    data=X_test_transformed[
        employee_index
    ],
    feature_names=clean_feature_names
)

shap.plots.waterfall(
    explanation,
    max_display=15,
    show=False
)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR
    / "shap_individual_employee.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
# %%
# ============================================================
# 15. Calculate mean absolute SHAP importance
# ============================================================

shap_importance = np.abs(shap_values).mean(axis=0)

shap_importance_df = pd.DataFrame({
    "Feature": clean_feature_names,
    "Mean_Abs_SHAP": shap_importance
}).sort_values(
    "Mean_Abs_SHAP",
    ascending=False
)

print("\nTop 15 features by SHAP importance:")
print(shap_importance_df.head(15))

# %%

RESULTS_DIR = PROJECT_ROOT / "reports" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

shap_importance_df.to_csv(
    RESULTS_DIR / "shap_feature_importance.csv",
    index=False
)