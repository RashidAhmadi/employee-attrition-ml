#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 15:35:46 2026

@author: Rashid Ahmadi

Employee Attrition ML Project
--------------------------------
Hyperparameter tuning using Grid Search.
"""


# %% imports 

import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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
# 2.LOAD THE DATA
# ============================================================
df=pd.read_csv(PROCESSED_PATH)

print("=" * 60)
print("DATASET")
print("=" * 60)

print("Shape:", df.shape)
# %%

# ============================================================
# 3. Feature and target separation
# ============================================================

X=df.drop(columns=["Attrition"])
y=df["Attrition"]
print("\nTarget distribution:")
print(y.value_counts())
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
# 5. Identify feature types
# ============================================================

numerical_columns=X_train.select_dtype(include="number").columns.tolist()
categorical_columns=X_train.select_dtype(include="object").columns.toslist()

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
            OneHotEncoder(
                handle_unknown="ignore"
                ),
         
            categorical_columns
            )
        ]
    )

# %%

# ============================================================
# 7. Logistic Regression pipeline
# ============================================================

lg_pipeline=Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
            ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
                )
            ),
        ]
    )

# %%

# ============================================================
# 8. Define parameter grid
# ============================================================

param_grid = {
    "classifier__C": [
        0.01,
        0.1,
        1,
        10,
        100
    ],
    "classifier__solver": [
        "liblinear",
        "lbfgs"
    ]
}

# %%

# ============================================================
# 9. Grid Search
# ============================================================
grid_search = GridSearchCV(
    estimator=lg_pipeline,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=5,
    n_jobs=-1,
    verbose=1
)

# %%

# ============================================================
#  10. Fit Grid Search
# ============================================================

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION - GRID SEARCH")
print("=" * 60)

grid_search.fit(
    X_train,
    y_train
)


# %% 
# ============================================================
#  11. Best parameters
# ============================================================

print("\nBest parameters:")
print(grid_search.best_params_)

print("\nBest cross-validation ROC-AUC:")
print(round(grid_search.best_score_, 4))

# %% 
# ============================================================
#  12. Save Logistic Regression tuning results
# ============================================================

results = pd.DataFrame(grid_search.cv_results_)

results_path = (
    PROJECT_ROOT
    / "reports"
    / "logistic_grid_search_results.csv"
)

results.to_csv(
    results_path,
    index=False
)

print("\nGrid Search results saved to:")
print(results_path)

# %%

# %% 
# ============================================================
#  13. Random Forest pipeline
# ============================================================

random_forest_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            RandomForestClassifier(
                random_state=42,
                n_jobs=-1
            )
        ),
    ]
)

# %%

# %% 
# ============================================================
#  14. Random Forest parameter grid
# ============================================================

rf_param_grid = {
    "classifier__n_estimators": [
        100,
        200,
        300
    ],
    "classifier__max_depth": [
        None,
        5,
        10,
        20
    ],
    "classifier__min_samples_split": [
        2,
        5,
        10
    ],
    "classifier__min_samples_leaf": [
        1,
        2,
        4
    ]
}

# %%

# %% 
# ============================================================
#  15. Random Forest Grid Search
# ============================================================

rf_grid_search = GridSearchCV(
    estimator=random_forest_pipeline,
    param_grid=rf_param_grid,
    scoring="roc_auc",
    cv=5,
    n_jobs=-1,
    verbose=1
)
# %% 
# ============================================================
#  16. Fit Random Forest Grid Search
# ============================================================

print("\n" + "=" * 60)
print("RANDOM FOREST - GRID SEARCH")
print("=" * 60)

rf_grid_search.fit(
    X_train,
    y_train
)

# %% 
# ============================================================
#  17. Random Forest results
# ============================================================

print("\nBest Random Forest parameters:")
print(rf_grid_search.best_params_)

print("\nBest Random Forest CV ROC-AUC:")
print(round(rf_grid_search.best_score_, 4))
