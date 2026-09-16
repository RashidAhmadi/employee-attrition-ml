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
import optuna
import pyhopper
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.model_selection import ( train_test_split, 
                                     GridSearchCV, 
                                     RandomizedSearchCV,
                                     cross_val_score
                                     )
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from scipy.stats import loguniform
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
y=df["Attrition"].map(
    {
    "No": 0,
    "Yes": 1
    }
    )
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

numerical_columns=X_train.select_dtypes(include="number").columns.tolist()
categorical_columns=X_train.select_dtypes(include="object").columns.tolist()

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
    / "reports" / "results"
    / "logistic_grid_search_results.csv"
)

results.to_csv(
    results_path,
    index=False
)

print("\nGrid Search results saved to:")
print(results_path)

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

# %% 
# ============================================================
#  18. XGBoost pipeline
# ============================================================

xgboost_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            XGBClassifier(
                random_state=42,
                eval_metric="logloss",
                n_jobs=-1
            )
        ),
    ]
)
# %% 
# ============================================================
# 19. XGBoost parameter grid
# ============================================================
xgb_param_grid = {
    "classifier__n_estimators": [
        100,
        200
    ],
    "classifier__max_depth": [
        3,
        5,
        7
    ],
    "classifier__learning_rate": [
        0.01,
        0.1,
        0.2
    ],
    "classifier__subsample": [
        0.8,
        1.0
    ]
}
# %% 
# ============================================================
# 20. XGBoost Grid Search
# ============================================================
xgb_grid_search = GridSearchCV(
    estimator=xgboost_pipeline,
    param_grid=xgb_param_grid,
    scoring="roc_auc",
    cv=5,
    n_jobs=-1,
    verbose=1
)

# %% 
# ============================================================
# 21. Fit XGBoost Grid Search
# ============================================================
print("\n" + "=" * 60)
print("XGBOOST - GRID SEARCH")
print("=" * 60)

xgb_grid_search.fit(
    X_train,
    y_train
)

# %% 
# ============================================================
# 22. XGBoost results
# ============================================================
print("\nBest XGBoost parameters:")
print(xgb_grid_search.best_params_)

print("\nBest XGBoost CV ROC-AUC:")
print(round(xgb_grid_search.best_score_, 4))

# %% 
# ============================================================
# 23. Save tuning results
# ============================================================


tuning_results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "XGBoost"
    ],
    "Best_CV_ROC_AUC": [
        grid_search.best_score_,
        rf_grid_search.best_score_,
        xgb_grid_search.best_score_
    ]
})

results_path = (
    PROJECT_ROOT
    / "reports" / "results"
    / "grid_search_results.csv"
)

tuning_results.to_csv(
    results_path,
    index=False
)

print("\n" + "=" * 60)
print("GRID SEARCH RESULTS SAVED")
print("=" * 60)

print(tuning_results)

# %% 
# ============================================================
# 24. Logistic Regression - Random Search
# ============================================================
print("\n" + "=" * 60)
print("LOGISTIC REGRESSION - RANDOM SEARCH")
print("=" * 60)

logistic_random = RandomizedSearchCV(
    estimator=lg_pipeline,
    param_distributions={
        "classifier__C": loguniform(0.01, 100),
    },
    n_iter=10,
    scoring="roc_auc",
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

logistic_random.fit(
    X_train,
    y_train
)

# %% 
# ============================================================
# 25. Logistic Regression Random Search results
# ============================================================

print("\nBest Logistic Regression parameters:")
print(logistic_random.best_params_)

print("\nBest Logistic Regression Random Search CV ROC-AUC:")
print(f"{logistic_random.best_score_:.4f}")

# %% 
# ============================================================
# 26. Random Forest - Random Search
# ============================================================

print("\n" + "=" * 60)
print("RANDOM FOREST - RANDOM SEARCH")
print("=" * 60)

rf_random = RandomizedSearchCV(
    estimator=random_forest_pipeline,
    param_distributions={
        "classifier__n_estimators": [100, 200, 300, 500],
        "classifier__max_depth": [None, 5, 10, 15, 20],
        "classifier__min_samples_split": [2, 5, 10, 20],
        "classifier__min_samples_leaf": [1, 2, 4, 8],
    },
    n_iter=15,
    scoring="roc_auc",
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

rf_random.fit(
    X_train,
    y_train
)

# %%
# ============================================================
#  27. Random Forest Random Search results
# ============================================================

print("\nBest Random Forest parameters:")
print(rf_random.best_params_)

print("\nBest Random Forest Random Search CV ROC-AUC:")
print(f"{rf_random.best_score_:.4f}")

# %%
# ============================================================
#   28. XGBoost - Random Search
# ============================================================

print("\n" + "=" * 60)
print("XGBOOST - RANDOM SEARCH")
print("=" * 60)

xgb_random = RandomizedSearchCV(
    estimator=xgboost_pipeline,
    param_distributions={
        "classifier__n_estimators": [100, 200, 300, 500],
        "classifier__max_depth": [3, 5, 7, 9],
        "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "classifier__subsample": [0.6, 0.8, 1.0],
    },
    n_iter=15,
    scoring="roc_auc",
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

xgb_random.fit(
    X_train,
    y_train
)

# %% 
# ============================================================
#  29. XGBoost Random Search results
# ============================================================
print("\nBest XGBoost parameters:")
print(xgb_random.best_params_)

print("\nBest XGBoost Random Search CV ROC-AUC:")
print(f"{xgb_random.best_score_:.4f}")

# %% 
# ============================================================
#  30. Save Random Search results
# ============================================================
random_search_results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "XGBoost"
    ],
    "Best_Random_Search_ROC_AUC": [
        logistic_random.best_score_,
        rf_random.best_score_,
        xgb_random.best_score_
    ]
})

results_path = (
    PROJECT_ROOT
    / "reports" / "results"
    / "random_search_results.csv"
)

random_search_results.to_csv(
    results_path,
    index=False
)

print("\n" + "=" * 60)
print("RANDOM SEARCH RESULTS SAVED")
print("=" * 60)

print(random_search_results)

# %% 
# ============================================================
#  31. Optuna - XGBoost
# ============================================================

def objective(trial):

    params = {
        "n_estimators": trial.suggest_int(
            "n_estimators",
            100,
            500,
            step=100
        ),
        "max_depth": trial.suggest_int(
            "max_depth",
            3,
            9
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.01,
            0.2,
            log=True
        ),
        "subsample": trial.suggest_float(
            "subsample",
            0.6,
            1.0
        )
    }

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                XGBClassifier(
                    **params,
                    random_state=42,
                    eval_metric="logloss",
                    n_jobs=-1
                )
            )
        ]
    )

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1
    )

    return scores.mean()

# %% 
# ============================================================
#  32. Create Optuna study
# ============================================================
print("\n" + "=" * 60)
print("XGBOOST - OPTUNA")
print("=" * 60)

study = optuna.create_study(
    direction="maximize",
    study_name="xgboost_attrition"
)

# %% 
# ============================================================
#  33. Run Optuna optimization
# ============================================================
study.optimize(
    objective,
    n_trials=20
)

# %% 
# ============================================================
#  34. Optuna results
# ============================================================

print("\nBest Optuna parameters:")
print(study.best_params)

print("\nBest Optuna CV ROC-AUC:")
print(f"{study.best_value:.4f}")

# %% 
# ============================================================
#  35. Save Optuna results
# ============================================================

optuna_results = pd.DataFrame({
    "Model": ["XGBoost"],
    "Best_Optuna_ROC_AUC": [study.best_value]
})

results_path = (
    PROJECT_ROOT
    / "reports" / "results"
    / "optuna_results.csv"
)

optuna_results.to_csv(
    results_path,
    index=False
)

print("\n" + "=" * 60)
print("OPTUNA RESULTS SAVED")
print("=" * 60)

print(optuna_results)

# %% 
# ============================================================
#  36. PyHopper - XGBoost
# ============================================================
def pyhopper_objective(params):

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", XGBClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                learning_rate=params["learning_rate"],
                subsample=params["subsample"],
                random_state=42,
                eval_metric="logloss",
                n_jobs=-1
            ))
        ]
    )

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1
    )

    return scores.mean()

# %%
# ============================================================
#   37. PyHopper search space
# ============================================================
search = pyhopper.Search(
    {
        "n_estimators": pyhopper.int(100, 500),
        "max_depth": pyhopper.int(3, 9),
        "learning_rate": pyhopper.float(0.01, 0.2, log=True),
        "subsample": pyhopper.float(0.6, 1.0),
    }
)

# %% 
# ============================================================
#   38. Run PyHopper
# ============================================================

print("\n" + "=" * 60)
print("XGBOOST - PYHOPPER")
print("=" * 60)

best_params = search.run(
    pyhopper_objective,
    direction="max",
    runtime="20m",
    n_jobs=1
)

# %%
# ============================================================
#   39. PyHopper results
# ============================================================

print("\nBest PyHopper parameters:")
print(best_params)

print("\nBest PyHopper CV ROC-AUC:")
print(f"{search.best_f:.4f}")
# %%
# ============================================================
# 40. Save PyHopper results
# ============================================================

pyhopper_results = pd.DataFrame({
    "Model": ["XGBoost"],
    "Best_PyHopper_ROC_AUC": [search.best_f],
    "Best_Params": [str(best_params)]
})

results_path = (
    PROJECT_ROOT
    / "reports" / "results"
    / "pyhopper_results.csv"
)

pyhopper_results.to_csv(
    results_path,
    index=False
)

print("\nPyHopper results saved.")
# %%

# ============================================================
# 41. Compare tuning methods
# ============================================================

tuning_comparison = pd.DataFrame({
    "Method": [
        "Grid Search",
        "Random Search",
        "Optuna",
        "PyHopper"
    ],
    "XGBoost_CV_ROC_AUC": [
        xgb_grid_search.best_score_,
        xgb_random.best_score_,
        study.best_value,
        search.best_f
    ]
})

results_path = (
    PROJECT_ROOT
    / "reports" / "results"
    / "xgboost_tuning_results.csv"
)

tuning_comparison.to_csv(
    results_path,
    index=False
)

print("\nXGBoost tuning comparison:")
print(tuning_comparison)

# %%

plt.figure(figsize=(9, 5))

sns.barplot(
    data=tuning_comparison,
    x="Method",
    y="XGBoost_CV_ROC_AUC"
)

plt.title("XGBoost Hyperparameter Tuning Comparison")
plt.xlabel("Tuning Method")
plt.ylabel("Cross-Validation ROC-AUC")
plt.ylim(0.75, 0.85)
plt.tight_layout()

plt.savefig(
    PROJECT_ROOT / "reports" / "figures" / "xgboost_tuning_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()