#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 8:15:04 2026

@author: Rashid Ahmadi
"""

# %%

"""
03_preprocessing.py

Employee Attrition ML Project
"""
# %%  imports
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
# %%

# ============================================================
# 1. Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
)
# %%

# ============================================================
# 2. Load Data
# ============================================================

df=pd.read_csv(DATA_PATH)

# %%

print(df.columns.tolist())

# %%

# ============================================================
# 3. Remove unnecessary columns
# ============================================================

"""
We are removing some unnecessary columns like EmployeeNumber
"""

columns_to_drop = [
    "EmployeeNumber",
    "EmployeeCount",
    "Over18",
    "StandardHours",
]

df = df.drop(columns=columns_to_drop, errors="ignore")

# %%
# ============================================================
# 4. Save cleaned dataset
# ============================================================

PROCESSED_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "employee_attrition_clean.csv"
)

PROCESSED_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    PROCESSED_PATH,
    index=False
)

print(f"\nCleaned dataset saved to:")
print(PROCESSED_PATH)
# %%

# ============================================================
# 5. Feature and Target 
# ============================================================

X=df.drop(columns=["Attrition"])

y=df["Attrition"]


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
# 6. Train/Test Split
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
# 7. Identify numerical and categorical features
# ============================================================

numeric_columns=df.select_dtypes(
    include=np.number,
    ).columns.tolist()

category_columns=df.select_dtypes(
    include="object"
    ).columns.tolist()



# Remove target variable from feature lists
numeric_columns.remove("Attrition") if "Attrition" in numeric_columns else None
category_columns.remove("Attrition") if "Attrition" in category_columns else None

print("\n" + "=" * 60)
print("NUMERICAL AND CATEGORICAL FEATURES")
print("=" * 60)


print("\nNumber of numerical features:", len(numeric_columns))

print("\nNumber of categorical features:", len(category_columns))

# %%

# ============================================================
# 8. Numerical preprocessing
# ============================================================

numerical_transformer = StandardScaler()
# %%

# ============================================================
# 9. Categorical preprocessing
# ============================================================
categorical_transformer = OneHotEncoder(
    handle_unknown="ignore"
)
# %%
# ============================================================
# 10. ColumnTransformer
# ============================================================

preprocessor=ColumnTransformer(
    transformers=[
        (
            "num",
            numerical_transformer,
            numeric_columns
            ),
        (
            "cat",
            categorical_transformer,
            category_columns
            ),
        ]
    
    )

# %%


# ============================================================
# 11. Fit preprocessing ONLY on training data
# ============================================================
# We fit the preprocessing only on X_train.

X_train_processed=preprocessor.fit_transform(X_train)
X_test_processed=preprocessor.transform(X_test)

# %%

# ============================================================
# 12. Information about transformed data
# ============================================================

print("=" * 60)
print("PREPROCESSING RESULT")
print("=" * 60)

print(f"Original training shape: {X_train.shape}")
print(f"Processed training shape: {X_train_processed.shape}")

print(f"Original test shape:     {X_test.shape}")
print(f"Processed test shape:    {X_test_processed.shape}")

print()

# %%

# ============================================================
# 13. Target distribution
# ============================================================

print("=" * 60)
print("TARGET DISTRIBUTION")
print("=" * 60)

print("Training set:")
print(y_train.value_counts())
print()

print("Training proportions:")
print(y_train.value_counts(normalize=True))
print()

print("Test set:")
print(y_test.value_counts())
print()

print("Test proportions:")
print(y_test.value_counts(normalize=True))
# %%


























