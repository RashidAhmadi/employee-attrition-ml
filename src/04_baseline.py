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

from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
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

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

FIGURES_PATH = PROJECT_ROOT / "reports" / "figures"

FIGURES_PATH.mkdir(parents=True, exist_ok=True)

# %%

# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("BASELINE MODELING")
print("=" * 60)

print("Dataset shape:", df.shape)


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

























