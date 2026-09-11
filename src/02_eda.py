#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 21:12:53 2026

@author: Rashid Ahmadi

Employee Attrition Prediction
02 - Exploratory Data Analysis

Machine Learning Project
"""
# %%

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
# 2. Load data
# ============================================================
df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# %%
# ============================================================
# 3. Target distribution
# ============================================================

print("\n" + "=" * 60)
print("ATTRITION DISTRIBUTION")
print("=" * 60)

print(df["Attrition"].value_counts())

print("\nPercentages:")
print(
    df["Attrition"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

# %% create the first plot:
    
plt.figure(figsize=(6,4))

sns.countplot(
    data=df,
    x="Attrition"
    )    
    
plt.title("Employee Attrition Distribution")
plt.xlabel("Attrition")
plt.ylabel("Number of Employee")

plt.tight_layout()
plt.show()    
    
    

