#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 13:28:11 2026

@author: Rashid Ahmadi
"""

# %% Data Understanding

# What exactly is inside this dataset?

from pathlib import Path

import pandas as pd
import numpy as np

# %%  1. Project paths

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

# %% 2. Load dataset

df = pd.read_csv(DATA_PATH)

# %% 3. Basic information


print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)

print(f"Dataset path: {DATA_PATH}")
print(f"Shape: {df.shape}")

# %% 4. First rows


print("\n" + "=" * 60)
print("FIRST FIVE ROWS")
print("=" * 60)

print(df.head())

# %%  5. Column names

print("\n" + "=" * 60)
print("COLUMNS")
print("=" * 60)

for i, column in enumerate(df.columns, start=1):
    print(f"{i:2d}. {column}")
    
# %% 6. Data types

print("\n" + "=" * 60)
print("DATA TYPES")
print("=" * 60)

print(df.dtypes)

# %% 7. Missing values

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

missing = df.isnull().sum()

print(missing[missing > 0])

if missing.sum() == 0:
    print("No missing values found.")

# %% 8. Duplicate rows

print("\n" + "=" * 60)
print("DUPLICATES")
print("=" * 60)

print("Number of duplicate rows:", df.duplicated().sum())

# %% 9. Target distribution

print("\n" + "=" * 60)
print("TARGET: ATTRITION")
print("=" * 60)

print(df["Attrition"].value_counts())

print("\nRelative distribution:")
print(df["Attrition"].value_counts(normalize=True))