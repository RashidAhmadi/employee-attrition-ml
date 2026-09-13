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
    
# %%
    
# ============================================================
# 4. Numerical features
# ============================================================

numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

print("\n" + "=" * 60)
print("NUMERICAL FEATURES")
print("=" * 60)

print(numeric_columns)

print("\nNumber of numerical features:", len(numeric_columns))

print("\nNumerical summary:")
print(df[numeric_columns].describe().T)


# %%

# ============================================================
# 5. Categorical features
# ============================================================

categorical_columns = df.select_dtypes(
    include="object"
    ).columns.tolist()

print("\n" + "=" * 60)
print("CATEGORICAL FEATURES")
print("=" * 60)

print(categorical_columns)

print("\nNumber of categorical features:", len(categorical_columns))


# %%

# ============================================================
# 6. Attrition by categorical features
# ============================================================

categorical_to_analyze = [
    "OverTime",
    "JobRole",
    "BusinessTravel",
    "MaritalStatus",
    "Department",
    "JobLevel",
]

for column in categorical_to_analyze:

    print("\n" + "=" * 60)
    print(f"ATTRITION RATE BY {column.upper()}")
    print("=" * 60)

    attrition_rate = (
        df.groupby(column)["Attrition"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .sort_values(ascending=False)
    )

    print(attrition_rate.round(2))



# %%

Overtime_attrition = (
    df.groupby("OverTime")["Attrition"]
    .apply(lambda x: (x == "Yes").mean() * 100)
    .sort_values(ascending=False)
)


plt.figure(figsize=(7, 5))


sns.barplot(
    x=Overtime_attrition.index,
    y=Overtime_attrition.values,
)

plt.title("Attrition Rate by Overtime")
plt.xlabel("Overtime")
plt.ylabel("Attrition Rate")

plt.tight_layout()
plt.show()


# %%
# ============================================================
# 7. Attrition by numerical features
# ============================================================

numerical_to_analyze = [
    "Age",
    "MonthlyIncome",
    "DistanceFromHome",
    "TotalWorkingYears",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
]


for column in numerical_to_analyze:

    plt.figure(figsize=(7, 4))

    sns.boxplot(
        data=df,
        x="Attrition",
        y=column
    )

    plt.title(f"{column} vs. Attrition")
    plt.xlabel("Attrition")
    plt.ylabel(column)

    plt.tight_layout()
    plt.show()



# %%

# ============================================================
# 8. Correlation matrix
# ============================================================

plt.figure(figsize=(14, 10))

correlation_matrix = df[numeric_columns].corr()

sns.heatmap(
    correlation_matrix,
    cmap="coolwarm",
    center=0
)

plt.title("Correlation Matrix - Numerical Features")

plt.tight_layout()
plt.show()

# %%

# ============================================================
# 9. Attrition Rate by job role
# ============================================================

job_role_attrition = (
    df.groupby("JobRole")["Attrition"]
    .apply(lambda x: (x == "Yes").mean() * 100)
    .sort_values(ascending=False)
)


plt.figure(figsize=(10, 6))

sns.barplot(
    x=job_role_attrition.values,
    y=job_role_attrition.index
)

plt.title("Attrition Rate by Job Role")
plt.xlabel("Attrition Rate (%)")
plt.ylabel("Job Role")

plt.tight_layout()
plt.show()
