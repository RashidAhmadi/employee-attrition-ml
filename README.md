Employee Attrition Prediction

Machine learning portfolio project for predicting employee attrition using a complete supervised learning pipeline.

Project Overview

Employee attrition is an important business problem for organizations. Being able to identify employees who are at higher risk of leaving can help companies understand workforce patterns and develop appropriate retention strategies.

The goal of this project is to build and evaluate machine learning models that predict whether an employee will leave the company.

The project follows a complete machine learning workflow:

Business problem definition
Exploratory Data Analysis (EDA)
Data cleaning and preprocessing
Train/test splitting
Baseline models
Machine learning model training
Hyperparameter tuning
Model comparison
Model evaluation
Model interpretation with SHAP
Employee segmentation with KMeans
Final model selection
Model serialization
Machine Learning Models

Only models covered in the ML course are used:

Logistic Regression
Random Forest
XGBoost
Stacking Ensemble

KMeans is additionally used for unsupervised employee segmentation.

Project Structure
employee-attrition-ml/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── 01_data_loading.py
│   ├── 02_eda.py
│   ├── 03_preprocessing.py
│   ├── 04_baseline.py
│   ├── 05_model_training.py
│   ├── 06_hyperparameter_tuning.py
│   ├── 07_evaluation.py
│   ├── 08_interpretability.py
│   └── 09_clustering.py
│
├── models/
├── reports/
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
Dataset

The dataset contains employee information such as demographic characteristics, job-related information, compensation, satisfaction, working conditions, and other employee attributes.

The raw dataset is intentionally not included in this repository.

Evaluation

Because employee attrition is a binary classification problem and the classes may be imbalanced, model performance will not be evaluated using accuracy alone.

The project will consider metrics such as:

ROC-AUC
Precision
Recall
F1-score
Confusion Matrix
Precision-Recall analysis
Hyperparameter Tuning

Several approaches will be compared, including:

Baseline/default parameters
Randomized Search
Optuna-based optimization

The objective is not simply to maximize a single score, but to understand how different models and hyperparameters affect predictive performance and generalization.

Interpretability

SHAP will be used to investigate which employee characteristics have the strongest influence on model predictions.

This helps connect the machine learning results to the underlying business problem.

Clustering

KMeans clustering will be used as an unsupervised learning extension to identify groups of employees with similar characteristics.

Status

🚧 Project currently in development.

The repository will be updated progressively as the different stages of the machine learning pipeline are completed.