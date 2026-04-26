#!/usr/bin/env python3
"""
MLflow Experiment Tracking Script
=================================
Trains models and logs experiments to MLflow.
"""

import os
import sys
import warnings

import joblib
import matplotlib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

# Change to project root
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 70)
print("MLflow Experiment Tracking")
print("=" * 70)

# Set MLflow tracking URI
mlflow.set_tracking_uri("file:./mlruns")
experiment = mlflow.set_experiment("heart-disease-prediction")

# Load data
df = pd.read_csv("data/heart_disease_cleaned.csv")
print(f"Loaded {len(df)} samples")

NUMERICAL_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]

X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create preprocessor
numerical_pipeline = Pipeline(
    [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)
categorical_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
)
preprocessor = ColumnTransformer(
    [
        ("num", numerical_pipeline, NUMERICAL_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ]
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Models to train
models = {
    "Logistic_Regression": (
        LogisticRegression(max_iter=1000, random_state=42),
        {"C": 1.0, "max_iter": 1000},
    ),
    "Random_Forest": (
        RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        {"n_estimators": 100, "max_depth": 10},
    ),
    "Gradient_Boosting": (
        GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
        {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1},
    ),
}

print()
for name, (model, params) in models.items():
    with mlflow.start_run(run_name=name):
        print(f"Training {name}...")

        # Train
        model.fit(X_train_processed, y_train)

        # Predict
        y_pred = model.predict(X_test_processed)
        y_prob = model.predict_proba(X_test_processed)[:, 1]

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)

        # Log parameters
        mlflow.log_param("model_type", name)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        for k, v in params.items():
            mlflow.log_param(k, v)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        # Save and log model artifact
        temp_model_path = f"/tmp/temp_{name}_model.pkl"
        joblib.dump(model, temp_model_path)
        mlflow.log_artifact(temp_model_path, "model")
        os.remove(temp_model_path)

        # Log existing artifacts
        mlflow.log_artifact("models/preprocessor.pkl")

        print(f"  Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}")
        print(f"  Run ID: {mlflow.active_run().info.run_id}")

print()
print("=" * 70)
print("MLflow tracking complete!")
print(f"Experiment ID: {experiment.experiment_id}")
print("Runs logged to: mlruns/")
print("=" * 70)
