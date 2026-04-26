"""
Feature Engineering Module
==========================
Handles feature transformations and preprocessing pipelines.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

# Feature definitions
NUMERICAL_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom feature engineering transformer.
    Creates derived features to improve model performance.
    """

    def __init__(self, create_interactions: bool = True):
        self.create_interactions = create_interactions

    def fit(self, X: pd.DataFrame, y=None):
        """Fit the transformer (no-op for this transformer)."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features by creating derived features.

        Args:
            X: Input features DataFrame

        Returns:
            DataFrame with original and derived features
        """
        X_transformed = X.copy()

        # Age groups (binning)
        if "age" in X_transformed.columns:
            X_transformed["age_group"] = pd.cut(
                X_transformed["age"], bins=[0, 40, 50, 60, 100], labels=[0, 1, 2, 3]
            ).astype(float)

        # Heart rate reserve (estimated)
        if "age" in X_transformed.columns and "thalach" in X_transformed.columns:
            # Max heart rate estimate: 220 - age
            X_transformed["hr_reserve"] = (220 - X_transformed["age"]) - X_transformed[
                "thalach"
            ]

        # Cholesterol to age ratio
        if "chol" in X_transformed.columns and "age" in X_transformed.columns:
            X_transformed["chol_age_ratio"] = (
                X_transformed["chol"] / X_transformed["age"]
            )

        # Blood pressure category
        if "trestbps" in X_transformed.columns:
            X_transformed["bp_category"] = pd.cut(
                X_transformed["trestbps"],
                bins=[0, 120, 140, 180, 300],
                labels=[0, 1, 2, 3],  # Normal, Elevated, High, Very High
            ).astype(float)

        # Interaction features
        if self.create_interactions:
            if "exang" in X_transformed.columns and "oldpeak" in X_transformed.columns:
                X_transformed["exang_oldpeak"] = (
                    X_transformed["exang"] * X_transformed["oldpeak"]
                )

            if "cp" in X_transformed.columns and "thalach" in X_transformed.columns:
                X_transformed["cp_thalach"] = (
                    X_transformed["cp"] * X_transformed["thalach"]
                )

        return X_transformed


def create_preprocessing_pipeline(
    numerical_features: List[str] = None,
    categorical_features: List[str] = None,
    scaling_method: str = "standard",
) -> ColumnTransformer:
    """
    Create a preprocessing pipeline for the heart disease dataset.

    Args:
        numerical_features: List of numerical feature names
        categorical_features: List of categorical feature names
        scaling_method: 'standard' or 'minmax'

    Returns:
        ColumnTransformer preprocessing pipeline
    """
    if numerical_features is None:
        numerical_features = NUMERICAL_FEATURES
    if categorical_features is None:
        categorical_features = CATEGORICAL_FEATURES

    # Numerical preprocessing pipeline
    if scaling_method == "standard":
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()

    numerical_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", scaler)]
    )

    # Categorical preprocessing pipeline
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    # Combine pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="passthrough",
    )

    return preprocessor


def create_full_pipeline(
    model, include_feature_engineering: bool = False, scaling_method: str = "standard"
) -> Pipeline:
    """
    Create a complete ML pipeline including preprocessing and model.

    Args:
        model: Sklearn-compatible model instance
        include_feature_engineering: Whether to include custom feature engineering
        scaling_method: Scaling method to use

    Returns:
        Complete sklearn Pipeline
    """
    steps = []

    # Optional feature engineering
    if include_feature_engineering:
        steps.append(("feature_engineer", FeatureEngineer()))

    # Preprocessing
    steps.append(
        ("preprocessor", create_preprocessing_pipeline(scaling_method=scaling_method))
    )

    # Model
    steps.append(("model", model))

    return Pipeline(steps)


def save_pipeline(pipeline: Pipeline, filepath: str) -> None:
    """
    Save a trained pipeline to disk.

    Args:
        pipeline: Trained sklearn Pipeline
        filepath: Path to save the pipeline
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, filepath)
    logger.info(f"Pipeline saved to {filepath}")


def load_pipeline(filepath: str) -> Pipeline:
    """
    Load a trained pipeline from disk.

    Args:
        filepath: Path to the saved pipeline

    Returns:
        Loaded sklearn Pipeline
    """
    pipeline = joblib.load(filepath)
    logger.info(f"Pipeline loaded from {filepath}")
    return pipeline


def get_feature_names_after_preprocessing(
    preprocessor: ColumnTransformer,
    numerical_features: List[str] = None,
    categorical_features: List[str] = None,
) -> List[str]:
    """
    Get feature names after preprocessing transformation.

    Args:
        preprocessor: Fitted ColumnTransformer
        numerical_features: Original numerical feature names
        categorical_features: Original categorical feature names

    Returns:
        List of feature names after transformation
    """
    if numerical_features is None:
        numerical_features = NUMERICAL_FEATURES
    if categorical_features is None:
        categorical_features = CATEGORICAL_FEATURES

    feature_names = []

    # Numerical features (names stay the same)
    feature_names.extend(numerical_features)

    # Categorical features (expanded by OneHotEncoder)
    try:
        cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
        cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)
        feature_names.extend(cat_feature_names)
    except Exception:
        # Fallback if encoder not fitted
        feature_names.extend(categorical_features)

    return feature_names


if __name__ == "__main__":
    # Test the preprocessing pipeline
    from data_processing import prepare_data

    X_train, X_test, y_train, y_test = prepare_data()

    # Create and fit preprocessor
    preprocessor = create_preprocessing_pipeline()
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    print(f"Original shape: {X_train.shape}")
    print(f"Transformed shape: {X_train_transformed.shape}")
    print(f"Feature names: {get_feature_names_after_preprocessing(preprocessor)}")
