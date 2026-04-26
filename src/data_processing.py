"""
Data Processing Module
======================
Handles data loading, cleaning, and preprocessing for the Heart Disease dataset.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Feature names for the UCI Heart Disease dataset
FEATURE_NAMES = [
    "age",  # Age in years
    "sex",  # Sex (1 = male; 0 = female)
    "cp",  # Chest pain type (1-4)
    "trestbps",  # Resting blood pressure (mm Hg)
    "chol",  # Serum cholesterol (mg/dl)
    "fbs",  # Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
    "restecg",  # Resting ECG results (0-2)
    "thalach",  # Maximum heart rate achieved
    "exang",  # Exercise induced angina (1 = yes; 0 = no)
    "oldpeak",  # ST depression induced by exercise
    "slope",  # Slope of peak exercise ST segment (1-3)
    "ca",  # Number of major vessels (0-3) colored by fluoroscopy
    "thal",  # Thalassemia (3 = normal; 6 = fixed defect; 7 = reversible defect)
    "target",  # Diagnosis of heart disease (0 = no disease, 1-4 = disease)
]

# Categorical and numerical features
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
NUMERICAL_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]


def load_data(
    data_path: str = "heart_disease_data/processed.cleveland.data",
) -> pd.DataFrame:
    """
    Load the Heart Disease dataset from CSV file.

    Args:
        data_path: Path to the data file

    Returns:
        DataFrame with loaded data
    """
    logger.info(f"Loading data from {data_path}")

    # Load data with proper column names
    df = pd.read_csv(
        data_path,
        names=FEATURE_NAMES,
        na_values=["?"],  # Handle missing values marked as '?'
        header=None,
    )

    logger.info(f"Loaded {len(df)} samples with {len(df.columns)} features")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by handling missing values and data type issues.

    Args:
        df: Raw DataFrame

    Returns:
        Cleaned DataFrame
    """
    logger.info("Cleaning data...")
    df_clean = df.copy()

    # Check for missing values
    missing_counts = df_clean.isnull().sum()
    if missing_counts.any():
        logger.info(f"Missing values found:\n{missing_counts[missing_counts > 0]}")

    # Handle missing values
    # For numerical features: fill with median
    for col in NUMERICAL_FEATURES:
        if col in df_clean.columns and df_clean[col].isnull().any():
            median_val = df_clean[col].median()
            df_clean[col].fillna(median_val, inplace=True)
            logger.info(f"Filled {col} missing values with median: {median_val}")

    # For categorical features: fill with mode
    for col in CATEGORICAL_FEATURES:
        if col in df_clean.columns and df_clean[col].isnull().any():
            mode_val = df_clean[col].mode()[0]
            df_clean[col].fillna(mode_val, inplace=True)
            logger.info(f"Filled {col} missing values with mode: {mode_val}")

    # Handle 'ca' column specifically (often has missing values)
    if "ca" in df_clean.columns and df_clean["ca"].isnull().any():
        df_clean["ca"].fillna(df_clean["ca"].median(), inplace=True)

    # Handle 'thal' column specifically
    if "thal" in df_clean.columns and df_clean["thal"].isnull().any():
        df_clean["thal"].fillna(df_clean["thal"].mode()[0], inplace=True)

    logger.info(f"Cleaning complete. Final shape: {df_clean.shape}")
    return df_clean


def convert_target_to_binary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the target variable to binary classification.
    Original: 0 = no disease, 1-4 = disease presence
    Binary: 0 = no disease, 1 = disease

    Args:
        df: DataFrame with original target

    Returns:
        DataFrame with binary target
    """
    df_binary = df.copy()
    df_binary["target"] = (df_binary["target"] > 0).astype(int)

    class_counts = df_binary["target"].value_counts()
    logger.info(
        f"Target distribution: 0 (No Disease): {class_counts.get(0, 0)}, "
        f"1 (Disease): {class_counts.get(1, 0)}"
    )

    return df_binary


def get_feature_target_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split DataFrame into features (X) and target (y).

    Args:
        df: DataFrame with features and target

    Returns:
        Tuple of (X, y)
    """
    X = df.drop("target", axis=1)
    y = df["target"]
    return X, y


def prepare_data(
    data_path: str = "heart_disease_data/processed.cleveland.data",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Complete data preparation pipeline.

    Args:
        data_path: Path to the data file
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    from sklearn.model_selection import train_test_split

    # Load and clean data
    df = load_data(data_path)
    df = clean_data(df)
    df = convert_target_to_binary(df)

    # Split features and target
    X, y = get_feature_target_split(df)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info(f"Train set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")

    return X_train, X_test, y_train, y_test


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Generate a summary of the dataset for reporting.

    Args:
        df: DataFrame to summarize

    Returns:
        Dictionary with summary statistics
    """
    summary = {
        "n_samples": len(df),
        "n_features": len(df.columns) - 1,  # Excluding target
        "missing_values": df.isnull().sum().sum(),
        "target_distribution": (
            df["target"].value_counts().to_dict() if "target" in df.columns else {}
        ),
        "feature_types": {
            "numerical": NUMERICAL_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
        },
        "numeric_stats": df[NUMERICAL_FEATURES].describe().to_dict(),
    }
    return summary


if __name__ == "__main__":
    # Test the data processing pipeline
    X_train, X_test, y_train, y_test = prepare_data()
    print(f"\nData preparation complete!")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {list(X_train.columns)}")
