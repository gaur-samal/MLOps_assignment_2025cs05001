"""
Model Training Module
=====================
Handles model training, evaluation, and experiment tracking with MLflow.
"""

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt

# Note: XGBoost excluded due to libomp dependency issues on macOS
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# Model configurations
MODEL_CONFIGS = {
    "logistic_regression": {
        "model": LogisticRegression,
        "params": {
            "C": [0.01, 0.1, 1, 10],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear", "saga"],
            "max_iter": [1000],
        },
    },
    "random_forest": {
        "model": RandomForestClassifier,
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 10, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
    },
    "gradient_boosting": {
        "model": GradientBoostingClassifier,
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.2],
        },
    },
}


def get_model(model_name: str, **kwargs) -> Any:
    """
    Get a model instance by name.

    Args:
        model_name: Name of the model
        **kwargs: Additional model parameters

    Returns:
        Model instance
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}"
        )

    model_class = MODEL_CONFIGS[model_name]["model"]

    # Set default parameters
    default_params = {}
    if model_name == "logistic_regression":
        default_params = {"max_iter": 1000, "random_state": 42}
    elif model_name == "random_forest":
        default_params = {"random_state": 42, "n_jobs": -1}
    elif model_name == "xgboost":
        default_params = {
            "random_state": 42,
            "use_label_encoder": False,
            "eval_metric": "logloss",
        }
    elif model_name == "gradient_boosting":
        default_params = {"random_state": 42}

    default_params.update(kwargs)
    return model_class(**default_params)


def evaluate_model(
    model, X_test: np.ndarray, y_test: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate a trained model on test data.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        threshold: Classification threshold

    Returns:
        Dictionary of evaluation metrics
    """
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

    return metrics


def cross_validate_model(
    model, X: np.ndarray, y: np.ndarray, cv: int = 5, scoring: List[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Perform cross-validation on a model.

    Args:
        model: Model to evaluate
        X: Features
        y: Labels
        cv: Number of folds
        scoring: List of scoring metrics

    Returns:
        Dictionary with mean and std for each metric
    """
    if scoring is None:
        scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    results = {}
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    for metric in scoring:
        scores = cross_val_score(model, X, y, cv=skf, scoring=metric)
        results[metric] = {
            "mean": scores.mean(),
            "std": scores.std(),
            "scores": scores.tolist(),
        }

    return results


def hyperparameter_tuning(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: int = 5,
    n_jobs: int = -1,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Perform hyperparameter tuning using GridSearchCV.

    Args:
        model_name: Name of the model to tune
        X_train: Training features
        y_train: Training labels
        cv: Number of cross-validation folds
        n_jobs: Number of parallel jobs

    Returns:
        Tuple of (best_model, best_params)
    """
    config = MODEL_CONFIGS[model_name]
    model = get_model(model_name)
    param_grid = config["params"]

    logger.info(f"Starting hyperparameter tuning for {model_name}")

    grid_search = GridSearchCV(
        model, param_grid, cv=cv, scoring="roc_auc", n_jobs=n_jobs, verbose=1
    )

    grid_search.fit(X_train, y_train)

    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best ROC-AUC score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, grid_search.best_params_


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, save_path: str = None
) -> plt.Figure:
    """
    Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save the figure

    Returns:
        Matplotlib figure
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Disease", "Disease"],
        yticklabels=["No Disease", "Disease"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Confusion matrix saved to {save_path}")

    return fig


def plot_roc_curve(
    y_true: np.ndarray, y_prob: np.ndarray, save_path: str = None
) -> plt.Figure:
    """
    Plot ROC curve.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        save_path: Path to save the figure

    Returns:
        Matplotlib figure
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, "b-", linewidth=2, label=f"ROC Curve (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.3)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"ROC curve saved to {save_path}")

    return fig


def plot_feature_importance(
    model, feature_names: List[str], top_n: int = 15, save_path: str = None
) -> plt.Figure:
    """
    Plot feature importance for tree-based models.

    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        top_n: Number of top features to show
        save_path: Path to save the figure

    Returns:
        Matplotlib figure
    """
    if not hasattr(model, "feature_importances_"):
        logger.warning("Model does not have feature_importances_ attribute")
        return None

    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(indices)), importances[indices], align="center")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel("Feature Importance")
    ax.set_title("Top Feature Importances")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Feature importance plot saved to {save_path}")

    return fig


def train_with_mlflow(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    experiment_name: str = "heart_disease_classification",
    run_name: str = None,
    hyperparameter_tuning_enabled: bool = False,
) -> Tuple[Any, Dict[str, float]]:
    """
    Train a model with MLflow experiment tracking.

    Args:
        model_name: Name of the model to train
        X_train, y_train: Training data
        X_test, y_test: Test data
        experiment_name: MLflow experiment name
        run_name: Name for this run
        hyperparameter_tuning_enabled: Whether to perform hyperparameter tuning

    Returns:
        Tuple of (trained_model, metrics)
    """
    # Set up MLflow
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name or f"{model_name}_run"):
        # Log model type
        mlflow.log_param("model_type", model_name)

        # Train model (with optional tuning)
        if hyperparameter_tuning_enabled:
            model, best_params = hyperparameter_tuning(model_name, X_train, y_train)
            mlflow.log_params(best_params)
        else:
            model = get_model(model_name)
            model.fit(X_train, y_train)

        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)

        # Log metrics
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)

        # Log cross-validation results
        cv_results = cross_validate_model(model, X_train, y_train)
        for metric_name, values in cv_results.items():
            mlflow.log_metric(f"cv_{metric_name}_mean", values["mean"])
            mlflow.log_metric(f"cv_{metric_name}_std", values["std"])

        # Create and log plots
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Confusion matrix
        cm_fig = plot_confusion_matrix(y_test, y_pred)
        mlflow.log_figure(cm_fig, "confusion_matrix.png")
        plt.close(cm_fig)

        # ROC curve
        roc_fig = plot_roc_curve(y_test, y_prob)
        mlflow.log_figure(roc_fig, "roc_curve.png")
        plt.close(roc_fig)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        logger.info(f"Model training complete. Metrics: {metrics}")

    return model, metrics


def save_model(model, filepath: str) -> None:
    """
    Save a trained model to disk.

    Args:
        model: Trained model
        filepath: Path to save the model
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)
    logger.info(f"Model saved to {filepath}")


def load_model(filepath: str):
    """
    Load a trained model from disk.

    Args:
        filepath: Path to the saved model

    Returns:
        Loaded model
    """
    model = joblib.load(filepath)
    logger.info(f"Model loaded from {filepath}")
    return model


if __name__ == "__main__":
    from data_processing import prepare_data
    from feature_engineering import create_preprocessing_pipeline

    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data()

    # Preprocess
    preprocessor = create_preprocessing_pipeline()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Train models
    for model_name in ["logistic_regression", "random_forest", "xgboost"]:
        print(f"\nTraining {model_name}...")
        model, metrics = train_with_mlflow(
            model_name,
            X_train_processed,
            y_train,
            X_test_processed,
            y_test,
            run_name=f"{model_name}_baseline",
        )
        print(f"Metrics: {metrics}")
