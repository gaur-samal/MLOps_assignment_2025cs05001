"""
Prediction Module
=================
Handles model inference and prediction utilities.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Feature definitions (must match training)
FEATURE_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

FEATURE_DESCRIPTIONS = {
    "age": "Age in years",
    "sex": "Sex (1 = male; 0 = female)",
    "cp": "Chest pain type (1-4)",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)",
    "restecg": "Resting ECG results (0-2)",
    "thalach": "Maximum heart rate achieved",
    "exang": "Exercise induced angina (1 = yes; 0 = no)",
    "oldpeak": "ST depression induced by exercise",
    "slope": "Slope of peak exercise ST segment (1-3)",
    "ca": "Number of major vessels (0-3) colored by fluoroscopy",
    "thal": "Thalassemia (3 = normal; 6 = fixed defect; 7 = reversible defect)",
}


class HeartDiseasePredictor:
    """
    Heart Disease Prediction class for inference.
    """

    def __init__(self, model_path: str = None, preprocessor_path: str = None):
        """
        Initialize the predictor.

        Args:
            model_path: Path to the saved model
            preprocessor_path: Path to the saved preprocessor
        """
        self.model = None
        self.preprocessor = None
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path

        if model_path:
            self.load_model(model_path)
        if preprocessor_path:
            self.load_preprocessor(preprocessor_path)

    def load_model(self, model_path: str) -> None:
        """Load the trained model."""
        self.model = joblib.load(model_path)
        self.model_path = model_path
        logger.info(f"Model loaded from {model_path}")

    def load_preprocessor(self, preprocessor_path: str) -> None:
        """Load the preprocessor."""
        self.preprocessor = joblib.load(preprocessor_path)
        self.preprocessor_path = preprocessor_path
        logger.info(f"Preprocessor loaded from {preprocessor_path}")

    def validate_input(self, data: Dict[str, Any]) -> List[str]:
        """
        Validate input data.

        Args:
            data: Input data dictionary

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check for required features
        for feature in FEATURE_NAMES:
            if feature not in data:
                errors.append(f"Missing required feature: {feature}")

        # Validate ranges
        if "age" in data and (data["age"] < 0 or data["age"] > 120):
            errors.append("Age must be between 0 and 120")

        if "sex" in data and data["sex"] not in [0, 1]:
            errors.append("Sex must be 0 or 1")

        if "cp" in data and data["cp"] not in [1, 2, 3, 4]:
            errors.append("Chest pain type (cp) must be 1, 2, 3, or 4")

        if "trestbps" in data and (data["trestbps"] < 50 or data["trestbps"] > 250):
            errors.append("Resting blood pressure must be between 50 and 250")

        if "chol" in data and (data["chol"] < 100 or data["chol"] > 600):
            errors.append("Cholesterol must be between 100 and 600")

        if "fbs" in data and data["fbs"] not in [0, 1]:
            errors.append("Fasting blood sugar (fbs) must be 0 or 1")

        if "restecg" in data and data["restecg"] not in [0, 1, 2]:
            errors.append("Resting ECG (restecg) must be 0, 1, or 2")

        if "thalach" in data and (data["thalach"] < 60 or data["thalach"] > 220):
            errors.append("Maximum heart rate must be between 60 and 220")

        if "exang" in data and data["exang"] not in [0, 1]:
            errors.append("Exercise induced angina (exang) must be 0 or 1")

        if "oldpeak" in data and (data["oldpeak"] < 0 or data["oldpeak"] > 10):
            errors.append("ST depression (oldpeak) must be between 0 and 10")

        if "slope" in data and data["slope"] not in [1, 2, 3]:
            errors.append("Slope must be 1, 2, or 3")

        if "ca" in data and data["ca"] not in [0, 1, 2, 3]:
            errors.append("Number of vessels (ca) must be 0, 1, 2, or 3")

        if "thal" in data and data["thal"] not in [3, 6, 7]:
            errors.append("Thalassemia (thal) must be 3, 6, or 7")

        return errors

    def preprocess(self, data: Union[Dict, pd.DataFrame]) -> np.ndarray:
        """
        Preprocess input data for prediction.

        Args:
            data: Input data (dict or DataFrame)

        Returns:
            Preprocessed numpy array
        """
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()

        # Ensure correct column order
        df = df[FEATURE_NAMES]

        if self.preprocessor:
            return self.preprocessor.transform(df)
        else:
            return df.values

    def predict(self, data: Union[Dict, pd.DataFrame]) -> Dict[str, Any]:
        """
        Make a prediction.

        Args:
            data: Input data (dict or DataFrame)

        Returns:
            Prediction result dictionary
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Validate input
        if isinstance(data, dict):
            errors = self.validate_input(data)
            if errors:
                return {"success": False, "errors": errors}

        # Preprocess
        X = self.preprocess(data)

        # Predict
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0]

        # Format result
        result = {
            "success": True,
            "prediction": int(prediction),
            "prediction_label": (
                "Heart Disease" if prediction == 1 else "No Heart Disease"
            ),
            "confidence": float(max(probability)),
            "probabilities": {
                "no_disease": float(probability[0]),
                "disease": float(probability[1]),
            },
        }

        # Add risk level
        disease_prob = probability[1]
        if disease_prob < 0.3:
            result["risk_level"] = "Low"
        elif disease_prob < 0.5:
            result["risk_level"] = "Moderate"
        elif disease_prob < 0.7:
            result["risk_level"] = "High"
        else:
            result["risk_level"] = "Very High"

        return result

    def predict_batch(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Make predictions on multiple samples.

        Args:
            data: DataFrame with multiple samples

        Returns:
            List of prediction results
        """
        results = []
        for idx, row in data.iterrows():
            result = self.predict(row.to_dict())
            result["index"] = idx
            results.append(result)
        return results


def create_sample_input() -> Dict[str, Any]:
    """
    Create a sample input for testing.

    Returns:
        Sample input dictionary
    """
    return {
        "age": 63,
        "sex": 1,
        "cp": 1,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 2,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 3,
        "ca": 0,
        "thal": 6,
    }


def get_feature_info() -> Dict[str, str]:
    """
    Get feature descriptions for API documentation.

    Returns:
        Dictionary of feature descriptions
    """
    return FEATURE_DESCRIPTIONS


if __name__ == "__main__":
    # Test prediction with sample data
    print("Testing HeartDiseasePredictor...")

    sample = create_sample_input()
    print(f"\nSample input: {sample}")

    # Note: This will fail without a trained model
    try:
        predictor = HeartDiseasePredictor(
            model_path="models/model.pkl", preprocessor_path="models/preprocessor.pkl"
        )
        result = predictor.predict(sample)
        print(f"\nPrediction result: {result}")
    except FileNotFoundError:
        print("\nModel files not found. Train the model first.")
