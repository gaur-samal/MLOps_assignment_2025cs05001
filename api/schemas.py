"""
Pydantic Schemas for Heart Disease Prediction API
==================================================
Defines request and response models for the FastAPI application.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class ChestPainType(int, Enum):
    """Chest pain type enumeration."""

    TYPICAL_ANGINA = 1
    ATYPICAL_ANGINA = 2
    NON_ANGINAL = 3
    ASYMPTOMATIC = 4


class RestingECG(int, Enum):
    """Resting ECG results."""

    NORMAL = 0
    ST_T_ABNORMALITY = 1
    LEFT_VENTRICULAR_HYPERTROPHY = 2


class Slope(int, Enum):
    """Slope of peak exercise ST segment."""

    UPSLOPING = 1
    FLAT = 2
    DOWNSLOPING = 3


class Thalassemia(int, Enum):
    """Thalassemia types."""

    NORMAL = 3
    FIXED_DEFECT = 6
    REVERSIBLE_DEFECT = 7


class PatientData(BaseModel):
    """
    Input schema for patient health data.
    All features required for heart disease prediction.
    """

    age: int = Field(..., ge=1, le=120, description="Age in years (1-120)")
    sex: int = Field(..., ge=0, le=1, description="Sex (0 = female, 1 = male)")
    cp: int = Field(..., ge=1, le=4, description="Chest pain type (1-4)")
    trestbps: float = Field(
        ..., ge=50, le=250, description="Resting blood pressure (mm Hg)"
    )
    chol: float = Field(..., ge=100, le=600, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(
        ..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (0=false, 1=true)"
    )
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results (0-2)")
    thalach: float = Field(
        ..., ge=60, le=220, description="Maximum heart rate achieved"
    )
    exang: int = Field(
        ..., ge=0, le=1, description="Exercise induced angina (0=no, 1=yes)"
    )
    oldpeak: float = Field(
        ..., ge=0, le=10, description="ST depression induced by exercise"
    )
    slope: int = Field(
        ..., ge=1, le=3, description="Slope of peak exercise ST segment (1-3)"
    )
    ca: int = Field(
        ...,
        ge=0,
        le=3,
        description="Number of major vessels colored by fluoroscopy (0-3)",
    )
    thal: int = Field(
        ..., description="Thalassemia (3=normal, 6=fixed defect, 7=reversible defect)"
    )

    @validator("thal")
    def validate_thal(cls, v):
        if v not in [3, 6, 7]:
            raise ValueError("thal must be 3, 6, or 7")
        return v

    class Config:
        json_schema_extra = {
            "example": {
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
        }


class PredictionResponse(BaseModel):
    """
    Response schema for prediction results.
    """

    success: bool = Field(..., description="Whether the prediction was successful")
    prediction: int = Field(..., description="Prediction (0 = No Disease, 1 = Disease)")
    prediction_label: str = Field(..., description="Human-readable prediction label")
    confidence: float = Field(..., ge=0, le=1, description="Model confidence score")
    risk_level: str = Field(
        ..., description="Risk level (Low, Moderate, High, Very High)"
    )
    probabilities: Dict[str, float] = Field(..., description="Class probabilities")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "prediction": 1,
                "prediction_label": "Heart Disease",
                "confidence": 0.85,
                "risk_level": "Very High",
                "probabilities": {"no_disease": 0.15, "disease": 0.85},
            }
        }


class ErrorResponse(BaseModel):
    """
    Response schema for errors.
    """

    success: bool = Field(default=False, description="Always False for errors")
    error: str = Field(..., description="Error message")
    details: Optional[List[str]] = Field(None, description="Detailed error messages")

    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "error": "Validation error",
                "details": ["age must be between 1 and 120"],
            }
        }


class HealthResponse(BaseModel):
    """
    Response schema for health check endpoint.
    """

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    version: str = Field(..., description="API version")

    class Config:
        json_schema_extra = {
            "example": {"status": "healthy", "model_loaded": True, "version": "1.0.0"}
        }


class FeatureInfo(BaseModel):
    """
    Information about input features.
    """

    name: str
    description: str
    type: str
    range: Optional[str] = None


class FeaturesResponse(BaseModel):
    """
    Response schema for feature information endpoint.
    """

    features: List[FeatureInfo]
    total_features: int
