"""
Heart Disease Prediction API
============================
FastAPI application for serving heart disease predictions.

Features:
- /predict endpoint for predictions
- /health endpoint for health checks
- /features endpoint for feature information
- Prometheus metrics integration
- Structured logging
"""

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.responses import Response

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.schemas import (
    ErrorResponse,
    FeatureInfo,
    FeaturesResponse,
    HealthResponse,
    PatientData,
    PredictionResponse,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Prometheus metrics
PREDICTION_COUNTER = Counter(
    "heart_disease_predictions_total",
    "Total number of predictions made",
    ["result", "risk_level"],
)
PREDICTION_LATENCY = Histogram(
    "heart_disease_prediction_latency_seconds", "Prediction latency in seconds"
)
REQUEST_COUNTER = Counter(
    "heart_disease_requests_total",
    "Total number of API requests",
    ["endpoint", "method", "status"],
)

# Feature names (must match training)
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
    "age": ("Age in years", "int", "1-120"),
    "sex": ("Sex (0=female, 1=male)", "int", "0-1"),
    "cp": ("Chest pain type", "int", "1-4"),
    "trestbps": ("Resting blood pressure (mm Hg)", "float", "50-250"),
    "chol": ("Serum cholesterol (mg/dl)", "float", "100-600"),
    "fbs": ("Fasting blood sugar > 120 mg/dl", "int", "0-1"),
    "restecg": ("Resting ECG results", "int", "0-2"),
    "thalach": ("Maximum heart rate achieved", "float", "60-220"),
    "exang": ("Exercise induced angina", "int", "0-1"),
    "oldpeak": ("ST depression induced by exercise", "float", "0-10"),
    "slope": ("Slope of peak exercise ST segment", "int", "1-3"),
    "ca": ("Number of major vessels (0-3)", "int", "0-3"),
    "thal": ("Thalassemia", "int", "3, 6, or 7"),
}

# Global model and preprocessor
model = None
preprocessor = None


def load_model_and_preprocessor():
    """Load model and preprocessor from disk."""
    global model, preprocessor

    # Determine model paths
    base_path = Path(__file__).parent.parent
    model_path = os.environ.get("MODEL_PATH", str(base_path / "models" / "model.pkl"))
    preprocessor_path = os.environ.get(
        "PREPROCESSOR_PATH", str(base_path / "models" / "preprocessor.pkl")
    )

    try:
        if Path(model_path).exists():
            model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
        else:
            logger.warning(f"Model file not found at {model_path}")

        if Path(preprocessor_path).exists():
            preprocessor = joblib.load(preprocessor_path)
            logger.info(f"Preprocessor loaded from {preprocessor_path}")
        else:
            logger.warning(f"Preprocessor file not found at {preprocessor_path}")

    except Exception as e:
        logger.error(f"Error loading model: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    # Startup
    logger.info("Starting Heart Disease Prediction API...")
    load_model_and_preprocessor()
    logger.info("API startup complete")
    yield
    # Shutdown
    logger.info("Shutting down API...")


# Create FastAPI app
app = FastAPI(
    title="Heart Disease Prediction API",
    description="""
    API for predicting heart disease risk based on patient health data.
    
    ## Features
    - **Prediction**: Get heart disease risk prediction with confidence scores
    - **Health Check**: Monitor API health status
    - **Feature Info**: Get information about required input features
    - **Metrics**: Prometheus metrics endpoint for monitoring
    
    ## Usage
    Send a POST request to `/predict` with patient data to get a prediction.
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware to log all requests."""
    start_time = time.time()

    # Process request
    response = await call_next(request)

    # Calculate latency
    latency = time.time() - start_time

    # Log request
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Latency: {latency:.3f}s"
    )

    # Update metrics
    REQUEST_COUNTER.labels(
        endpoint=request.url.path, method=request.method, status=response.status_code
    ).inc()

    return response


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint - redirect to docs."""
    return {"message": "Heart Disease Prediction API", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.

    Returns the current health status of the API and model.
    """
    return HealthResponse(
        status="healthy", model_loaded=model is not None, version="1.0.0"
    )


@app.get("/features", response_model=FeaturesResponse, tags=["Info"])
async def get_features():
    """
    Get information about required input features.

    Returns descriptions, types, and valid ranges for all features.
    """
    features = []
    for name in FEATURE_NAMES:
        desc, dtype, range_str = FEATURE_DESCRIPTIONS[name]
        features.append(
            FeatureInfo(name=name, description=desc, type=dtype, range=range_str)
        )

    return FeaturesResponse(features=features, total_features=len(features))


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Prediction error"},
    },
    tags=["Prediction"],
)
async def predict(patient_data: PatientData):
    """
    Make a heart disease prediction.

    Accepts patient health data and returns a prediction with confidence score.

    ## Input Features
    - **age**: Age in years (1-120)
    - **sex**: Sex (0 = female, 1 = male)
    - **cp**: Chest pain type (1-4)
    - **trestbps**: Resting blood pressure (mm Hg)
    - **chol**: Serum cholesterol (mg/dl)
    - **fbs**: Fasting blood sugar > 120 mg/dl (0/1)
    - **restecg**: Resting ECG results (0-2)
    - **thalach**: Maximum heart rate achieved
    - **exang**: Exercise induced angina (0/1)
    - **oldpeak**: ST depression induced by exercise
    - **slope**: Slope of peak exercise ST segment (1-3)
    - **ca**: Number of major vessels colored by fluoroscopy (0-3)
    - **thal**: Thalassemia (3=normal, 6=fixed defect, 7=reversible defect)

    ## Returns
    - **prediction**: 0 (No Disease) or 1 (Disease)
    - **confidence**: Model confidence score (0-1)
    - **risk_level**: Low, Moderate, High, or Very High
    - **probabilities**: Probability for each class
    """
    if model is None:
        raise HTTPException(
            status_code=500, detail="Model not loaded. Please check server logs."
        )

    start_time = time.time()

    try:
        # Convert input to DataFrame
        input_dict = patient_data.model_dump()
        df = pd.DataFrame([input_dict])
        df = df[FEATURE_NAMES]  # Ensure correct column order

        # Preprocess
        if preprocessor is not None:
            X = preprocessor.transform(df)
        else:
            X = df.values

        # Predict
        prediction = int(model.predict(X)[0])
        probabilities = model.predict_proba(X)[0]

        # Determine risk level
        disease_prob = float(probabilities[1])
        if disease_prob < 0.3:
            risk_level = "Low"
        elif disease_prob < 0.5:
            risk_level = "Moderate"
        elif disease_prob < 0.7:
            risk_level = "High"
        else:
            risk_level = "Very High"

        # Log prediction
        logger.info(
            f"Prediction: {prediction}, "
            f"Confidence: {max(probabilities):.3f}, "
            f"Risk: {risk_level}"
        )

        # Update metrics
        latency = time.time() - start_time
        PREDICTION_LATENCY.observe(latency)
        PREDICTION_COUNTER.labels(
            result="disease" if prediction == 1 else "no_disease",
            risk_level=risk_level.lower().replace(" ", "_"),
        ).inc()

        return PredictionResponse(
            success=True,
            prediction=prediction,
            prediction_label="Heart Disease" if prediction == 1 else "No Heart Disease",
            confidence=float(max(probabilities)),
            risk_level=risk_level,
            probabilities={
                "no_disease": float(probabilities[0]),
                "disease": float(probabilities[1]),
            },
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """
    Prometheus metrics endpoint.

    Returns metrics in Prometheus format for scraping.
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "details": [str(exc)],
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
