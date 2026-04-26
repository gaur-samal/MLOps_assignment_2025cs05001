# Heart Disease Prediction - MLOps Project

[![CI/CD Pipeline](https://github.com/gaur-samal/MLOps_assignment_2025cs05001/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/gaur-samal/MLOps_assignment_2025cs05001/actions/workflows/ci-cd.yml)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready MLOps solution for heart disease prediction using machine learning. This project demonstrates end-to-end ML model development, CI/CD pipelines, containerization, and cloud deployment.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Project Structure](#project-structure)
- [Development](#development)
- [Deployment](#deployment)
- [Monitoring](#monitoring)
- [Contributing](#contributing)

## 🎯 Overview

This project predicts the risk of heart disease based on patient health data from the UCI Heart Disease dataset. It implements modern MLOps best practices including:

- **Automated ML Pipeline**: Data processing, model training, and evaluation
- **Experiment Tracking**: MLflow integration for tracking experiments
- **CI/CD Pipeline**: GitHub Actions for automated testing and deployment
- **Containerization**: Docker for consistent deployments
- **Kubernetes Deployment**: Scalable production deployment
- **Monitoring**: Prometheus + Grafana for observability

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔬 **Multiple ML Models** | Logistic Regression, Random Forest, XGBoost |
| 📊 **Experiment Tracking** | MLflow for parameters, metrics, and artifacts |
| 🐳 **Docker Support** | Multi-stage builds for production |
| ☸️ **Kubernetes Ready** | Deployment manifests and Helm charts |
| 🔄 **CI/CD Pipeline** | GitHub Actions with linting, testing, and deployment |
| 📈 **Monitoring** | Prometheus metrics and Grafana dashboards |
| 🧪 **Testing** | Comprehensive unit tests with pytest |
| 📖 **Documentation** | OpenAPI/Swagger for API docs |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CI/CD Pipeline                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │   Lint   │→│   Test   │→│  Train   │→│  Build & Deploy  │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     Kubernetes Cluster                           │
│  ┌─────────────────┐     ┌─────────────────┐                    │
│  │   Ingress       │────→│   Service       │                    │
│  └─────────────────┘     └────────┬────────┘                    │
│                                   ↓                              │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    Deployment (Pods)                         ││
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          ││
│  │  │   API Pod   │  │   API Pod   │  │   API Pod   │          ││
│  │  └─────────────┘  └─────────────┘  └─────────────┘          ││
│  └─────────────────────────────────────────────────────────────┘│
│                                   ↓                              │
│  ┌─────────────────┐     ┌─────────────────┐                    │
│  │   Prometheus    │────→│    Grafana      │                    │
│  └─────────────────┘     └─────────────────┘                    │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Using Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/gaur-samal/MLOps_assignment_2025cs05001.git
cd MLOps_assignment_2025cs05001

# Build and run with Docker Compose
docker-compose up -d

# Access the API
curl http://localhost:8000/health
```

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the API
cd api && uvicorn app:app --reload
```

## 📦 Installation

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- kubectl (for Kubernetes deployment)

### Step-by-Step Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/gaur-samal/MLOps_assignment_2025cs05001.git
   cd MLOps_assignment_2025cs05001
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download dataset**
   ```bash
   chmod +x download_data.sh
   ./download_data.sh
   ```

5. **Run EDA and train models** (optional)
   ```bash
   # Open Jupyter notebooks
   jupyter notebook notebooks/
   ```

## 📖 Usage

### Training a Model

```bash
# Run the training script
python -m src.model

# Or use the notebooks
jupyter notebook notebooks/02_model_training.ipynb
```

### Making Predictions

#### Using the API

```bash
# Start the API
uvicorn api.app:app --host 0.0.0.0 --port 8000

# Make a prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233,
    "fbs": 1, "restecg": 2, "thalach": 150, "exang": 0,
    "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6
  }'
```

#### Using Python

```python
from src.predict import HeartDiseasePredictor

predictor = HeartDiseasePredictor(
    model_path="models/model.pkl",
    preprocessor_path="models/preprocessor.pkl"
)

result = predictor.predict({
    "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233,
    "fbs": 1, "restecg": 2, "thalach": 150, "exang": 0,
    "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6
})

print(result)
# {'prediction': 1, 'prediction_label': 'Heart Disease', 'confidence': 0.85, ...}
```

## 📚 API Documentation

Once the API is running, access the interactive documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/features` | Get feature descriptions |
| POST | `/predict` | Make a prediction |
| GET | `/metrics` | Prometheus metrics |

## 📁 Project Structure

```
MLOps_assignment_2025cs05001/
├── .github/
│   └── workflows/
│       └── ci-cd.yml           # GitHub Actions CI/CD pipeline
├── api/
│   ├── __init__.py
│   ├── app.py                  # FastAPI application
│   └── schemas.py              # Pydantic models
├── data/
│   └── heart_disease_cleaned.csv
├── deployment/
│   ├── kubernetes/
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   ├── ingress.yaml
│   │   └── configmap.yaml
│   └── monitoring/
│       ├── prometheus.yml
│       └── grafana/
├── heart_disease_data/         # Raw dataset
├── models/
│   ├── model.pkl
│   └── preprocessor.pkl
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_inference.ipynb
├── screenshots/                # Deployment screenshots
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── model.py
│   └── predict.py
├── tests/
│   ├── test_api.py
│   ├── test_data_processing.py
│   └── test_model.py
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── download_data.sh
├── README.md
├── REPORT.md
├── requirements.txt
└── setup.py
```

## 🛠️ Development

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov=api --cov-report=html

# Run specific test file
pytest tests/test_api.py -v
```

### Code Quality

```bash
# Format code
black src/ api/ tests/

# Sort imports
isort src/ api/ tests/

# Lint code
flake8 src/ api/ tests/
```

### MLflow Experiment Tracking

```bash
# Start MLflow UI
mlflow ui --backend-store-uri file:./mlruns --port 5000

# Open http://localhost:5000
```

## 🚢 Deployment

### Docker Deployment

```bash
# Build the image
docker build -t heart-disease-api:latest .

# Run the container
docker run -d -p 8000:8000 heart-disease-api:latest

# Or use Docker Compose
docker-compose up -d
```

### Kubernetes Deployment

```bash
# Apply all manifests
kubectl apply -f deployment/kubernetes/

# Check deployment status
kubectl get pods -l app=heart-disease-api
kubectl get services

# Access the service
kubectl port-forward svc/heart-disease-api 8000:80
```

### Minikube (Local Kubernetes)

```bash
# Start Minikube
minikube start

# Build image in Minikube
eval $(minikube docker-env)
docker build -t heart-disease-api:latest .

# Deploy
kubectl apply -f deployment/kubernetes/

# Get service URL
minikube service heart-disease-api --url
```

## 📊 Monitoring

### Prometheus Metrics

Access metrics at: http://localhost:8000/metrics

Available metrics:
- `heart_disease_predictions_total` - Total predictions by result
- `heart_disease_prediction_latency_seconds` - Prediction latency
- `heart_disease_requests_total` - Total API requests

### Grafana Dashboard

1. Start services: `docker-compose up -d`
2. Access Grafana: http://localhost:3000
3. Login: admin / admin
4. Create dashboards using Prometheus data source

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the Heart Disease dataset
- Cleveland Clinic Foundation for data collection
- Principal Investigators: Robert Detrano, M.D., Ph.D.

## 📞 Support

For questions or issues, please open a GitHub issue or contact the maintainers.

---

*Built with ❤️ for MLOps learning*
