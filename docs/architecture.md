# MLOps Pipeline Architecture

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           MLOps Pipeline Architecture                            │
│                        Heart Disease Prediction System                           │
└─────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────────┐
│                              DATA LAYER                                          │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐             │
│   │   Raw Data      │    │  Data Cleaning  │    │  Feature        │             │
│   │   (CSV)         │───▶│  & Validation   │───▶│  Engineering    │             │
│   │   heart.csv     │    │  data_process.py│    │  feature_eng.py │             │
│   └─────────────────┘    └─────────────────┘    └─────────────────┘             │
│                                                          │                       │
│                                                          ▼                       │
│                                               ┌─────────────────┐               │
│                                               │  Processed Data │               │
│                                               │  (cleaned.csv)  │               │
│                                               └─────────────────┘               │
└──────────────────────────────────────────────────────────────────────────────────┘
                                                          │
                                                          ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                           MODEL TRAINING LAYER                                    │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐             │
│   │  Model Training │    │   Hyperparameter │    │    MLflow       │             │
│   │  - Logistic Reg │    │   Tuning         │    │    Tracking     │             │
│   │  - Random Forest│───▶│   GridSearchCV   │───▶│    - Params     │             │
│   │  - Gradient Boost│   │                   │    │    - Metrics    │             │
│   └─────────────────┘    └─────────────────┘    │    - Artifacts  │             │
│                                                  └────────┬────────┘             │
│                                                           │                      │
│                                                           ▼                      │
│                                               ┌─────────────────┐               │
│                                               │  Best Model     │               │
│                                               │  (model.pkl)    │               │
│                                               └─────────────────┘               │
└──────────────────────────────────────────────────────────────────────────────────┘
                                                          │
                                                          ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              API LAYER                                            │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                          FastAPI Application                             │   │
│   │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │   │
│   │   │  /health     │  │  /predict    │  │  /features   │  │  /metrics  │ │   │
│   │   │  Health Check│  │  Predictions │  │  Feature Info│  │  Prometheus│ │   │
│   │   └──────────────┘  └──────────────┘  └──────────────┘  └────────────┘ │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                   │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                      │
│   │   Pydantic   │    │   Uvicorn    │    │   Request    │                      │
│   │   Schemas    │    │   Server     │    │   Validation │                      │
│   └──────────────┘    └──────────────┘    └──────────────┘                      │
└──────────────────────────────────────────────────────────────────────────────────┘
                                                          │
                                                          ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                          CONTAINERIZATION LAYER                                   │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   ┌──────────────────────────────────────────────────────────────────────────┐  │
│   │                         Docker Container                                   │  │
│   │   ┌────────────────┐                                                      │  │
│   │   │  Python 3.12   │                                                      │  │
│   │   │  + Dependencies │                                                      │  │
│   │   └────────────────┘                                                      │  │
│   │   ┌────────────────┐  ┌────────────────┐  ┌────────────────┐             │  │
│   │   │  FastAPI App   │  │  Model.pkl     │  │  Preprocessor  │             │  │
│   │   │  (uvicorn)     │  │                │  │  .pkl          │             │  │
│   │   └────────────────┘  └────────────────┘  └────────────────┘             │  │
│   │                              Port: 8000                                    │  │
│   └──────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────────┘
                                                          │
                                                          ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                           KUBERNETES LAYER                                        │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                        Kubernetes Cluster                                │   │
│   │   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                 │   │
│   │   │  Deployment │───▶│   Service   │───▶│   Ingress   │                 │   │
│   │   │  (3 replicas)│    │  (ClusterIP)│    │  (Optional) │                 │   │
│   │   └─────────────┘    └─────────────┘    └─────────────┘                 │   │
│   │         │                                                                │   │
│   │         ▼                                                                │   │
│   │   ┌─────────────┐    ┌─────────────┐                                    │   │
│   │   │  ConfigMap  │    │   HPA       │                                    │   │
│   │   │  (env vars) │    │  (autoscale)│                                    │   │
│   │   └─────────────┘    └─────────────┘                                    │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────────┘
                                                          │
                                                          ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              CI/CD LAYER                                          │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                      GitHub Actions Pipeline                             │   │
│   │                                                                          │   │
│   │   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐            │   │
│   │   │   Lint   │──▶│   Test   │──▶│  Train   │──▶│  Build   │            │   │
│   │   │ (flake8) │   │ (pytest) │   │  Model   │   │  Docker  │            │   │
│   │   └──────────┘   └──────────┘   └──────────┘   └──────────┘            │   │
│   │                                                      │                   │   │
│   │                                                      ▼                   │   │
│   │                                               ┌──────────┐              │   │
│   │                                               │  Deploy  │              │   │
│   │                                               │  (GHCR)  │              │   │
│   │                                               └──────────┘              │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────────┘


## Component Details

### 1. Data Layer
- **Raw Data**: UCI Heart Disease dataset (303 records, 14 features)
- **Data Processing**: Cleaning, validation, handling missing values
- **Feature Engineering**: Standardization, encoding categorical features

### 2. Model Training Layer
- **Models**: Logistic Regression, Random Forest, Gradient Boosting
- **Hyperparameter Tuning**: GridSearchCV with cross-validation
- **Experiment Tracking**: MLflow for logging parameters, metrics, artifacts

### 3. API Layer
- **Framework**: FastAPI with automatic OpenAPI documentation
- **Endpoints**: Health check, prediction, feature info, Prometheus metrics
- **Validation**: Pydantic schemas for request/response validation

### 4. Containerization Layer
- **Docker**: Multi-stage build with Python 3.12-slim base
- **Artifacts**: Model, preprocessor, and API code packaged together
- **Port**: 8000 (configurable via environment variable)

### 5. Kubernetes Layer
- **Deployment**: 3 replicas for high availability
- **Service**: ClusterIP for internal communication
- **HPA**: Horizontal Pod Autoscaler (2-10 pods, 70% CPU threshold)

### 6. CI/CD Layer
- **Platform**: GitHub Actions
- **Stages**: Lint → Test → Train → Build → Deploy
- **Registry**: GitHub Container Registry (GHCR)


## Data Flow

```
User Request → Ingress → Service → Pod → FastAPI → Model → Response
                                            │
                                            ▼
                                    Prometheus Metrics
```

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.12 |
| ML Framework | scikit-learn |
| API Framework | FastAPI |
| Experiment Tracking | MLflow |
| Containerization | Docker |
| Orchestration | Kubernetes |
| CI/CD | GitHub Actions |
| Monitoring | Prometheus |
