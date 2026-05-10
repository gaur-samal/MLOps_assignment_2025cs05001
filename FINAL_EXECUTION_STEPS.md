# Final Execution Steps for MLOps Assignment Submission

**BITS ID**: 2025cs05001  
**Course**: MLOps (S2-25_AMLCSZG523)  
**Project**: Heart Disease Prediction - End-to-End MLOps Solution

---

## Overview

This document contains ALL the commands needed to set up and run the complete MLOps pipeline for final submission. Execute these in order.

---

## PART 1: One-Time Setup (Do Once)

### 1.1 Install Dependencies

```bash
# Create Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install all Python dependencies
cd /Users/gsamal/Downloads/Mlops_assignment
pip install -r requirements.txt
```

### 1.2 Download Dataset

```bash
# Make download script executable
chmod +x download_data.sh

# Download heart disease dataset
./download_data.sh

# Verify dataset
ls -la heart_disease_data/processed.cleveland.data
```

### 1.3 Train and Save Models

```bash
# Run Jupyter notebook (or run offline if already trained)
# Models will be saved to: models/model.pkl, models/preprocessor.pkl

# Option A: If models already exist, skip this
ls -la models/model.pkl models/preprocessor.pkl

# Option B: If needed, retrain via notebook
jupyter notebook notebooks/02_model_training.ipynb
```

---

## PART 2: Full Stack Startup (Do Every Time)

Execute these steps in order to have everything running for the video recording.

### 2.1 Start Docker Desktop

```bash
# IMPORTANT: Clear any Minikube environment variables from previous sessions
unset DOCKER_HOST
unset DOCKER_CERT_PATH
unset DOCKER_TLS_VERIFY
unset DOCKER_MACHINE_NAME

# Ensure Docker Desktop is running
open -a Docker

# Wait 10 seconds for Docker to fully start
sleep 10

# Verify Docker is running (should show containers, not error)
docker ps
```

> **Note**: If you get `Cannot connect to the Docker daemon`, it means Minikube environment is still active. Run the `unset` commands above, then try `docker ps` again.

### 2.2 Start Minikube (Kubernetes)

```bash
# Start Minikube with Docker driver
minikube start --driver=docker

# Wait for Minikube to be ready
minikube status

# Set Docker environment to use Minikube's Docker daemon
eval $(minikube docker-env)
```

### 2.3 Build Docker Image (Inside Minikube)

```bash
# Navigate to project
cd /Users/gsamal/Downloads/Mlops_assignment

# Build image (this will be built inside Minikube's Docker)
docker build -t heart-disease-api:latest .

# Verify build succeeded
docker images | grep heart-disease-api
```

### 2.4 Deploy to Kubernetes

```bash
# Apply all Kubernetes manifests
kubectl apply -f deployment/kubernetes/

# Wait for pods to start (30 seconds)
sleep 30

# Verify deployment
kubectl get pods -l app=heart-disease-api
kubectl get services
kubectl get deployments
kubectl get hpa
```

### 2.5 Start Monitoring Stack (Docker Compose)

```bash
# Make sure we're using the regular Docker (not Minikube)
unset DOCKER_HOST
unset DOCKER_CERT_PATH
unset DOCKER_TLS_VERIFY
unset DOCKER_MACHINE_NAME

# Start monitoring services
cd /Users/gsamal/Downloads/Mlops_assignment
docker-compose up -d

# Wait for services to start (20 seconds)
sleep 20

# Verify all services running
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

### 2.6 Verify All Services Are Accessible

```bash
# Test API health
curl http://localhost:8000/health

# Test Prometheus
curl http://localhost:9090/api/v1/targets

# Generate sample predictions for metrics
for i in {1..5}; do
  curl -s http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6}' > /dev/null
done

# Verify services in browser
echo "
✅ Services should be accessible at:
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- MLflow: http://localhost:5001
- Prometheus: http://localhost:9090/targets
- Grafana: http://localhost:3000 (admin/admin)
- Kubernetes API: $(minikube ip):6443
"
```

## PART 4: Stop All Services (When Done)

```bash
# Stop Docker Compose services
cd /Users/gsamal/Downloads/Mlops_assignment
docker-compose down

# Stop Kubernetes deployment
kubectl delete -f deployment/kubernetes/

# Stop Minikube (optional, but saves resources)
minikube stop

# Verify everything stopped
docker ps
kubectl get pods
```

---

## PART 5: Final Checklist

Before submitting, verify you have:

### Code & Documentation ✅
- [ ] `REPORT.md` - Comprehensive 13-section report
- [ ] `REPORT.pdf` - PDF version of report
- [ ] `README.md` - Project overview
- [ ] `requirements.txt` - Dependencies with pinned versions
- [ ] `.github/workflows/ci-cd.yml` - GitHub Actions workflow

### Data & Models ✅
- [ ] `heart_disease_data/processed.cleveland.data` - Dataset (303 samples)
- [ ] `models/model.pkl` - Trained Random Forest model
- [ ] `models/preprocessor.pkl` - Fitted preprocessing pipeline
- [ ] `download_data.sh` - Data acquisition script

### Code Quality ✅
- [ ] `src/data_processing.py` - Data handling
- [ ] `src/model_training.py` - Model training logic
- [ ] `tests/` - 695+ lines of tests (>80% coverage)
- [ ] CI/CD: All GitHub Actions jobs passing

### Deployment ✅
- [ ] `api/app.py` - FastAPI application with all endpoints
- [ ] `deployment/kubernetes/` - K8s manifests (deployment, service, HPA, ingress)
- [ ] `deployment/monitoring/` - Prometheus & Grafana configs
- [ ] `Dockerfile` - Multi-stage Docker build
- [ ] `docker-compose.yml` - Monitoring stack

### Notebooks ✅
- [ ] `notebooks/01_eda.ipynb` - Exploratory data analysis
- [ ] `notebooks/02_model_training.ipynb` - Model training with MLflow
- [ ] `notebooks/03_inference.ipynb` - Prediction examples

---

## Troubleshooting

### Issue: "Cannot connect to the Docker daemon" (Docker Desktop is running)
```bash
# This means Minikube environment variables are still active
# Clear them immediately:
unset DOCKER_HOST
unset DOCKER_CERT_PATH
unset DOCKER_TLS_VERIFY
unset DOCKER_MACHINE_NAME

# Verify you're connected to Docker Desktop (not Minikube)
docker ps

# Check which Docker daemon you're using
docker version | grep "Server Version"
```

### Issue: Services won't start
```bash
# Kill lingering processes
pkill -f docker-compose
pkill -f minikube
pkill -f prometheus
pkill -f grafana

# Start fresh
docker-compose up -d
```

### Issue: Port conflicts
```bash
# Find process on port
lsof -i :8000  # API
lsof -i :9090  # Prometheus
lsof -i :3000  # Grafana

# Kill specific process
kill -9 <PID>
```

### Issue: Kubernetes pods not starting
```bash
# Check pod logs
kubectl logs -l app=heart-disease-api

# Describe pod for events
kubectl describe pods -l app=heart-disease-api

# Rebuild and redeploy
eval $(minikube docker-env)
docker build -t heart-disease-api:latest .
kubectl rollout restart deployment/heart-disease-api
```

### Issue: Model not loading in API
```bash
# Verify models exist
ls -la models/

# Check API logs
docker logs heart-disease-api

# Test model directly
python3 -c "import joblib; m = joblib.load('models/model.pkl'); print('Model loaded:', m)"
```

---

## Quick Reference

| Service | URL | Credentials |
|---------|-----|-------------|
| API | http://localhost:8000 | None |
| API Docs | http://localhost:8000/docs | None |
| MLflow | http://localhost:5001 | None |
| Prometheus | http://localhost:9090 | None |
| Grafana | http://localhost:3000 | admin/admin |

---

## Summary

**Total Setup Time**: ~15-20 minutes (first time)  
**Total Runtime Time**: ~5 minutes (recording only)  
**Services Running**: 5 (API, MLflow, Prometheus, Grafana, Kubernetes API)  
**Test Coverage**: >80% (25+ passing tests)  
**Model Performance**: 92% ROC-AUC, 87% Accuracy  

---
