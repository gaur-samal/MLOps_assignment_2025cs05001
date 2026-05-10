# 🚀 Quick Start Checklist - MLOps Assignment Final Submission

**Status**: ✅ All systems ready for recording  
**Video Duration**: 3-5 minutes  
**Total Setup Time**: ~20 minutes

---

## ☑️ PRE-RECORDING CHECKLIST

### System Requirements
- [ ] macOS with 8GB+ RAM
- [ ] Docker Desktop installed and running
- [ ] Minikube installed
- [ ] kubectl installed
- [ ] Terminal ready (zoom to Cmd+)

### File Verification
```bash
cd /Users/gsamal/Downloads/Mlops_assignment

# Check critical files exist
ls -la models/model.pkl models/preprocessor.pkl
ls -la REPORT.md docker-compose.yml Dockerfile
ls -la deployment/kubernetes/deployment.yaml
ls -la .github/workflows/ci-cd.yml
```

- [ ] `models/model.pkl` exists
- [ ] `REPORT.md` exists
- [ ] `docker-compose.yml` exists
- [ ] All Kubernetes manifests exist
- [ ] `.github/workflows/ci-cd.yml` exists

---

## 🎬 EXECUTION STEPS (Copy-Paste Ready)

### Command Block 1: Start Infrastructure (20 minutes)

```bash
# 1. Start Docker
open -a Docker && sleep 10

# 2. Start Minikube
minikube start --driver=docker && sleep 10

# 3. Use Minikube Docker
eval $(minikube docker-env)

# 4. Navigate to project
cd /Users/gsamal/Downloads/Mlops_assignment

# 5. Build Docker image
docker build -t heart-disease-api:latest .

# 6. Deploy to Kubernetes
kubectl apply -f deployment/kubernetes/
sleep 30

# 7. Exit Minikube Docker environment
unset DOCKER_HOST DOCKER_CERT_PATH DOCKER_TLS_VERIFY DOCKER_MACHINE_NAME

# 8. Start monitoring stack
docker-compose up -d
sleep 20

# 9. Verify everything
docker ps
kubectl get pods -l app=heart-disease-api
kubectl get services
```

### Verification: All Green? ✅

```bash
# If all return data, you're good!
curl http://localhost:8000/health
curl http://localhost:9090/api/v1/targets
docker ps | wc -l  # Should show 6 lines
kubectl get pods | wc -l  # Should show 4 lines (1 header + 3 pods)
```

---

## 🎥 OPEN BROWSER TABS

Open these 5 tabs in THIS ORDER (before hitting record):

| # | Tab | URL |
|---|-----|-----|
| 1 | GitHub Repo | https://github.com/gaur-samal/MLOps_assignment_2025cs05001 |
| 2 | GitHub Actions | https://github.com/gaur-samal/MLOps_assignment_2025cs05001/actions |
| 3 | Swagger API | http://localhost:8000/docs |
| 4 | Prometheus | http://localhost:9090/targets |
| 5 | Grafana | http://localhost:3000 (admin/admin) |

---

## 🎬 RECORDING TIMELINE

| Time | Section | Show | Say |
|------|---------|------|-----|
| 0:00-0:30 | **INTRO** | VS Code/terminal | Name, BITS ID, what you'll demo |
| 0:30-1:00 | **Project** | `ls -la` in terminal | Explain folder structure |
| 1:00-1:45 | **CI/CD** | GitHub Actions tab | Show all 4 jobs passing ✅ |
| 1:45-2:45 | **API Demo** | Swagger UI | Make prediction, show result |
| 2:45-3:30 | **Kubernetes** | Terminal `kubectl get pods` | Show 2 replicas running, HPA |
| 3:30-4:15 | **Monitoring** | Prometheus + Grafana tabs | Show targets UP, metrics, dashboard |
| 4:15-4:30 | **OUTRO** | GitHub repo | Recap and thank you |

**Total: 4.5 minutes ✅**

---

## 🎙️ RECORDING COMMAND

```bash
# macOS built-in (Recommended)
Cmd + Shift + 5
# Then: Click Options → Select Microphone → Record Entire Screen

# Save as: MLOps_Demo_2025cs05001.mp4
```

---

## ✅ DURING RECORDING - Demo Commands

Have these ready to copy-paste:

**Show project structure:**
```bash
cd /Users/gsamal/Downloads/Mlops_assignment && ls -la
```

**Show API health:**
```bash
curl http://localhost:8000/health | python3 -m json.tool
```

**Show Kubernetes:**
```bash
kubectl get pods -l app=heart-disease-api
kubectl get services
kubectl get deployments
```

---

## 🛑 AFTER RECORDING - Cleanup

```bash
# Stop monitoring stack
cd /Users/gsamal/Downloads/Mlops_assignment
docker-compose down

# Stop Kubernetes
kubectl delete -f deployment/kubernetes/

# Stop Minikube (optional)
minikube stop

# Verify stopped
docker ps
kubectl get pods
```

---

## 📤 SUBMISSION - What to Include

### GitHub Repository (Already there!)
- ✅ All code in `/Users/gsamal/Downloads/Mlops_assignment`
- ✅ Push to GitHub: https://github.com/gaur-samal/MLOps_assignment_2025cs05001

### Video File
- ✅ Save: `MLOps_Demo_2025cs05001.mp4`
- ✅ Add to GitHub repo `/video/` folder
- ✅ OR share Google Drive/YouTube link in README

### All Deliverables Checklist

**Documentation** ✅
- [ ] `REPORT.md` (13 pages)
- [ ] `REPORT.pdf` (generated)
- [ ] `README.md` (overview)

**Code** ✅
- [ ] `api/app.py` (FastAPI)
- [ ] `src/data_processing.py`
- [ ] `src/model_training.py`
- [ ] `models/model.pkl` (trained)
- [ ] `models/preprocessor.pkl` (fitted)

**Tests** ✅
- [ ] `tests/` (695+ lines, 25+ tests)
- [ ] Coverage >80%
- [ ] All passing

**Deployment** ✅
- [ ] `Dockerfile` (multi-stage)
- [ ] `docker-compose.yml` (monitoring)
- [ ] `deployment/kubernetes/` (all manifests)
- [ ] `.github/workflows/ci-cd.yml` (4 jobs)

**Data & Notebooks** ✅
- [ ] `heart_disease_data/processed.cleveland.data`
- [ ] `download_data.sh` (download script)
- [ ] `notebooks/01_eda.ipynb`
- [ ] `notebooks/02_model_training.ipynb`
- [ ] `notebooks/03_inference.ipynb`

**Proof** ✅
- [ ] `screenshots/` (25+ images)
- [ ] Video demo (3-5 minutes)

---

## 🐛 Troubleshooting (Quick Fixes)

### Services won't start?
```bash
pkill -f docker-compose
pkill -f minikube
sleep 5
# Then repeat infrastructure setup
```

### Port conflicts?
```bash
lsof -i :8000  # Find process on port
kill -9 <PID>  # Kill it
```

### API not responding?
```bash
docker logs heart-disease-api
docker inspect heart-disease-api
```

### Kubernetes pods not running?
```bash
kubectl describe pods -l app=heart-disease-api
kubectl logs -l app=heart-disease-api
```

---

## 🎯 Success Indicators

### ✅ Green Lights to Look For

- [ ] Docker: `docker ps` shows 4 services (api, mlflow, prometheus, grafana)
- [ ] Kubernetes: `kubectl get pods` shows 2+ heart-disease-api pods
- [ ] API: `curl http://localhost:8000/health` returns 200
- [ ] Prometheus: `http://localhost:9090/targets` shows API target as UP
- [ ] Grafana: Can see metrics on dashboard
- [ ] CI/CD: All GitHub Actions jobs have ✅ green checkmarks

---

## 📊 Key Metrics for Recording

Mention these in your video:

| Metric | Value |
|--------|-------|
| Model Performance | 92% ROC-AUC |
| Accuracy | 87% |
| Test Coverage | >80% |
| Deployed Replicas | 2 (scalable to 10) |
| API Response Time | <100ms |
| Monitoring Interval | 10 seconds |
| CI/CD Runtime | ~10 minutes |

---

## 🆘 Need Help?

**Detailed instructions**: Read [FINAL_EXECUTION_STEPS.md](FINAL_EXECUTION_STEPS.md)  
**Recording guide**: See [docs/VIDEO_RECORDING_GUIDE.md](docs/VIDEO_RECORDING_GUIDE.md)  
**Full report**: Check [REPORT.md](REPORT.md)

---

## ⏱️ Time Breakdown

| Task | Duration | Ready? |
|------|----------|--------|
| Setup infrastructure | 15-20 min | ✅ |
| Recording | 4-5 min | ✅ |
| Post-processing | 5-10 min | ✅ |
| Cleanup | 2-3 min | ✅ |
| **Total** | **~35 minutes** | **✅** |

---

**You're all set!** 🚀 Follow the checklist and execute the command blocks in order.

Good luck with your recording! 🎬

---
