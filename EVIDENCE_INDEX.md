# Evidence Index (Rubric-Mapped)

This file maps rubric requirements to concrete repository evidence.

## M1 - Model Development & Experiment Tracking

- Git structure and versioning: `.git/`, `.github/`, `src/`, `tests/`
- Data versioning metadata:
  - `.dvc/`, `.dvc/config`, `.dvcignore`, `dvc.yaml`
  - `data/raw.dvc`, `data/processed.dvc`
  - Git-LFS tracking rules: `.gitattributes`
- Preprocessing and model:
  - 224x224 RGB + augmentations: `src/data/preprocessing.py`
  - 80/10/10 split parameters: `params.yaml`, `scripts/prepare_data.py`
  - Baseline models: `src/models/cnn_model.py`
  - Saved model artifacts: `models/model.pt`, `models/best_model.pt`
- MLflow tracking:
  - Tracking code: `src/models/train.py`
  - Local run artifacts: `mlruns/`

## M2 - Packaging & Containerization

- Inference API (FastAPI):
  - Health endpoint: `GET /health` in `src/api/main.py`
  - Prediction endpoint: `POST /predict` in `src/api/main.py`
- Environment specification:
  - Pinned dependencies: `requirements.txt`
- Docker:
  - Build/runtime config: `Dockerfile`
  - Local orchestration: `docker-compose.yml`
  - Deployment compose with registry image pull: `docker-compose.prod.yml`

## M3 - CI Pipeline

- CI configuration: `.github/workflows/ci-cd.yml`
- Automated tests:
  - Preprocessing tests: `tests/test_preprocessing.py`
  - Model tests: `tests/test_model.py`
  - API tests: `tests/test_api.py`
- CI build/test/publish flow:
  - Checkout + dependency install + pytest + docker build + push to GHCR
  - Image publication evidence artifact in workflow (`image-publication-evidence`)

## M4 - CD & Deployment

- Deployment targets:
  - Kubernetes manifests: `kubernetes/deployment.yaml`, `kubernetes/configmap.yaml`
  - Compose-based deployment: `docker-compose.prod.yml`
- Automatic CD on branch:
  - `deploy-staging` for `develop`
  - `deploy-production` for `main`
  - Workflow file: `.github/workflows/ci-cd.yml`
- Smoke tests with fail-fast behavior:
  - Script: `scripts/smoke_test.py`
  - Executed in deployment jobs in `.github/workflows/ci-cd.yml`

## M5 - Monitoring, Logs & Final Submission

- Monitoring/logging:
  - Structured request/response logging: `src/api/main.py`
  - Prometheus metrics endpoint and counters: `src/api/main.py`
  - Prometheus scrape config: `prometheus.yml`
- Post-deployment performance tracking:
  - Monitoring script (with simulated truth labels + drift): `scripts/monitor_performance.py`
  - Drift and metrics modules: `src/monitoring/metrics.py`, `src/monitoring/drift_detector.py`
  - CI artifact upload for monitoring evidence in deploy job
- Final deliverables support:
  - Submission checklist: `SUBMISSION_CHECKLIST.md`
  - Zip bundling script: `scripts/create_submission_bundle.py`
