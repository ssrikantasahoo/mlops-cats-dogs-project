# MLOps Pipeline: Cats vs Dogs Classification

## Project Overview
End-to-end MLOps pipeline for binary image classification (Cats vs Dogs) for a pet adoption platform.

## Project Structure
```
mlops-cats-dogs-project/
├── .github/workflows/      # CI/CD pipeline definitions
├── data/
│   ├── raw/               # Original dataset
│   └── processed/         # Preprocessed images
├── kubernetes/            # K8s deployment manifests
├── logs/                  # Application logs
├── models/                # Trained model artifacts
├── notebooks/             # Jupyter notebooks for exploration
├── scripts/               # Utility scripts
├── src/
│   ├── api/              # FastAPI inference service
│   ├── data/             # Data processing modules
│   ├── models/           # Model training code
│   └── monitoring/       # Monitoring utilities
├── tests/                 # Unit tests
├── Dockerfile            # Container definition
├── docker-compose.yml    # Local deployment
├── dvc.yaml              # DVC pipeline
├── requirements.txt      # Python dependencies
└── README.md
```

## Quick Start

### 1. Setup Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset
```bash
# Using Kaggle CLI
kaggle datasets download -d tongpython/cat-and-dog
unzip cat-and-dog.zip -d data/raw/
```

### 3. Initialize DVC
```bash
dvc init
dvc add data/raw
git add data/raw.dvc .gitignore
```

### 4. Train Model
```bash
python src/models/train.py
```

### 5. Run API Locally
```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 6. Build & Run Docker
```bash
docker build -t cats-dogs-classifier:latest .
docker run -p 8000:8000 cats-dogs-classifier:latest
```

### 7. Deploy to Kubernetes
```bash
kubectl apply -f kubernetes/
```

## API Endpoints
- `GET /health` - Health check
- `POST /predict` - Image classification prediction
- `GET /metrics` - Prometheus metrics

## CI/CD & Deployment

- CI/CD workflow: `.github/workflows/ci-cd.yml`
- Container registry target: `ghcr.io/<owner>/<repo>`
- Automatic deploy:
  - `develop` -> staging (`deploy-staging`)
  - `main` -> production (`deploy-production`)
- Post-deployment checks:
  - Smoke tests: `scripts/smoke_test.py`
  - Performance monitoring: `scripts/monitor_performance.py`

## MLflow Tracking
```bash
mlflow ui --port 5000
```

## Running Tests
```bash
pytest tests/ -v --cov=src
```

## Submission Packaging
```bash
python scripts/create_submission_bundle.py
```

See `EVIDENCE_INDEX.md` and `SUBMISSION_CHECKLIST.md` before final submission.

## Authors
MLOps Assignment - Binary Image Classification
