# Makefile for Cats vs Dogs MLOps Project

.PHONY: help install test lint format train api docker-build docker-run deploy clean

# Default target
help:
	@echo "Available commands:"
	@echo "  make install      - Install dependencies"
	@echo "  make test         - Run tests"
	@echo "  make lint         - Run linter"
	@echo "  make format       - Format code"
	@echo "  make train        - Train model"
	@echo "  make api          - Run API server"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-run   - Run Docker container"
	@echo "  make deploy       - Deploy to Kubernetes"
	@echo "  make mlflow       - Start MLflow server"
	@echo "  make smoke-test   - Run smoke tests"
	@echo "  make clean        - Clean generated files"

# Install dependencies
install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .

# Run tests
test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

# Run linter
lint:
	flake8 src/ tests/ --max-line-length=100
	black --check src/ tests/
	isort --check-only src/ tests/

# Format code
format:
	black src/ tests/
	isort src/ tests/

# Prepare data
prepare-data:
	python scripts/prepare_data.py --download

# Train model
train:
	python src/models/train.py --data-dir data/processed --output-dir models --epochs 10

# Evaluate model
evaluate:
	python scripts/evaluate.py --model-path models/model.pt --data-dir data/processed

# Run API locally
api:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Start MLflow server
mlflow:
	mlflow ui --port 5000

# Build Docker image
docker-build:
	docker build -t cats-dogs-classifier:latest .

# Run Docker container
docker-run:
	docker run -p 8000:8000 -v $(PWD)/models:/app/models cats-dogs-classifier:latest

# Run with Docker Compose
docker-compose-up:
	docker-compose up -d

docker-compose-down:
	docker-compose down

# Deploy to Kubernetes
deploy:
	kubectl apply -f kubernetes/

# Run smoke tests
smoke-test:
	python scripts/smoke_test.py --url http://localhost:8000

# Monitor performance
monitor:
	python scripts/monitor_performance.py --url http://localhost:8000 --requests 100

# Initialize DVC
dvc-init:
	dvc init
	dvc remote add -d myremote gdrive://your-gdrive-folder-id

# Run DVC pipeline
dvc-repro:
	dvc repro

# Clean generated files
clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info
	rm -rf mlruns
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Full pipeline: prepare -> train -> evaluate -> build
full-pipeline: prepare-data train evaluate docker-build
	@echo "Full pipeline completed!"
