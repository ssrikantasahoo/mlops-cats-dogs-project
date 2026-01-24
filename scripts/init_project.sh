#!/bin/bash
# Project Initialization Script

set -e

echo "=========================================="
echo "Initializing MLOps Cats vs Dogs Project"
echo "=========================================="

# 1. Create virtual environment
echo "Creating virtual environment..."
python -m venv venv

# Activate virtual environment
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null

# 2. Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 3. Initialize Git
echo "Initializing Git repository..."
git init
git add .
git commit -m "Initial commit: MLOps Cats vs Dogs project setup"

# 4. Initialize DVC
echo "Initializing DVC..."
dvc init
git add .dvc
git commit -m "Initialize DVC"

# 5. Create sample data (if no data exists)
if [ ! -d "data/processed" ] || [ -z "$(ls -A data/processed 2>/dev/null)" ]; then
    echo "Creating sample dataset..."
    python scripts/prepare_data.py
fi

# 6. Train initial model
echo "Training initial model..."
python src/models/train.py --epochs 5 --data-dir data/processed --output-dir models

echo "=========================================="
echo "Project initialized successfully!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Download real dataset from Kaggle:"
echo "     kaggle datasets download -d tongpython/cat-and-dog -p data/raw --unzip"
echo ""
echo "  2. Prepare the dataset:"
echo "     python scripts/prepare_data.py"
echo ""
echo "  3. Train the model:"
echo "     python src/models/train.py"
echo ""
echo "  4. Run the API:"
echo "     uvicorn src.api.main:app --reload"
echo ""
echo "  5. Build Docker image:"
echo "     docker build -t cats-dogs-classifier ."
echo ""
echo "  6. View MLflow experiments:"
echo "     mlflow ui --port 5000"
