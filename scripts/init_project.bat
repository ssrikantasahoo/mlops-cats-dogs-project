@echo off
REM Project Initialization Script for Windows

echo ==========================================
echo Initializing MLOps Cats vs Dogs Project
echo ==========================================

REM 1. Create virtual environment
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
call venv\Scripts\activate.bat

REM 2. Install dependencies
echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

REM 3. Initialize Git
echo Initializing Git repository...
git init
git add .
git commit -m "Initial commit: MLOps Cats vs Dogs project setup"

REM 4. Initialize DVC
echo Initializing DVC...
dvc init
git add .dvc
git commit -m "Initialize DVC"

REM 5. Create sample data
echo Creating sample dataset...
python scripts\prepare_data.py

REM 6. Train initial model
echo Training initial model...
python src\models\train.py --epochs 5 --data-dir data\processed --output-dir models

echo ==========================================
echo Project initialized successfully!
echo ==========================================
echo.
echo Next steps:
echo   1. Download real dataset from Kaggle:
echo      kaggle datasets download -d tongpython/cat-and-dog -p data\raw --unzip
echo.
echo   2. Prepare the dataset:
echo      python scripts\prepare_data.py
echo.
echo   3. Train the model:
echo      python src\models\train.py
echo.
echo   4. Run the API:
echo      uvicorn src.api.main:app --reload
echo.
echo   5. Build Docker image:
echo      docker build -t cats-dogs-classifier .
echo.
echo   6. View MLflow experiments:
echo      mlflow ui --port 5000

pause
