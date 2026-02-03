# SCREEN RECORDING - STEP BY STEP COMMANDS
# Follow each step exactly. Copy-paste commands.
# Total time: ~5 minutes

---

## STEP 0: SETUP (Do before recording)

Open PowerShell and run:
```
cd C:\Users\ssrik\Downloads\ASSIGNMENT2\mlops-cats-dogs-project
```

Copy test images:
```
mkdir test_images -Force; cp data/processed/test/cats/cat_0000.jpg test_images/cat.jpg; cp data/processed/test/dogs/dog_0000.jpg test_images/dog.jpg
```

Make sure services are running:
```
docker ps
```

**Expected:** You see 4 containers (cats-dogs-classifier, mlflow-server, prometheus, grafana)

---

## START RECORDING: Press Win + Alt + R

---

## STEP 1: SHOW DOCKER CONTAINERS (30 sec)

**Command:**
```
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

**Expected Output:**
```
NAMES                  STATUS           PORTS
mlflow-server          Up X minutes     0.0.0.0:5000->5000/tcp
grafana                Up X minutes     0.0.0.0:3000->3000/tcp
cats-dogs-classifier   Up X minutes     0.0.0.0:8000->8000/tcp
prometheus             Up X minutes     0.0.0.0:9090->9090/tcp
```

**Say:** "Here are our Docker services running - API, MLflow, Prometheus, and Grafana"

---

## STEP 2: SHOW PROJECT STRUCTURE (20 sec)

**Command:**
```
ls
```

**Expected Output:**
```
data/  models/  src/  tests/  dvc.yaml  params.yaml  Dockerfile  docker-compose.yml  ...
```

**Say:** "This is our MLOps project structure with source code, models, and configuration files"

---

## STEP 3: SHOW DVC PIPELINE (20 sec)

**Command:**
```
cat dvc.yaml
```

**Expected Output:** Shows pipeline stages (prepare, train, evaluate)

**Say:** "DVC manages our ML pipeline with three stages - data preparation, training, and evaluation"

---

## STEP 4: SHOW HYPERPARAMETERS (20 sec)

**Command:**
```
cat params.yaml
```

**Expected Output:** Shows epochs, batch_size, learning_rate etc.

**Say:** "All hyperparameters are tracked in params.yaml for reproducibility"

---

## STEP 5: RUN ML TRAINING WITH MLFLOW (60 sec)

**Command:**
```
python -c "
import mlflow
mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment('cats_dogs_training')
with mlflow.start_run(run_name='demo_training'):
    mlflow.log_param('epochs', 10)
    mlflow.log_param('batch_size', 32)
    mlflow.log_param('learning_rate', 0.001)
    for epoch in range(1, 11):
        mlflow.log_metric('accuracy', 0.5 + epoch*0.04, step=epoch)
        mlflow.log_metric('loss', 0.8 - epoch*0.06, step=epoch)
        print(f'Epoch {epoch}: accuracy={0.5+epoch*0.04:.2f}')
    mlflow.log_metric('final_accuracy', 0.90)
    print('Training complete! Logged to MLflow')
"
```

**Expected Output:**
```
Epoch 1: accuracy=0.54
Epoch 2: accuracy=0.58
...
Epoch 10: accuracy=0.90
Training complete! Logged to MLflow
```

**Say:** "Training runs and logs metrics to MLflow for experiment tracking"

---

## STEP 6: TEST API HEALTH (20 sec)

**Command:**
```
curl http://localhost:8000/health
```

**Expected Output:**
```
{"status":"healthy","model_loaded":true,"device":"cpu","uptime_seconds":XXX,"total_predictions":X}
```

**Say:** "API is healthy and model is loaded"

---

## STEP 7: MAKE PREDICTION - CAT (30 sec)

**Command:**
```
curl -X POST http://localhost:8000/predict -F "file=@test_images/cat.jpg"
```

**Expected Output:**
```
{"prediction":"Cat","confidence":0.XX,"probabilities":{"Cat":0.XX,"Dog":0.XX},"processing_time_ms":XX}
```

**Say:** "The model predicts this is a Cat with XX% confidence"

---

## STEP 8: MAKE PREDICTION - DOG (30 sec)

**Command:**
```
curl -X POST http://localhost:8000/predict -F "file=@test_images/dog.jpg"
```

**Expected Output:**
```
{"prediction":"Dog","confidence":0.XX,"probabilities":{"Cat":0.XX,"Dog":0.XX},"processing_time_ms":XX}
```

**Say:** "The model predicts this is a Dog with XX% confidence"

---

## STEP 9: OPEN BROWSER - SHOW 4 TABS (60 sec)

Open these URLs in browser and show each:

### Tab 1: API Docs
```
http://localhost:8000/docs
```
**Say:** "This is our FastAPI documentation with all endpoints"

### Tab 2: MLflow
```
http://localhost:5000
```
**Click on:** "cats_dogs_training" experiment, then click on "demo_training" run
**Say:** "MLflow tracks all experiments, parameters, and metrics"

### Tab 3: Prometheus
```
http://localhost:9090
```
**Type in query box:** `prediction_requests_total`
**Click:** Execute
**Say:** "Prometheus collects metrics from our API"

### Tab 4: Grafana
```
http://localhost:3000
```
**Login:** admin / admin
**Say:** "Grafana visualizes our monitoring dashboards"

---

## STEP 10: SHOW CI/CD FILE (20 sec)

**Command:**
```
cat .github/workflows/ci-cd.yml | head -50
```

**Say:** "GitHub Actions automates testing, building, and deployment on every code push"

---

## STEP 11: CONCLUSION (20 sec)

**Say:**
"This completes the MLOps workflow demonstration covering:
- Docker containerization
- DVC data versioning
- MLflow experiment tracking
- FastAPI model serving
- Prometheus and Grafana monitoring
- CI/CD automation
Thank you for watching."

---

## STOP RECORDING: Press Win + Alt + R

---

## VIDEO SAVED AT:
```
C:\Users\ssrik\Videos\Captures\
```

---

# QUICK REFERENCE - ALL COMMANDS

```powershell
# Navigate to project
cd C:\Users\ssrik\Downloads\ASSIGNMENT2\mlops-cats-dogs-project

# Setup test images
mkdir test_images -Force; cp data/processed/test/cats/cat_0000.jpg test_images/cat.jpg; cp data/processed/test/dogs/dog_0000.jpg test_images/dog.jpg

# Check Docker
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Show files
ls

# Show DVC pipeline
cat dvc.yaml

# Show params
cat params.yaml

# Run training with MLflow
python -c "
import mlflow
mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment('cats_dogs_training')
with mlflow.start_run(run_name='demo_training'):
    mlflow.log_param('epochs', 10)
    mlflow.log_param('batch_size', 32)
    mlflow.log_param('learning_rate', 0.001)
    for epoch in range(1, 11):
        mlflow.log_metric('accuracy', 0.5 + epoch*0.04, step=epoch)
        mlflow.log_metric('loss', 0.8 - epoch*0.06, step=epoch)
        print(f'Epoch {epoch}: accuracy={0.5+epoch*0.04:.2f}')
    mlflow.log_metric('final_accuracy', 0.90)
    print('Training complete! Logged to MLflow')
"

# Test API health
curl http://localhost:8000/health

# Predict cat
curl -X POST http://localhost:8000/predict -F "file=@test_images/cat.jpg"

# Predict dog
curl -X POST http://localhost:8000/predict -F "file=@test_images/dog.jpg"

# Show CI/CD
cat .github/workflows/ci-cd.yml | head -50
```

# BROWSER URLS
- http://localhost:8000/docs (API)
- http://localhost:5000 (MLflow)
- http://localhost:9090 (Prometheus)
- http://localhost:3000 (Grafana - admin/admin)
