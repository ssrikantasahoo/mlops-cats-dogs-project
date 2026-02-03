"""
FastAPI Inference Service for Cats vs Dogs Classification

Endpoints:
- GET /health - Health check
- POST /predict - Image classification prediction
- GET /metrics - Prometheus metrics
"""

import os
import sys
import time
import json
import io
import base64
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import logging
from contextlib import asynccontextmanager

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import structlog

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.cnn_model import create_model
from src.data.preprocessing import preprocess_image_tensor

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    'prediction_requests_total',
    'Total number of prediction requests',
    ['endpoint', 'status']
)
REQUEST_LATENCY = Histogram(
    'prediction_request_latency_seconds',
    'Request latency in seconds',
    ['endpoint']
)
PREDICTION_COUNT = Counter(
    'predictions_total',
    'Total predictions by class',
    ['predicted_class']
)

# Global model
model = None
device = None
class_names = ['Cat', 'Dog']

# Application metrics
app_metrics = {
    'total_requests': 0,
    'successful_predictions': 0,
    'failed_predictions': 0,
    'start_time': None,
    'last_prediction_time': None
}


def ensure_model_ready() -> None:
    """Lazy-load model for contexts where lifespan hooks are not executed."""
    if model is None and not load_model():
        raise HTTPException(status_code=500, detail="Model is not available")


class PredictionRequest(BaseModel):
    """Request model for base64 encoded image."""
    image_base64: str


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    processing_time_ms: float
    timestamp: str


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    model_loaded: bool
    device: str
    uptime_seconds: float
    total_predictions: int


def load_model(model_path: str = None) -> bool:
    """Load the trained model."""
    global model, device

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info("loading_model", device=device)

    try:
        # Create model architecture
        model = create_model('cnn', num_classes=2)

        # Try to load weights
        if model_path is None:
            model_path = os.getenv('MODEL_PATH', 'models/model.pt')

        model_file = Path(model_path)

        if model_file.exists():
            state_dict = torch.load(model_file, map_location=device)
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                model.load_state_dict(state_dict['model_state_dict'])
            else:
                model.load_state_dict(state_dict)
            logger.info("model_loaded", path=str(model_file))
        else:
            logger.warning("model_file_not_found", path=str(model_file))
            logger.info("using_random_weights_for_demo")

        model.to(device)
        model.eval()

        return True

    except Exception as e:
        logger.error("model_load_error", error=str(e))
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    app_metrics['start_time'] = datetime.now()
    success = load_model()
    if not success:
        logger.error("failed_to_load_model")
    yield
    # Shutdown
    logger.info("shutting_down")


# Create FastAPI app
app = FastAPI(
    title="Cats vs Dogs Classifier API",
    description="Binary image classification for pet adoption platform",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests."""
    start_time = time.time()

    response = await call_next(request)

    process_time = time.time() - start_time

    logger.info(
        "request",
        method=request.method,
        url=str(request.url),
        status_code=response.status_code,
        process_time_ms=round(process_time * 1000, 2)
    )

    return response


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Cats vs Dogs Classifier API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    Returns service status and basic metrics.
    """
    REQUEST_COUNT.labels(endpoint='health', status='success').inc()

    uptime = 0.0
    if app_metrics['start_time']:
        uptime = (datetime.now() - app_metrics['start_time']).total_seconds()

    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        device=device or "not_initialized",
        uptime_seconds=round(uptime, 2),
        total_predictions=app_metrics['successful_predictions']
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict whether an image contains a cat or dog.

    Args:
        file: Image file (JPEG, PNG)

    Returns:
        Prediction result with confidence scores
    """
    start_time = time.time()
    app_metrics['total_requests'] += 1

    try:
        ensure_model_ready()

        # Validate file type
        if not file.content_type.startswith('image/'):
            REQUEST_COUNT.labels(endpoint='predict', status='error').inc()
            raise HTTPException(
                status_code=400,
                detail="File must be an image (JPEG, PNG)"
            )

        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Preprocess
        image_tensor = preprocess_image_tensor(image)
        image_tensor = image_tensor.unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        predicted_class = class_names[predicted.item()]
        confidence_value = confidence.item()

        # Update metrics
        processing_time = (time.time() - start_time) * 1000
        app_metrics['successful_predictions'] += 1
        app_metrics['last_prediction_time'] = datetime.now()

        REQUEST_COUNT.labels(endpoint='predict', status='success').inc()
        REQUEST_LATENCY.labels(endpoint='predict').observe(processing_time / 1000)
        PREDICTION_COUNT.labels(predicted_class=predicted_class).inc()

        logger.info(
            "prediction",
            predicted_class=predicted_class,
            confidence=round(confidence_value, 4),
            processing_time_ms=round(processing_time, 2)
        )

        return PredictionResponse(
            prediction=predicted_class,
            confidence=round(confidence_value, 4),
            probabilities={
                class_names[i]: round(probabilities[0][i].item(), 4)
                for i in range(len(class_names))
            },
            processing_time_ms=round(processing_time, 2),
            timestamp=datetime.now().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        app_metrics['failed_predictions'] += 1
        REQUEST_COUNT.labels(endpoint='predict', status='error').inc()
        logger.error("prediction_error", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/base64", response_model=PredictionResponse)
async def predict_base64(request: PredictionRequest):
    """
    Predict from base64 encoded image.

    Args:
        request: Request with base64 encoded image

    Returns:
        Prediction result with confidence scores
    """
    start_time = time.time()
    app_metrics['total_requests'] += 1

    try:
        ensure_model_ready()

        # Decode base64 image
        image_data = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_data))

        # Preprocess
        image_tensor = preprocess_image_tensor(image)
        image_tensor = image_tensor.unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        predicted_class = class_names[predicted.item()]
        confidence_value = confidence.item()

        processing_time = (time.time() - start_time) * 1000
        app_metrics['successful_predictions'] += 1

        REQUEST_COUNT.labels(endpoint='predict_base64', status='success').inc()
        REQUEST_LATENCY.labels(endpoint='predict_base64').observe(processing_time / 1000)
        PREDICTION_COUNT.labels(predicted_class=predicted_class).inc()

        return PredictionResponse(
            prediction=predicted_class,
            confidence=round(confidence_value, 4),
            probabilities={
                class_names[i]: round(probabilities[0][i].item(), 4)
                for i in range(len(class_names))
            },
            processing_time_ms=round(processing_time, 2),
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        app_metrics['failed_predictions'] += 1
        REQUEST_COUNT.labels(endpoint='predict_base64', status='error').inc()
        logger.error("prediction_error", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/stats")
async def stats():
    """Get application statistics."""
    uptime = 0.0
    if app_metrics['start_time']:
        uptime = (datetime.now() - app_metrics['start_time']).total_seconds()

    return {
        "total_requests": app_metrics['total_requests'],
        "successful_predictions": app_metrics['successful_predictions'],
        "failed_predictions": app_metrics['failed_predictions'],
        "success_rate": round(
            app_metrics['successful_predictions'] / max(1, app_metrics['total_requests']) * 100, 2
        ),
        "uptime_seconds": round(uptime, 2),
        "last_prediction": (
            app_metrics['last_prediction_time'].isoformat()
            if app_metrics['last_prediction_time'] else None
        )
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
