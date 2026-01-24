"""
Smoke Test Script for Post-Deployment Verification
Tests the deployed API endpoints to ensure the service is working correctly.
"""

import os
import sys
import io
import time
import argparse
import requests
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_image():
    """Create a test image for prediction."""
    img = Image.new('RGB', (224, 224), color='blue')
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    buffer.seek(0)
    return buffer


def test_health_endpoint(base_url: str) -> bool:
    """Test the health check endpoint."""
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        response.raise_for_status()

        data = response.json()
        assert data['status'] == 'healthy', f"Unhealthy status: {data['status']}"
        assert data['model_loaded'] is True, "Model not loaded"

        logger.info(f"Health check passed: {data}")
        return True

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return False


def test_prediction_endpoint(base_url: str) -> bool:
    """Test the prediction endpoint."""
    try:
        # Create test image
        image_buffer = create_test_image()

        # Send prediction request
        response = requests.post(
            f"{base_url}/predict",
            files={'file': ('test.jpg', image_buffer.getvalue(), 'image/jpeg')},
            timeout=30
        )
        response.raise_for_status()

        data = response.json()

        # Validate response
        assert 'prediction' in data, "Missing prediction in response"
        assert 'confidence' in data, "Missing confidence in response"
        assert data['prediction'] in ['Cat', 'Dog'], f"Invalid prediction: {data['prediction']}"
        assert 0 <= data['confidence'] <= 1, f"Invalid confidence: {data['confidence']}"

        logger.info(f"Prediction test passed: {data}")
        return True

    except Exception as e:
        logger.error(f"Prediction test failed: {e}")
        return False


def test_metrics_endpoint(base_url: str) -> bool:
    """Test the metrics endpoint."""
    try:
        response = requests.get(f"{base_url}/metrics", timeout=10)
        response.raise_for_status()

        # Check content type
        assert 'text/plain' in response.headers.get('content-type', ''), \
            "Invalid content type for metrics"

        logger.info("Metrics endpoint test passed")
        return True

    except Exception as e:
        logger.error(f"Metrics test failed: {e}")
        return False


def test_stats_endpoint(base_url: str) -> bool:
    """Test the stats endpoint."""
    try:
        response = requests.get(f"{base_url}/stats", timeout=10)
        response.raise_for_status()

        data = response.json()
        assert 'total_requests' in data, "Missing total_requests"
        assert 'successful_predictions' in data, "Missing successful_predictions"

        logger.info(f"Stats test passed: {data}")
        return True

    except Exception as e:
        logger.error(f"Stats test failed: {e}")
        return False


def run_smoke_tests(base_url: str, retries: int = 3, retry_delay: int = 5) -> bool:
    """
    Run all smoke tests with retries.

    Args:
        base_url: Base URL of the API
        retries: Number of retries for failed tests
        retry_delay: Delay between retries in seconds

    Returns:
        True if all tests pass, False otherwise
    """
    tests = [
        ("Health Check", test_health_endpoint),
        ("Prediction", test_prediction_endpoint),
        ("Metrics", test_metrics_endpoint),
        ("Stats", test_stats_endpoint),
    ]

    results = {}

    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info('='*50)

        passed = False
        for attempt in range(retries):
            if test_func(base_url):
                passed = True
                break
            else:
                if attempt < retries - 1:
                    logger.warning(f"Retrying in {retry_delay} seconds... ({attempt + 1}/{retries})")
                    time.sleep(retry_delay)

        results[test_name] = passed
        status = "PASSED" if passed else "FAILED"
        logger.info(f"Result: {status}")

    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("SMOKE TEST SUMMARY")
    logger.info('='*50)

    all_passed = True
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        logger.info(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    logger.info('='*50)
    final_status = "ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED"
    logger.info(f"Final Result: {final_status}")

    return all_passed


def main():
    parser = argparse.ArgumentParser(description='Run smoke tests on deployed API')
    parser.add_argument('--url', type=str, default='http://localhost:8000',
                       help='Base URL of the API')
    parser.add_argument('--retries', type=int, default=3,
                       help='Number of retries for failed tests')
    parser.add_argument('--retry-delay', type=int, default=5,
                       help='Delay between retries in seconds')
    parser.add_argument('--wait', type=int, default=0,
                       help='Wait time before starting tests (seconds)')

    args = parser.parse_args()

    if args.wait > 0:
        logger.info(f"Waiting {args.wait} seconds before starting tests...")
        time.sleep(args.wait)

    logger.info(f"Starting smoke tests against: {args.url}")

    success = run_smoke_tests(
        base_url=args.url,
        retries=args.retries,
        retry_delay=args.retry_delay
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
