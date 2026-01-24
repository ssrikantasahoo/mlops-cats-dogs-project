"""
Post-Deployment Model Performance Monitoring Script
Collects predictions and compares with ground truth to track model performance.
"""

import os
import sys
import json
import io
import random
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple
import requests
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.monitoring.metrics import MetricsCollector
from src.monitoring.drift_detector import DataDriftDetector


def generate_simulated_requests(
    num_requests: int = 100,
    api_url: str = "http://localhost:8000"
) -> List[Dict]:
    """
    Generate simulated prediction requests with known labels.

    Args:
        num_requests: Number of requests to generate
        api_url: API endpoint URL

    Returns:
        List of prediction results with true labels
    """
    results = []

    for i in range(num_requests):
        # Generate random image with known class
        true_class = random.choice(['Cat', 'Dog'])

        # Create image with class-specific characteristics
        if true_class == 'Cat':
            # Cats: More reddish tones
            color = (
                random.randint(150, 255),
                random.randint(50, 150),
                random.randint(50, 150)
            )
        else:
            # Dogs: More brownish tones
            color = (
                random.randint(100, 200),
                random.randint(80, 180),
                random.randint(50, 150)
            )

        img = Image.new('RGB', (224, 224), color=color)
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        buffer.seek(0)

        try:
            # Send prediction request
            start_time = time.time()
            response = requests.post(
                f"{api_url}/predict",
                files={'file': ('test.jpg', buffer.getvalue(), 'image/jpeg')},
                timeout=30
            )
            latency = (time.time() - start_time) * 1000

            if response.status_code == 200:
                data = response.json()
                results.append({
                    'timestamp': datetime.now().isoformat(),
                    'predicted_class': data['prediction'],
                    'confidence': data['confidence'],
                    'true_label': true_class,
                    'correct': data['prediction'] == true_class,
                    'latency_ms': latency
                })
            else:
                logger.warning(f"Request {i} failed: {response.status_code}")

        except Exception as e:
            logger.error(f"Request {i} error: {e}")

        # Small delay between requests
        time.sleep(0.1)

    return results


def calculate_performance_metrics(results: List[Dict]) -> Dict:
    """Calculate performance metrics from results."""
    if not results:
        return {'error': 'No results to analyze'}

    correct = sum(1 for r in results if r['correct'])
    total = len(results)

    # Per-class metrics
    cat_results = [r for r in results if r['true_label'] == 'Cat']
    dog_results = [r for r in results if r['true_label'] == 'Dog']

    cat_correct = sum(1 for r in cat_results if r['correct'])
    dog_correct = sum(1 for r in dog_results if r['correct'])

    # Calculate precision/recall for Dogs (positive class)
    true_positives = sum(1 for r in results
                        if r['predicted_class'] == 'Dog' and r['true_label'] == 'Dog')
    false_positives = sum(1 for r in results
                         if r['predicted_class'] == 'Dog' and r['true_label'] == 'Cat')
    false_negatives = sum(1 for r in results
                         if r['predicted_class'] == 'Cat' and r['true_label'] == 'Dog')

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    latencies = [r['latency_ms'] for r in results]

    return {
        'total_samples': total,
        'accuracy': round(correct / total, 4),
        'cat_accuracy': round(cat_correct / len(cat_results), 4) if cat_results else 0,
        'dog_accuracy': round(dog_correct / len(dog_results), 4) if dog_results else 0,
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4),
        'avg_confidence': round(sum(r['confidence'] for r in results) / total, 4),
        'avg_latency_ms': round(sum(latencies) / len(latencies), 2),
        'p50_latency_ms': round(sorted(latencies)[len(latencies) // 2], 2),
        'p95_latency_ms': round(sorted(latencies)[int(len(latencies) * 0.95)], 2),
        'timestamp': datetime.now().isoformat()
    }


def run_monitoring(
    api_url: str,
    num_requests: int,
    output_dir: str
):
    """
    Run full monitoring workflow.

    Args:
        api_url: API endpoint URL
        num_requests: Number of requests to simulate
        output_dir: Directory to save results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting performance monitoring with {num_requests} requests...")

    # Generate simulated requests
    results = generate_simulated_requests(num_requests, api_url)

    if not results:
        logger.error("No results collected!")
        return

    logger.info(f"Collected {len(results)} prediction results")

    # Calculate metrics
    metrics = calculate_performance_metrics(results)

    # Save raw results
    with open(output_path / 'prediction_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save metrics
    with open(output_path / 'performance_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Initialize drift detector and check for drift
    detector = DataDriftDetector()
    detector.set_reference_baseline(results[:len(results)//2])  # Use first half as reference
    drift_result = detector.detect_drift(results[len(results)//2:])  # Check second half

    with open(output_path / 'drift_report.json', 'w') as f:
        json.dump(drift_result, f, indent=2)

    # Log to metrics collector
    collector = MetricsCollector(str(output_path / 'metrics'))
    for r in results:
        collector.log_prediction(
            predicted_class=r['predicted_class'],
            confidence=r['confidence'],
            true_label=r['true_label'],
            processing_time_ms=r['latency_ms']
        )

    # Print summary
    logger.info("\n" + "="*50)
    logger.info("PERFORMANCE MONITORING SUMMARY")
    logger.info("="*50)
    logger.info(f"Total Samples: {metrics['total_samples']}")
    logger.info(f"Accuracy: {metrics['accuracy']:.2%}")
    logger.info(f"  - Cat Accuracy: {metrics['cat_accuracy']:.2%}")
    logger.info(f"  - Dog Accuracy: {metrics['dog_accuracy']:.2%}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
    logger.info(f"Avg Confidence: {metrics['avg_confidence']:.4f}")
    logger.info(f"Avg Latency: {metrics['avg_latency_ms']:.2f}ms")
    logger.info(f"P95 Latency: {metrics['p95_latency_ms']:.2f}ms")
    logger.info("="*50)

    if drift_result.get('drift_detected'):
        logger.warning("DATA DRIFT DETECTED!")
        logger.warning(f"Confidence Drift: {drift_result['confidence_drift']['drifted']}")
        logger.warning(f"Distribution Drift: {drift_result['distribution_drift']['drifted']}")
    else:
        logger.info("No significant data drift detected")

    logger.info(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Monitor deployed model performance')
    parser.add_argument('--url', type=str, default='http://localhost:8000',
                       help='API endpoint URL')
    parser.add_argument('--requests', type=int, default=100,
                       help='Number of requests to simulate')
    parser.add_argument('--output-dir', type=str, default='logs/monitoring',
                       help='Directory to save monitoring results')

    args = parser.parse_args()

    run_monitoring(
        api_url=args.url,
        num_requests=args.requests,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
