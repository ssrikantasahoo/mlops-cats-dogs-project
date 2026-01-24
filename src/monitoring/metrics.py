"""
Metrics Collection for Model Monitoring
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict
import threading
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Collects and stores model performance metrics for monitoring.
    """

    def __init__(self, storage_path: str = "logs/metrics"):
        """
        Initialize metrics collector.

        Args:
            storage_path: Path to store metrics files
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # In-memory metrics
        self.predictions: List[Dict] = []
        self.request_times: List[float] = []
        self.class_counts: Dict[str, int] = defaultdict(int)
        self.error_count: int = 0

        # Locks for thread safety
        self._lock = threading.Lock()

        # Load existing metrics
        self._load_metrics()

    def _load_metrics(self):
        """Load metrics from storage."""
        metrics_file = self.storage_path / "predictions.json"
        if metrics_file.exists():
            try:
                with open(metrics_file, 'r') as f:
                    self.predictions = json.load(f)
                logger.info(f"Loaded {len(self.predictions)} historical predictions")
            except Exception as e:
                logger.warning(f"Error loading metrics: {e}")

    def _save_metrics(self):
        """Save metrics to storage."""
        metrics_file = self.storage_path / "predictions.json"
        try:
            with open(metrics_file, 'w') as f:
                json.dump(self.predictions[-10000:], f)  # Keep last 10000
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")

    def log_prediction(
        self,
        predicted_class: str,
        confidence: float,
        true_label: Optional[str] = None,
        processing_time_ms: float = 0.0,
        metadata: Optional[Dict] = None
    ):
        """
        Log a prediction for monitoring.

        Args:
            predicted_class: Predicted class label
            confidence: Prediction confidence
            true_label: True label (if available for ground truth)
            processing_time_ms: Time taken for prediction
            metadata: Additional metadata
        """
        with self._lock:
            prediction_record = {
                "timestamp": datetime.now().isoformat(),
                "predicted_class": predicted_class,
                "confidence": confidence,
                "true_label": true_label,
                "processing_time_ms": processing_time_ms,
                "correct": predicted_class == true_label if true_label else None,
                "metadata": metadata or {}
            }

            self.predictions.append(prediction_record)
            self.request_times.append(processing_time_ms)
            self.class_counts[predicted_class] += 1

            # Periodically save
            if len(self.predictions) % 100 == 0:
                self._save_metrics()

    def log_error(self, error_type: str, error_message: str):
        """Log an error."""
        with self._lock:
            self.error_count += 1

            error_record = {
                "timestamp": datetime.now().isoformat(),
                "error_type": error_type,
                "error_message": error_message
            }

            error_file = self.storage_path / "errors.jsonl"
            with open(error_file, 'a') as f:
                f.write(json.dumps(error_record) + "\n")

    def get_summary_metrics(self, window_size: int = 1000) -> Dict:
        """
        Get summary metrics for the last N predictions.

        Args:
            window_size: Number of recent predictions to analyze

        Returns:
            Dictionary of summary metrics
        """
        with self._lock:
            recent = self.predictions[-window_size:]

            if not recent:
                return {"error": "No predictions logged yet"}

            # Calculate metrics
            total = len(recent)
            confidences = [p['confidence'] for p in recent]
            latencies = [p['processing_time_ms'] for p in recent]

            # Accuracy (if ground truth available)
            labeled = [p for p in recent if p['correct'] is not None]
            accuracy = (
                sum(1 for p in labeled if p['correct']) / len(labeled)
                if labeled else None
            )

            # Class distribution
            class_dist = defaultdict(int)
            for p in recent:
                class_dist[p['predicted_class']] += 1

            return {
                "total_predictions": total,
                "avg_confidence": round(sum(confidences) / total, 4),
                "min_confidence": round(min(confidences), 4),
                "max_confidence": round(max(confidences), 4),
                "avg_latency_ms": round(sum(latencies) / total, 2),
                "p50_latency_ms": round(sorted(latencies)[len(latencies) // 2], 2),
                "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 2),
                "class_distribution": dict(class_dist),
                "accuracy": round(accuracy, 4) if accuracy else None,
                "labeled_samples": len(labeled),
                "error_count": self.error_count
            }

    def get_time_series_metrics(
        self,
        interval_minutes: int = 60,
        window_hours: int = 24
    ) -> List[Dict]:
        """
        Get time series metrics aggregated by interval.

        Args:
            interval_minutes: Aggregation interval
            window_hours: Time window to analyze

        Returns:
            List of metrics per interval
        """
        from datetime import timedelta

        with self._lock:
            now = datetime.now()
            cutoff = now - timedelta(hours=window_hours)

            # Filter recent predictions
            recent = [
                p for p in self.predictions
                if datetime.fromisoformat(p['timestamp']) > cutoff
            ]

            if not recent:
                return []

            # Group by interval
            intervals = defaultdict(list)
            for p in recent:
                ts = datetime.fromisoformat(p['timestamp'])
                bucket = ts.replace(
                    minute=(ts.minute // interval_minutes) * interval_minutes,
                    second=0,
                    microsecond=0
                )
                intervals[bucket.isoformat()].append(p)

            # Calculate metrics per interval
            time_series = []
            for bucket, preds in sorted(intervals.items()):
                confidences = [p['confidence'] for p in preds]
                latencies = [p['processing_time_ms'] for p in preds]

                time_series.append({
                    "timestamp": bucket,
                    "count": len(preds),
                    "avg_confidence": round(sum(confidences) / len(preds), 4),
                    "avg_latency_ms": round(sum(latencies) / len(preds), 2)
                })

            return time_series

    def export_for_retraining(self, output_path: str) -> int:
        """
        Export labeled predictions for model retraining.

        Args:
            output_path: Path to save labeled data

        Returns:
            Number of exported samples
        """
        with self._lock:
            labeled = [
                {
                    "predicted": p['predicted_class'],
                    "true_label": p['true_label'],
                    "confidence": p['confidence'],
                    "timestamp": p['timestamp'],
                    **p.get('metadata', {})
                }
                for p in self.predictions
                if p['true_label'] is not None
            ]

            with open(output_path, 'w') as f:
                json.dump(labeled, f, indent=2)

            logger.info(f"Exported {len(labeled)} labeled samples to {output_path}")
            return len(labeled)


# Global instance
_metrics_collector = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


if __name__ == "__main__":
    # Test metrics collection
    collector = MetricsCollector("test_metrics")

    # Log some predictions
    for i in range(100):
        import random
        collector.log_prediction(
            predicted_class=random.choice(['Cat', 'Dog']),
            confidence=random.uniform(0.5, 1.0),
            true_label=random.choice(['Cat', 'Dog', None]),
            processing_time_ms=random.uniform(10, 100)
        )

    # Get summary
    print("Summary Metrics:")
    print(json.dumps(collector.get_summary_metrics(), indent=2))

    # Cleanup
    import shutil
    shutil.rmtree("test_metrics")
