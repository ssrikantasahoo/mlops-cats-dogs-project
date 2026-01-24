"""
Data Drift Detection for Model Monitoring
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataDriftDetector:
    """
    Detects data drift in model predictions.
    Monitors for:
    - Prediction distribution shifts
    - Confidence score changes
    - Input feature drift (if available)
    """

    def __init__(
        self,
        reference_window_days: int = 7,
        detection_window_hours: int = 24,
        confidence_threshold: float = 0.1,
        distribution_threshold: float = 0.15
    ):
        """
        Initialize drift detector.

        Args:
            reference_window_days: Days of data to use as reference
            detection_window_hours: Hours of recent data to compare
            confidence_threshold: Threshold for confidence drift alert
            distribution_threshold: Threshold for distribution drift alert
        """
        self.reference_window_days = reference_window_days
        self.detection_window_hours = detection_window_hours
        self.confidence_threshold = confidence_threshold
        self.distribution_threshold = distribution_threshold

        self.reference_stats: Optional[Dict] = None
        self.alerts: List[Dict] = []

    def set_reference_baseline(self, predictions: List[Dict]):
        """
        Set reference statistics from historical predictions.

        Args:
            predictions: List of prediction records
        """
        if not predictions:
            logger.warning("No predictions to set as reference")
            return

        confidences = [p['confidence'] for p in predictions]
        class_counts = defaultdict(int)
        for p in predictions:
            class_counts[p['predicted_class']] += 1

        total = len(predictions)
        class_distribution = {k: v / total for k, v in class_counts.items()}

        self.reference_stats = {
            "created_at": datetime.now().isoformat(),
            "sample_count": total,
            "mean_confidence": np.mean(confidences),
            "std_confidence": np.std(confidences),
            "class_distribution": class_distribution,
            "confidence_percentiles": {
                "p25": np.percentile(confidences, 25),
                "p50": np.percentile(confidences, 50),
                "p75": np.percentile(confidences, 75)
            }
        }

        logger.info(f"Reference baseline set with {total} samples")

    def detect_drift(self, recent_predictions: List[Dict]) -> Dict:
        """
        Detect drift in recent predictions compared to reference.

        Args:
            recent_predictions: Recent prediction records

        Returns:
            Drift detection results
        """
        if self.reference_stats is None:
            return {"error": "Reference baseline not set"}

        if not recent_predictions:
            return {"error": "No recent predictions to analyze"}

        # Calculate current statistics
        confidences = [p['confidence'] for p in recent_predictions]
        class_counts = defaultdict(int)
        for p in recent_predictions:
            class_counts[p['predicted_class']] += 1

        total = len(recent_predictions)
        current_distribution = {k: v / total for k, v in class_counts.items()}

        # Confidence drift
        current_mean_conf = np.mean(confidences)
        ref_mean_conf = self.reference_stats['mean_confidence']
        confidence_drift = abs(current_mean_conf - ref_mean_conf)
        confidence_drifted = confidence_drift > self.confidence_threshold

        # Distribution drift (using Total Variation Distance)
        all_classes = set(current_distribution.keys()) | set(
            self.reference_stats['class_distribution'].keys()
        )
        distribution_drift = sum(
            abs(
                current_distribution.get(c, 0) -
                self.reference_stats['class_distribution'].get(c, 0)
            )
            for c in all_classes
        ) / 2
        distribution_drifted = distribution_drift > self.distribution_threshold

        # Create result
        result = {
            "timestamp": datetime.now().isoformat(),
            "sample_count": total,
            "drift_detected": confidence_drifted or distribution_drifted,
            "confidence_drift": {
                "current_mean": round(current_mean_conf, 4),
                "reference_mean": round(ref_mean_conf, 4),
                "drift_magnitude": round(confidence_drift, 4),
                "threshold": self.confidence_threshold,
                "drifted": confidence_drifted
            },
            "distribution_drift": {
                "current_distribution": {k: round(v, 4) for k, v in current_distribution.items()},
                "reference_distribution": {
                    k: round(v, 4)
                    for k, v in self.reference_stats['class_distribution'].items()
                },
                "total_variation_distance": round(distribution_drift, 4),
                "threshold": self.distribution_threshold,
                "drifted": distribution_drifted
            }
        }

        # Log alert if drift detected
        if result['drift_detected']:
            alert = {
                "timestamp": result['timestamp'],
                "type": "drift_detected",
                "confidence_drifted": confidence_drifted,
                "distribution_drifted": distribution_drifted,
                "details": result
            }
            self.alerts.append(alert)
            logger.warning(f"Data drift detected: {alert}")

        return result

    def get_alerts(self, hours: int = 24) -> List[Dict]:
        """Get recent alerts."""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [
            a for a in self.alerts
            if datetime.fromisoformat(a['timestamp']) > cutoff
        ]

    def calculate_psi(
        self,
        reference_distribution: Dict[str, float],
        current_distribution: Dict[str, float]
    ) -> float:
        """
        Calculate Population Stability Index (PSI).

        PSI < 0.1: No significant change
        0.1 <= PSI < 0.2: Moderate change
        PSI >= 0.2: Significant change

        Args:
            reference_distribution: Reference class distribution
            current_distribution: Current class distribution

        Returns:
            PSI value
        """
        epsilon = 1e-10
        all_classes = set(reference_distribution.keys()) | set(current_distribution.keys())

        psi = 0.0
        for cls in all_classes:
            ref = reference_distribution.get(cls, epsilon)
            cur = current_distribution.get(cls, epsilon)

            # Avoid log(0)
            ref = max(ref, epsilon)
            cur = max(cur, epsilon)

            psi += (cur - ref) * np.log(cur / ref)

        return psi

    def generate_monitoring_report(
        self,
        predictions: List[Dict],
        output_path: Optional[str] = None
    ) -> Dict:
        """
        Generate comprehensive monitoring report.

        Args:
            predictions: All prediction records
            output_path: Optional path to save report

        Returns:
            Monitoring report
        """
        now = datetime.now()
        reference_cutoff = now - timedelta(days=self.reference_window_days)
        detection_cutoff = now - timedelta(hours=self.detection_window_hours)

        # Split predictions
        reference_preds = [
            p for p in predictions
            if datetime.fromisoformat(p['timestamp']) < detection_cutoff
            and datetime.fromisoformat(p['timestamp']) > reference_cutoff
        ]
        recent_preds = [
            p for p in predictions
            if datetime.fromisoformat(p['timestamp']) >= detection_cutoff
        ]

        # Set reference if not set
        if self.reference_stats is None and reference_preds:
            self.set_reference_baseline(reference_preds)

        # Detect drift
        drift_result = self.detect_drift(recent_preds) if recent_preds else None

        # Calculate accuracy if labels available
        labeled = [p for p in recent_preds if p.get('true_label') is not None]
        accuracy = None
        if labeled:
            correct = sum(1 for p in labeled if p['predicted_class'] == p['true_label'])
            accuracy = correct / len(labeled)

        report = {
            "generated_at": now.isoformat(),
            "reference_period": {
                "start": (now - timedelta(days=self.reference_window_days)).isoformat(),
                "end": detection_cutoff.isoformat(),
                "sample_count": len(reference_preds)
            },
            "detection_period": {
                "start": detection_cutoff.isoformat(),
                "end": now.isoformat(),
                "sample_count": len(recent_preds)
            },
            "drift_detection": drift_result,
            "model_performance": {
                "labeled_samples": len(labeled),
                "accuracy": round(accuracy, 4) if accuracy else None
            },
            "alerts": self.get_alerts()
        }

        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Report saved to {output_path}")

        return report


if __name__ == "__main__":
    # Test drift detection
    import random

    detector = DataDriftDetector()

    # Generate reference predictions (balanced)
    reference = [
        {
            "timestamp": (datetime.now() - timedelta(days=random.randint(1, 7))).isoformat(),
            "predicted_class": random.choice(['Cat', 'Dog']),
            "confidence": random.uniform(0.7, 0.95),
            "true_label": random.choice(['Cat', 'Dog'])
        }
        for _ in range(1000)
    ]

    detector.set_reference_baseline(reference)
    print("Reference stats:", json.dumps(detector.reference_stats, indent=2))

    # Generate drifted predictions (biased toward Dog with lower confidence)
    recent = [
        {
            "timestamp": datetime.now().isoformat(),
            "predicted_class": random.choices(['Cat', 'Dog'], weights=[0.3, 0.7])[0],
            "confidence": random.uniform(0.5, 0.8),
            "true_label": random.choice(['Cat', 'Dog'])
        }
        for _ in range(100)
    ]

    # Detect drift
    result = detector.detect_drift(recent)
    print("\nDrift Detection Result:")
    print(json.dumps(result, indent=2))

    # Generate report
    all_predictions = reference + recent
    report = detector.generate_monitoring_report(all_predictions)
    print("\nMonitoring Report:")
    print(json.dumps(report, indent=2))
