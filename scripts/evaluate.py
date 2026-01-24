"""
Model Evaluation Script
Evaluates the trained model on test set and generates metrics.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.cnn_model import create_model
from src.data.dataset import CatsDogsDataset
from src.data.preprocessing import get_val_transforms


def load_model(model_path: str, device: str = 'cpu'):
    """Load trained model."""
    model = create_model('cnn', num_classes=2)

    if Path(model_path).exists():
        state_dict = torch.load(model_path, map_location=device)
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'])
        else:
            model.load_state_dict(state_dict)
        logger.info(f"Loaded model from {model_path}")
    else:
        logger.warning(f"Model not found at {model_path}, using random weights")

    model.to(device)
    model.eval()
    return model


def evaluate(model, data_loader, device):
    """
    Evaluate model and return predictions and labels.
    """
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of class 1 (Dog)

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def compute_metrics(labels, predictions):
    """Compute all evaluation metrics."""
    return {
        'accuracy': float(accuracy_score(labels, predictions)),
        'precision': float(precision_score(labels, predictions, average='binary')),
        'recall': float(recall_score(labels, predictions, average='binary')),
        'f1_score': float(f1_score(labels, predictions, average='binary')),
        'confusion_matrix': confusion_matrix(labels, predictions).tolist()
    }


def plot_roc_curve(labels, probabilities, output_path):
    """Plot and save ROC curve."""
    fpr, tpr, thresholds = roc_curve(labels, probabilities)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return roc_auc


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--model-path', type=str, default='models/model.pt',
                       help='Path to trained model')
    parser.add_argument('--data-dir', type=str, default='data/processed',
                       help='Path to processed data')
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Directory for evaluation outputs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for evaluation')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_model(args.model_path, device)

    # Create test dataset
    test_dataset = CatsDogsDataset(
        args.data_dir,
        transform=get_val_transforms(),
        split='test'
    )

    if len(test_dataset) == 0:
        logger.error("No test data found!")
        # Create dummy metrics
        metrics = {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'auc_roc': 0.0,
            'test_samples': 0
        }
        with open(output_path / 'evaluation_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        return

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    logger.info(f"Evaluating on {len(test_dataset)} test samples...")

    # Evaluate
    labels, predictions, probabilities = evaluate(model, test_loader, device)

    # Compute metrics
    metrics = compute_metrics(labels, predictions)
    metrics['test_samples'] = len(test_dataset)

    # Plot ROC curve
    roc_auc = plot_roc_curve(labels, probabilities, output_path / 'roc_curve.png')
    metrics['auc_roc'] = float(roc_auc)

    # Save metrics
    with open(output_path / 'evaluation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Print classification report
    report = classification_report(labels, predictions, target_names=['Cat', 'Dog'])
    logger.info(f"\nClassification Report:\n{report}")

    # Save classification report
    with open(output_path / 'classification_report.txt', 'w') as f:
        f.write(report)

    logger.info(f"\nEvaluation Metrics:")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall: {metrics['recall']:.4f}")
    logger.info(f"  F1 Score: {metrics['f1_score']:.4f}")
    logger.info(f"  AUC-ROC: {metrics['auc_roc']:.4f}")

    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
