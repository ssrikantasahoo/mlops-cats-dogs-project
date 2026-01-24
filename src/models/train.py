"""
Model Training Script with MLflow Experiment Tracking
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Tuple, Dict, Optional
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.dataset import CatsDogsDataset, create_sample_dataset
from src.data.preprocessing import get_train_transforms, get_val_transforms, create_data_loaders
from src.models.cnn_model import create_model, count_parameters

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DEFAULT_CONFIG = {
    'model_type': 'cnn',
    'batch_size': 32,
    'epochs': 10,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'dropout_rate': 0.5,
    'seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int
) -> Tuple[float, float]:
    """
    Train for one epoch.

    Args:
        model: PyTorch model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")

    for batch_idx, (images, labels) in enumerate(progress_bar):
        images, labels = images.to(device), labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        progress_bar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })

    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: str,
    phase: str = "Val"
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Evaluate model on a dataset.

    Args:
        model: PyTorch model
        data_loader: Data loader
        criterion: Loss function
        device: Device
        phase: Phase name for logging

    Returns:
        Tuple of (loss, accuracy, all_labels, all_predictions)
    """
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc=f"[{phase}]"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    avg_loss = running_loss / len(data_loader)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    accuracy = accuracy_score(all_labels, all_preds) * 100

    return avg_loss, accuracy, all_labels, all_preds


def compute_metrics(labels: np.ndarray, predictions: np.ndarray) -> Dict:
    """Compute classification metrics."""
    return {
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions, average='binary'),
        'recall': recall_score(labels, predictions, average='binary'),
        'f1': f1_score(labels, predictions, average='binary')
    }


def plot_confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: list,
    save_path: str
):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(labels, predictions)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_training_curves(
    train_losses: list,
    val_losses: list,
    train_accs: list,
    val_accs: list,
    save_path: str
):
    """Plot and save training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(train_losses) + 1)

    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy plot
    ax2.plot(epochs, train_accs, 'b-', label='Train Acc')
    ax2.plot(epochs, val_accs, 'r-', label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def train_model(
    data_dir: str,
    output_dir: str = 'models',
    config: Optional[Dict] = None,
    experiment_name: str = 'cats_dogs_classification'
) -> str:
    """
    Main training function with MLflow tracking.

    Args:
        data_dir: Path to dataset
        output_dir: Path to save models
        config: Training configuration
        experiment_name: MLflow experiment name

    Returns:
        Path to saved model
    """
    # Merge with default config
    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config)

    # Set seed for reproducibility
    set_seed(cfg['seed'])

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Set up MLflow
    mlflow.set_experiment(experiment_name)

    # Check if data exists, if not create sample data
    data_path = Path(data_dir)
    if not data_path.exists() or not any(data_path.iterdir()):
        logger.info("No data found. Creating sample dataset for testing...")
        create_sample_dataset(str(data_path), num_samples=100)

    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(cfg)

        # Create datasets
        train_dataset = CatsDogsDataset(
            data_dir,
            transform=get_train_transforms(),
            split='train'
        )
        val_dataset = CatsDogsDataset(
            data_dir,
            transform=get_val_transforms(),
            split='val'
        )
        test_dataset = CatsDogsDataset(
            data_dir,
            transform=get_val_transforms(),
            split='test'
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg['batch_size'],
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg['batch_size'],
            shuffle=False,
            num_workers=0
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg['batch_size'],
            shuffle=False,
            num_workers=0
        )

        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Val samples: {len(val_dataset)}")
        logger.info(f"Test samples: {len(test_dataset)}")

        # Log dataset info
        mlflow.log_param("train_samples", len(train_dataset))
        mlflow.log_param("val_samples", len(val_dataset))
        mlflow.log_param("test_samples", len(test_dataset))

        # Create model
        device = cfg['device']
        model = create_model(
            model_type=cfg['model_type'],
            dropout_rate=cfg['dropout_rate']
        ).to(device)

        mlflow.log_param("model_parameters", count_parameters(model))
        logger.info(f"Model parameters: {count_parameters(model):,}")

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg['learning_rate'],
            weight_decay=cfg['weight_decay']
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=3, factor=0.5
        )

        # Training loop
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        best_val_acc = 0.0

        for epoch in range(cfg['epochs']):
            # Train
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device, epoch
            )

            # Validate
            val_loss, val_acc, _, _ = evaluate_model(
                model, val_loader, criterion, device, "Val"
            )

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Log metrics
            mlflow.log_metrics({
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc
            }, step=epoch)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)

            logger.info(
                f"Epoch {epoch+1}/{cfg['epochs']} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_path = output_path / 'best_model.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_accuracy': val_acc,
                    'config': cfg
                }, best_model_path)
                logger.info(f"Saved best model with val_acc: {val_acc:.2f}%")

        # Final evaluation on test set
        model.load_state_dict(torch.load(best_model_path)['model_state_dict'])
        test_loss, test_acc, test_labels, test_preds = evaluate_model(
            model, test_loader, criterion, device, "Test"
        )

        # Compute and log test metrics
        test_metrics = compute_metrics(test_labels, test_preds)
        mlflow.log_metrics({
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'test_precision': test_metrics['precision'],
            'test_recall': test_metrics['recall'],
            'test_f1': test_metrics['f1']
        })

        logger.info(f"\nTest Results:")
        logger.info(f"  Accuracy: {test_acc:.2f}%")
        logger.info(f"  Precision: {test_metrics['precision']:.4f}")
        logger.info(f"  Recall: {test_metrics['recall']:.4f}")
        logger.info(f"  F1 Score: {test_metrics['f1']:.4f}")

        # Save confusion matrix
        cm_path = output_path / 'confusion_matrix.png'
        plot_confusion_matrix(
            test_labels, test_preds,
            ['Cat', 'Dog'],
            str(cm_path)
        )
        mlflow.log_artifact(str(cm_path))

        # Save training curves
        curves_path = output_path / 'training_curves.png'
        plot_training_curves(
            train_losses, val_losses,
            train_accs, val_accs,
            str(curves_path)
        )
        mlflow.log_artifact(str(curves_path))

        # Save final model in multiple formats
        # PyTorch format
        final_model_path = output_path / 'model.pt'
        torch.save(model.state_dict(), final_model_path)
        mlflow.log_artifact(str(final_model_path))

        # MLflow PyTorch model
        mlflow.pytorch.log_model(model, "model")

        # Save model config
        config_path = output_path / 'model_config.json'
        with open(config_path, 'w') as f:
            json.dump(cfg, f, indent=2)
        mlflow.log_artifact(str(config_path))

        # Log classification report
        report = classification_report(
            test_labels, test_preds,
            target_names=['Cat', 'Dog']
        )
        report_path = output_path / 'classification_report.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        mlflow.log_artifact(str(report_path))

        logger.info(f"\nModel saved to: {output_path}")
        logger.info(f"MLflow run ID: {mlflow.active_run().info.run_id}")

        return str(final_model_path)


def main():
    parser = argparse.ArgumentParser(description='Train Cats vs Dogs Classifier')
    parser.add_argument('--data-dir', type=str, default='data/processed',
                        help='Path to dataset')
    parser.add_argument('--output-dir', type=str, default='models',
                        help='Path to save models')
    parser.add_argument('--model-type', type=str, default='cnn',
                        choices=['cnn', 'logistic', 'resnet18'],
                        help='Model architecture')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--experiment-name', type=str,
                        default='cats_dogs_classification',
                        help='MLflow experiment name')

    args = parser.parse_args()

    config = {
        'model_type': args.model_type,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.lr
    }

    train_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        config=config,
        experiment_name=args.experiment_name
    )


if __name__ == "__main__":
    main()
