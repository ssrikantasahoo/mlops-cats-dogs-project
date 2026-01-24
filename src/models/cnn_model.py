"""
CNN Model for Cats vs Dogs Binary Classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CatsDogsCNN(nn.Module):
    """
    Simple CNN architecture for binary image classification.
    Input: 224x224x3 RGB images
    Output: Binary classification (Cat=0, Dog=1)
    """

    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.5):
        """
        Initialize the CNN model.

        Args:
            num_classes: Number of output classes (2 for binary)
            dropout_rate: Dropout rate for regularization
        """
        super(CatsDogsCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers
        self.fc1 = nn.Linear(256, 512)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch, 3, 224, 224)

        Returns:
            Output logits of shape (batch, num_classes)
        """
        # Conv block 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))

        # Conv block 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))

        # Conv block 3
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        # Conv block 4
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))

        # Global pooling and flatten
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get class probabilities.

        Args:
            x: Input tensor

        Returns:
            Softmax probabilities
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=1)


class LogisticRegressionBaseline(nn.Module):
    """
    Simple logistic regression baseline on flattened pixels.
    Input: 224x224x3 = 150528 features
    """

    def __init__(self, input_size: int = 224 * 224 * 3, num_classes: int = 2):
        """
        Initialize logistic regression model.

        Args:
            input_size: Number of input features (flattened image)
            num_classes: Number of output classes
        """
        super(LogisticRegressionBaseline, self).__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.flatten(x)
        return self.fc(x)


def create_model(
    model_type: str = 'cnn',
    num_classes: int = 2,
    pretrained: bool = False,
    dropout_rate: float = 0.5
) -> nn.Module:
    """
    Factory function to create models.

    Args:
        model_type: Type of model ('cnn', 'logistic', 'resnet18')
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights (for resnet)
        dropout_rate: Dropout rate for regularization

    Returns:
        PyTorch model
    """
    if model_type == 'cnn':
        model = CatsDogsCNN(num_classes=num_classes, dropout_rate=dropout_rate)
        logger.info("Created CatsDogsCNN model")

    elif model_type == 'logistic':
        model = LogisticRegressionBaseline(num_classes=num_classes)
        logger.info("Created LogisticRegression baseline model")

    elif model_type == 'resnet18':
        from torchvision import models
        model = models.resnet18(pretrained=pretrained)
        # Modify final layer for binary classification
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        logger.info(f"Created ResNet18 model (pretrained={pretrained})")

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model: nn.Module, input_size: Tuple[int, ...] = (1, 3, 224, 224)) -> str:
    """
    Get a summary of the model architecture.

    Args:
        model: PyTorch model
        input_size: Input tensor size

    Returns:
        Summary string
    """
    total_params = count_parameters(model)

    summary = []
    summary.append("=" * 60)
    summary.append(f"Model: {model.__class__.__name__}")
    summary.append(f"Total trainable parameters: {total_params:,}")
    summary.append("=" * 60)

    # Test forward pass
    device = next(model.parameters()).device if len(list(model.parameters())) > 0 else 'cpu'
    x = torch.randn(*input_size).to(device)

    with torch.no_grad():
        output = model(x)

    summary.append(f"Input shape: {input_size}")
    summary.append(f"Output shape: {tuple(output.shape)}")
    summary.append("=" * 60)

    return "\n".join(summary)


if __name__ == "__main__":
    # Test model creation
    print("Testing model module...")

    # Test CNN
    cnn = create_model('cnn')
    print(get_model_summary(cnn))

    # Test logistic regression
    lr = create_model('logistic')
    print(get_model_summary(lr))

    # Test forward pass
    x = torch.randn(4, 3, 224, 224)
    out = cnn(x)
    print(f"CNN output shape: {out.shape}")

    probs = cnn.predict_proba(x)
    print(f"Probabilities shape: {probs.shape}")
    print(f"Probabilities sum: {probs.sum(dim=1)}")

    print("Model tests passed!")
