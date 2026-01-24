"""
Data Preprocessing Module for Cats vs Dogs Classification
Handles image preprocessing, augmentation, and data loading.
"""

import os
import shutil
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]  # ImageNet normalization
STD = [0.229, 0.224, 0.225]


def preprocess_image(
    image_path: str,
    target_size: Tuple[int, int] = (IMAGE_SIZE, IMAGE_SIZE),
    normalize: bool = True
) -> np.ndarray:
    """
    Preprocess a single image for model inference.

    Args:
        image_path: Path to the image file
        target_size: Target size (height, width) for resizing
        normalize: Whether to apply ImageNet normalization

    Returns:
        Preprocessed image as numpy array
    """
    # Load image
    image = Image.open(image_path)

    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Resize to target size
    image = image.resize(target_size, Image.Resampling.BILINEAR)

    # Convert to numpy array
    image_array = np.array(image, dtype=np.float32) / 255.0

    # Normalize if requested
    if normalize:
        image_array = (image_array - np.array(MEAN)) / np.array(STD)

    return image_array


def preprocess_image_tensor(
    image: Image.Image,
    target_size: Tuple[int, int] = (IMAGE_SIZE, IMAGE_SIZE)
) -> torch.Tensor:
    """
    Preprocess PIL Image to tensor for model inference.

    Args:
        image: PIL Image object
        target_size: Target size (height, width) for resizing

    Returns:
        Preprocessed image as torch tensor
    """
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    if image.mode != 'RGB':
        image = image.convert('RGB')

    return transform(image)


def augment_image(image: Image.Image) -> Image.Image:
    """
    Apply data augmentation to an image for training.

    Args:
        image: PIL Image object

    Returns:
        Augmented PIL Image
    """
    augmentation = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    ])

    return augmentation(image)


def get_train_transforms() -> transforms.Compose:
    """Get training data transforms with augmentation."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])


def get_val_transforms() -> transforms.Compose:
    """Get validation/test data transforms (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])


def create_data_loaders(
    dataset,
    batch_size: int = 32,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    num_workers: int = 4,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.

    Args:
        dataset: PyTorch dataset
        batch_size: Batch size for data loaders
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        test_ratio: Ratio of data for testing
        num_workers: Number of worker processes
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, \
        "Ratios must sum to 1.0"

    # Calculate sizes
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    logger.info(f"Dataset split: Train={train_size}, Val={val_size}, Test={test_size}")

    # Set seed for reproducibility
    generator = torch.Generator().manual_seed(seed)

    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


def organize_dataset(
    raw_data_path: str,
    processed_data_path: str,
    image_size: Tuple[int, int] = (IMAGE_SIZE, IMAGE_SIZE)
) -> dict:
    """
    Organize and preprocess raw dataset into structured format.

    Args:
        raw_data_path: Path to raw dataset
        processed_data_path: Path for processed output
        image_size: Target image size

    Returns:
        Dictionary with dataset statistics
    """
    raw_path = Path(raw_data_path)
    processed_path = Path(processed_data_path)

    # Create output directories
    for split in ['train', 'val', 'test']:
        for cls in ['cats', 'dogs']:
            (processed_path / split / cls).mkdir(parents=True, exist_ok=True)

    stats = {'total': 0, 'cats': 0, 'dogs': 0, 'errors': 0}

    # Process images
    for category in ['cats', 'dogs']:
        category_path = raw_path / category if (raw_path / category).exists() else raw_path

        # Find images for this category
        image_files = list(category_path.glob('*.jpg')) + list(category_path.glob('*.png'))
        image_files = [f for f in image_files if category.lower()[:-1] in f.name.lower()]

        for img_path in image_files:
            try:
                # Load and preprocess
                img = Image.open(img_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img = img.resize(image_size, Image.Resampling.BILINEAR)

                # Determine split (80/10/10)
                import random
                r = random.random()
                if r < 0.8:
                    split = 'train'
                elif r < 0.9:
                    split = 'val'
                else:
                    split = 'test'

                # Save processed image
                output_path = processed_path / split / category / img_path.name
                img.save(output_path, 'JPEG', quality=95)

                stats['total'] += 1
                stats[category] += 1

            except Exception as e:
                logger.warning(f"Error processing {img_path}: {e}")
                stats['errors'] += 1

    logger.info(f"Dataset organization complete: {stats}")
    return stats


def validate_image(image_path: str) -> bool:
    """
    Validate if an image file is valid and can be opened.

    Args:
        image_path: Path to the image file

    Returns:
        True if valid, False otherwise
    """
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False


if __name__ == "__main__":
    # Test preprocessing
    print("Testing preprocessing module...")

    # Create a test image
    test_img = Image.new('RGB', (300, 300), color='red')
    test_path = 'test_image.jpg'
    test_img.save(test_path)

    # Test preprocess_image
    processed = preprocess_image(test_path)
    print(f"Processed image shape: {processed.shape}")

    # Test preprocess_image_tensor
    tensor = preprocess_image_tensor(test_img)
    print(f"Tensor shape: {tensor.shape}")

    # Cleanup
    os.remove(test_path)
    print("Preprocessing tests passed!")
