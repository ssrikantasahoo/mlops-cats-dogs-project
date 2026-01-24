"""
PyTorch Dataset for Cats vs Dogs Classification
"""

import os
from pathlib import Path
from typing import Tuple, Optional, Callable, List
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CatsDogsDataset(Dataset):
    """
    Custom PyTorch Dataset for Cats vs Dogs binary classification.

    Labels:
        - 0: Cat
        - 1: Dog
    """

    def __init__(
        self,
        data_dir: str,
        transform: Optional[Callable] = None,
        split: str = 'train'
    ):
        """
        Initialize the dataset.

        Args:
            data_dir: Root directory containing the dataset
            transform: Optional transforms to apply to images
            split: One of 'train', 'val', 'test'
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.split = split
        self.classes = ['cats', 'dogs']  # 0 = cat, 1 = dog
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Collect image paths and labels
        self.samples: List[Tuple[str, int]] = []
        self._load_samples()

        logger.info(f"Loaded {len(self.samples)} images for {split} split")

    def _load_samples(self):
        """Load all image paths and their corresponding labels."""
        split_dir = self.data_dir / self.split

        if not split_dir.exists():
            # Try loading from flat structure
            self._load_flat_structure()
            return

        for class_name in self.classes:
            class_dir = split_dir / class_name

            if not class_dir.exists():
                logger.warning(f"Class directory not found: {class_dir}")
                continue

            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))

    def _load_flat_structure(self):
        """Load from flat directory structure with naming convention."""
        for img_path in self.data_dir.iterdir():
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                name_lower = img_path.name.lower()
                if 'cat' in name_lower:
                    self.samples.append((str(img_path), 0))
                elif 'dog' in name_lower:
                    self.samples.append((str(img_path), 1))

    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Tuple of (image_tensor, label)
        """
        img_path, label = self.samples[idx]

        # Load image
        image = Image.open(img_path)

        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Default transform
            default_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            image = default_transform(image)

        return image, label

    def get_class_distribution(self) -> dict:
        """Get the distribution of classes in the dataset."""
        distribution = {cls: 0 for cls in self.classes}
        for _, label in self.samples:
            distribution[self.classes[label]] += 1
        return distribution


class CatsDogsInferenceDataset(Dataset):
    """
    Dataset for inference on unlabeled images.
    """

    def __init__(
        self,
        image_paths: List[str],
        transform: Optional[Callable] = None
    ):
        """
        Initialize the inference dataset.

        Args:
            image_paths: List of paths to images
            transform: Optional transforms to apply
        """
        self.image_paths = image_paths
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image, img_path


def create_sample_dataset(output_dir: str, num_samples: int = 100):
    """
    Create a sample dataset for testing (synthetic images).

    Args:
        output_dir: Directory to save sample images
        num_samples: Number of samples per class
    """
    import random

    output_path = Path(output_dir)

    for split in ['train', 'val', 'test']:
        for cls in ['cats', 'dogs']:
            (output_path / split / cls).mkdir(parents=True, exist_ok=True)

    # Determine samples per split
    train_samples = int(num_samples * 0.8)
    val_samples = int(num_samples * 0.1)
    test_samples = num_samples - train_samples - val_samples

    for cls_idx, cls in enumerate(['cats', 'dogs']):
        for split, count in [('train', train_samples), ('val', val_samples), ('test', test_samples)]:
            for i in range(count):
                # Create synthetic image
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                img = Image.new('RGB', (224, 224), color=color)

                # Save image
                img_path = output_path / split / cls / f"{cls[:-1]}_{i:04d}.jpg"
                img.save(img_path)

    logger.info(f"Created sample dataset at {output_dir}")


if __name__ == "__main__":
    # Test dataset creation
    print("Testing dataset module...")

    # Create sample dataset
    create_sample_dataset("data/test_samples", num_samples=20)

    # Test loading
    dataset = CatsDogsDataset("data/test_samples", split='train')
    print(f"Dataset size: {len(dataset)}")
    print(f"Class distribution: {dataset.get_class_distribution()}")

    if len(dataset) > 0:
        img, label = dataset[0]
        print(f"Sample image shape: {img.shape}, label: {label}")

    # Cleanup
    import shutil
    shutil.rmtree("data/test_samples")
    print("Dataset tests passed!")
