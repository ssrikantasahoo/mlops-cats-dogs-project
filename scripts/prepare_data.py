"""
Data Preparation Script
Downloads and preprocesses the Cats vs Dogs dataset.
"""

import os
import sys
import argparse
import shutil
import random
from pathlib import Path
from PIL import Image
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_params():
    """Load parameters from params.yaml."""
    params_path = Path(__file__).parent.parent / 'params.yaml'
    if params_path.exists():
        with open(params_path, 'r') as f:
            return yaml.safe_load(f)
    return {
        'prepare': {
            'image_size': 224,
            'train_ratio': 0.8,
            'val_ratio': 0.1,
            'test_ratio': 0.1,
            'seed': 42
        }
    }


def download_dataset(output_dir: str):
    """
    Download Cats vs Dogs dataset from Kaggle.
    Requires Kaggle API credentials.
    """
    try:
        import kaggle
        logger.info("Downloading dataset from Kaggle...")
        kaggle.api.dataset_download_files(
            'tongpython/cat-and-dog',
            path=output_dir,
            unzip=True
        )
        logger.info("Download complete!")
    except Exception as e:
        logger.warning(f"Kaggle download failed: {e}")
        logger.info("Creating sample dataset for demonstration...")
        create_sample_dataset(output_dir)


def create_sample_dataset(output_dir: str, num_samples: int = 200):
    """Create a sample dataset for testing."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cats_dir = output_path / 'training_set' / 'cats'
    dogs_dir = output_path / 'training_set' / 'dogs'
    cats_dir.mkdir(parents=True, exist_ok=True)
    dogs_dir.mkdir(parents=True, exist_ok=True)

    # Create sample cat images
    for i in range(num_samples // 2):
        img = Image.new('RGB', (300, 300),
                       color=(random.randint(100, 200), random.randint(50, 150), random.randint(50, 150)))
        img.save(cats_dir / f"cat.{i}.jpg")

    # Create sample dog images
    for i in range(num_samples // 2):
        img = Image.new('RGB', (300, 300),
                       color=(random.randint(50, 150), random.randint(100, 200), random.randint(50, 150)))
        img.save(dogs_dir / f"dog.{i}.jpg")

    logger.info(f"Created sample dataset with {num_samples} images")


def preprocess_and_split(
    raw_dir: str,
    processed_dir: str,
    image_size: int = 224,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
):
    """
    Preprocess images and split into train/val/test sets.
    """
    random.seed(seed)

    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)

    # Create output directories
    for split in ['train', 'val', 'test']:
        for cls in ['cats', 'dogs']:
            (processed_path / split / cls).mkdir(parents=True, exist_ok=True)

    # Find all images
    all_images = {'cats': [], 'dogs': []}

    # Search for images in various possible structures
    for pattern in ['**/cat*.jpg', '**/cat*.png', '**/cats/*.jpg', '**/cats/*.png']:
        all_images['cats'].extend(raw_path.glob(pattern))

    for pattern in ['**/dog*.jpg', '**/dog*.png', '**/dogs/*.jpg', '**/dogs/*.png']:
        all_images['dogs'].extend(raw_path.glob(pattern))

    # Remove duplicates
    all_images['cats'] = list(set(all_images['cats']))
    all_images['dogs'] = list(set(all_images['dogs']))

    logger.info(f"Found {len(all_images['cats'])} cat images")
    logger.info(f"Found {len(all_images['dogs'])} dog images")

    stats = {'total': 0, 'train': 0, 'val': 0, 'test': 0, 'errors': 0}

    for category, images in all_images.items():
        # Shuffle images
        random.shuffle(images)

        # Split
        n = len(images)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        splits = {
            'train': images[:train_end],
            'val': images[train_end:val_end],
            'test': images[val_end:]
        }

        for split_name, split_images in splits.items():
            for img_path in split_images:
                try:
                    # Load and resize image
                    img = Image.open(img_path)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img = img.resize((image_size, image_size), Image.Resampling.BILINEAR)

                    # Save processed image
                    output_path = processed_path / split_name / category / img_path.name
                    # Ensure unique filename
                    if output_path.exists():
                        output_path = processed_path / split_name / category / f"{img_path.stem}_{random.randint(1000, 9999)}.jpg"

                    img.save(output_path, 'JPEG', quality=95)
                    stats['total'] += 1
                    stats[split_name] += 1

                except Exception as e:
                    logger.warning(f"Error processing {img_path}: {e}")
                    stats['errors'] += 1

    logger.info(f"Processing complete: {stats}")
    return stats


def main():
    parser = argparse.ArgumentParser(description='Prepare Cats vs Dogs dataset')
    parser.add_argument('--raw-dir', type=str, default='data/raw',
                       help='Directory containing raw data')
    parser.add_argument('--processed-dir', type=str, default='data/processed',
                       help='Directory for processed data')
    parser.add_argument('--download', action='store_true',
                       help='Download dataset from Kaggle')

    args = parser.parse_args()

    # Load parameters
    params = load_params()['prepare']

    # Download if requested or if raw data doesn't exist
    raw_path = Path(args.raw_dir)
    if args.download or not raw_path.exists() or not any(raw_path.glob('**/*.jpg')):
        download_dataset(args.raw_dir)

    # Preprocess and split
    preprocess_and_split(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        image_size=params['image_size'],
        train_ratio=params['train_ratio'],
        val_ratio=params['val_ratio'],
        test_ratio=params['test_ratio'],
        seed=params['seed']
    )


if __name__ == "__main__":
    main()
