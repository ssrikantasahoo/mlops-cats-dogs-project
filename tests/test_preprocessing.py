"""
Unit Tests for Data Preprocessing Module
"""

import os
import sys
import pytest
import numpy as np
from PIL import Image
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessing import (
    preprocess_image,
    preprocess_image_tensor,
    augment_image,
    get_train_transforms,
    get_val_transforms,
    validate_image,
    IMAGE_SIZE,
    MEAN,
    STD
)


@pytest.fixture
def sample_image():
    """Create a sample test image."""
    img = Image.new('RGB', (300, 400), color='red')
    return img


@pytest.fixture
def sample_image_path(tmp_path, sample_image):
    """Save sample image to temp path."""
    img_path = tmp_path / "test_image.jpg"
    sample_image.save(img_path)
    return str(img_path)


@pytest.fixture
def grayscale_image():
    """Create a grayscale test image."""
    img = Image.new('L', (200, 200), color=128)
    return img


class TestPreprocessImage:
    """Tests for preprocess_image function."""

    def test_preprocess_image_output_shape(self, sample_image_path):
        """Test that output has correct shape."""
        result = preprocess_image(sample_image_path)
        assert result.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)

    def test_preprocess_image_normalization(self, sample_image_path):
        """Test that normalized values are in expected range."""
        result = preprocess_image(sample_image_path, normalize=True)
        # After normalization, values should be roughly centered around 0
        assert result.min() < 0 or result.max() > 1

    def test_preprocess_image_no_normalization(self, sample_image_path):
        """Test that without normalization, values are in [0, 1]."""
        result = preprocess_image(sample_image_path, normalize=False)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_preprocess_image_custom_size(self, sample_image_path):
        """Test custom target size."""
        target_size = (128, 128)
        result = preprocess_image(sample_image_path, target_size=target_size)
        assert result.shape[:2] == target_size

    def test_preprocess_image_dtype(self, sample_image_path):
        """Test output dtype is float32."""
        result = preprocess_image(sample_image_path)
        assert result.dtype == np.float32


class TestPreprocessImageTensor:
    """Tests for preprocess_image_tensor function."""

    def test_output_is_tensor(self, sample_image):
        """Test that output is a PyTorch tensor."""
        result = preprocess_image_tensor(sample_image)
        assert isinstance(result, torch.Tensor)

    def test_tensor_shape(self, sample_image):
        """Test tensor has correct shape (C, H, W)."""
        result = preprocess_image_tensor(sample_image)
        assert result.shape == (3, IMAGE_SIZE, IMAGE_SIZE)

    def test_tensor_dtype(self, sample_image):
        """Test tensor dtype is float32."""
        result = preprocess_image_tensor(sample_image)
        assert result.dtype == torch.float32

    def test_grayscale_conversion(self, grayscale_image):
        """Test grayscale image is converted to RGB."""
        result = preprocess_image_tensor(grayscale_image)
        assert result.shape[0] == 3  # RGB channels

    def test_custom_size(self, sample_image):
        """Test custom target size."""
        target_size = (128, 128)
        result = preprocess_image_tensor(sample_image, target_size=target_size)
        assert result.shape == (3, 128, 128)


class TestAugmentImage:
    """Tests for augment_image function."""

    def test_output_is_pil_image(self, sample_image):
        """Test that output is a PIL Image."""
        result = augment_image(sample_image)
        assert isinstance(result, Image.Image)

    def test_output_same_size(self, sample_image):
        """Test that augmented image has same size."""
        result = augment_image(sample_image)
        assert result.size == sample_image.size

    def test_output_is_rgb(self, sample_image):
        """Test that output is RGB."""
        result = augment_image(sample_image)
        assert result.mode == 'RGB'


class TestTransforms:
    """Tests for transform functions."""

    def test_train_transforms_output_shape(self, sample_image):
        """Test train transforms produce correct shape."""
        transforms = get_train_transforms()
        result = transforms(sample_image)
        assert result.shape == (3, IMAGE_SIZE, IMAGE_SIZE)

    def test_val_transforms_output_shape(self, sample_image):
        """Test validation transforms produce correct shape."""
        transforms = get_val_transforms()
        result = transforms(sample_image)
        assert result.shape == (3, IMAGE_SIZE, IMAGE_SIZE)

    def test_val_transforms_deterministic(self, sample_image):
        """Test that validation transforms are deterministic."""
        transforms = get_val_transforms()
        result1 = transforms(sample_image)
        result2 = transforms(sample_image)
        assert torch.allclose(result1, result2)


class TestValidateImage:
    """Tests for validate_image function."""

    def test_valid_image(self, sample_image_path):
        """Test validation of valid image."""
        assert validate_image(sample_image_path) is True

    def test_invalid_path(self):
        """Test validation of non-existent file."""
        assert validate_image("nonexistent.jpg") is False

    def test_corrupted_file(self, tmp_path):
        """Test validation of corrupted file."""
        corrupted_path = tmp_path / "corrupted.jpg"
        with open(corrupted_path, 'wb') as f:
            f.write(b"not an image")
        assert validate_image(str(corrupted_path)) is False


class TestConstants:
    """Tests for module constants."""

    def test_image_size_positive(self):
        """Test IMAGE_SIZE is positive."""
        assert IMAGE_SIZE > 0
        assert IMAGE_SIZE == 224  # Standard CNN input

    def test_mean_values(self):
        """Test MEAN values are valid."""
        assert len(MEAN) == 3
        for v in MEAN:
            assert 0 <= v <= 1

    def test_std_values(self):
        """Test STD values are valid."""
        assert len(STD) == 3
        for v in STD:
            assert 0 < v <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
