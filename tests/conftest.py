"""
Pytest configuration and fixtures
"""

import os
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def device():
    """Return the device to use for testing."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.fixture
def sample_image():
    """Create a sample PIL Image."""
    return Image.new('RGB', (300, 400), color='red')


@pytest.fixture
def sample_tensor():
    """Create a sample image tensor."""
    return torch.randn(1, 3, 224, 224)


@pytest.fixture
def batch_tensor():
    """Create a batch of image tensors."""
    return torch.randn(4, 3, 224, 224)


@pytest.fixture
def temp_model_dir(tmp_path):
    """Create a temporary directory for model files."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary directory with sample data."""
    data_dir = tmp_path / "data"

    # Create directory structure
    for split in ['train', 'val', 'test']:
        for cls in ['cats', 'dogs']:
            (data_dir / split / cls).mkdir(parents=True)

            # Create sample images
            for i in range(5):
                img = Image.new('RGB', (224, 224), color='blue')
                img.save(data_dir / split / cls / f"{cls[:-1]}_{i}.jpg")

    return data_dir


@pytest.fixture
def mock_model():
    """Create a mock trained model."""
    from src.models.cnn_model import CatsDogsCNN
    model = CatsDogsCNN(num_classes=2)
    return model


def pytest_configure(config):
    """Pytest configuration hook."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    # Skip slow tests unless --runslow is given
    if not config.getoption("--runslow", default=False):
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
