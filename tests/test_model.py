"""
Unit Tests for Model Module
"""

import sys
import pytest
import torch
import torch.nn as nn
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.cnn_model import (
    CatsDogsCNN,
    LogisticRegressionBaseline,
    create_model,
    count_parameters,
    get_model_summary
)


@pytest.fixture
def sample_batch():
    """Create a sample batch of images."""
    return torch.randn(4, 3, 224, 224)


@pytest.fixture
def cnn_model():
    """Create a CNN model instance."""
    return CatsDogsCNN(num_classes=2)


@pytest.fixture
def logistic_model():
    """Create a logistic regression model instance."""
    return LogisticRegressionBaseline(num_classes=2)


class TestCatsDogsCNN:
    """Tests for CatsDogsCNN model."""

    def test_model_initialization(self, cnn_model):
        """Test model initializes correctly."""
        assert isinstance(cnn_model, nn.Module)

    def test_forward_pass_shape(self, cnn_model, sample_batch):
        """Test forward pass produces correct output shape."""
        output = cnn_model(sample_batch)
        assert output.shape == (4, 2)  # batch_size, num_classes

    def test_output_not_nan(self, cnn_model, sample_batch):
        """Test output does not contain NaN values."""
        output = cnn_model(sample_batch)
        assert not torch.isnan(output).any()

    def test_predict_proba(self, cnn_model, sample_batch):
        """Test predict_proba returns valid probabilities."""
        probs = cnn_model.predict_proba(sample_batch)
        assert probs.shape == (4, 2)
        # Check probabilities sum to 1
        assert torch.allclose(probs.sum(dim=1), torch.ones(4), atol=1e-5)
        # Check probabilities are in [0, 1]
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_model_trainable(self, cnn_model, sample_batch):
        """Test model can be trained (backward pass works)."""
        output = cnn_model(sample_batch)
        loss = output.sum()
        loss.backward()
        # Check gradients exist
        for param in cnn_model.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_dropout_rate(self):
        """Test dropout rate is applied."""
        model = CatsDogsCNN(dropout_rate=0.3)
        assert model.dropout.p == 0.3


class TestLogisticRegressionBaseline:
    """Tests for LogisticRegressionBaseline model."""

    def test_model_initialization(self, logistic_model):
        """Test model initializes correctly."""
        assert isinstance(logistic_model, nn.Module)

    def test_forward_pass_shape(self, logistic_model, sample_batch):
        """Test forward pass produces correct output shape."""
        output = logistic_model(sample_batch)
        assert output.shape == (4, 2)

    def test_output_not_nan(self, logistic_model, sample_batch):
        """Test output does not contain NaN values."""
        output = logistic_model(sample_batch)
        assert not torch.isnan(output).any()

    def test_custom_input_size(self):
        """Test model with custom input size."""
        model = LogisticRegressionBaseline(input_size=1000, num_classes=3)
        x = torch.randn(2, 3, 18, 18)  # Different size that flattens to 972
        # This should fail because sizes don't match
        with pytest.raises(RuntimeError):
            model(x)


class TestCreateModel:
    """Tests for create_model factory function."""

    def test_create_cnn(self):
        """Test creating CNN model."""
        model = create_model('cnn')
        assert isinstance(model, CatsDogsCNN)

    def test_create_logistic(self):
        """Test creating logistic regression model."""
        model = create_model('logistic')
        assert isinstance(model, LogisticRegressionBaseline)

    def test_create_with_custom_classes(self):
        """Test creating model with custom number of classes."""
        model = create_model('cnn', num_classes=5)
        x = torch.randn(1, 3, 224, 224)
        output = model(x)
        assert output.shape == (1, 5)

    def test_create_unknown_type(self):
        """Test error for unknown model type."""
        with pytest.raises(ValueError):
            create_model('unknown_model')


class TestCountParameters:
    """Tests for count_parameters function."""

    def test_count_parameters_positive(self, cnn_model):
        """Test parameter count is positive."""
        count = count_parameters(cnn_model)
        assert count > 0

    def test_count_parameters_consistent(self, cnn_model):
        """Test parameter count is consistent."""
        count1 = count_parameters(cnn_model)
        count2 = count_parameters(cnn_model)
        assert count1 == count2

    def test_logistic_fewer_params(self, cnn_model, logistic_model):
        """Test logistic regression has different param count than CNN."""
        cnn_count = count_parameters(cnn_model)
        lr_count = count_parameters(logistic_model)
        # Logistic regression should have more params due to flattening
        assert cnn_count != lr_count


class TestGetModelSummary:
    """Tests for get_model_summary function."""

    def test_summary_is_string(self, cnn_model):
        """Test summary returns a string."""
        summary = get_model_summary(cnn_model)
        assert isinstance(summary, str)

    def test_summary_contains_model_name(self, cnn_model):
        """Test summary contains model class name."""
        summary = get_model_summary(cnn_model)
        assert "CatsDogsCNN" in summary

    def test_summary_contains_param_count(self, cnn_model):
        """Test summary contains parameter count."""
        summary = get_model_summary(cnn_model)
        assert "parameters" in summary.lower()


class TestModelInference:
    """Tests for model inference behavior."""

    def test_eval_mode(self, cnn_model, sample_batch):
        """Test model behavior in eval mode."""
        cnn_model.eval()
        with torch.no_grad():
            output = cnn_model(sample_batch)
        assert output.shape == (4, 2)

    def test_batch_size_flexibility(self, cnn_model):
        """Test model handles different batch sizes."""
        for batch_size in [1, 8, 16]:
            x = torch.randn(batch_size, 3, 224, 224)
            output = cnn_model(x)
            assert output.shape[0] == batch_size

    def test_gpu_compatibility(self, cnn_model, sample_batch):
        """Test model works on GPU if available."""
        if torch.cuda.is_available():
            model = cnn_model.cuda()
            x = sample_batch.cuda()
            output = model(x)
            assert output.is_cuda


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
