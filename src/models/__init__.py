# Models module
from .cnn_model import CatsDogsCNN, create_model

__all__ = ['CatsDogsCNN', 'create_model']

# Optional training imports (only when mlflow is available)
try:
    from .train import train_model, evaluate_model
    __all__.extend(['train_model', 'evaluate_model'])
except ImportError:
    pass
