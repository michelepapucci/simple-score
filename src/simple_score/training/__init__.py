"""Training SIMPLE on your own data. See ``simple-score train --help`` and the README."""

from .config import Config, ModelConfig, TrainingConfig, config_from_dict, load_config
from .trainer import evaluate, export_training_data, train

__all__ = [
    "Config",
    "ModelConfig",
    "TrainingConfig",
    "config_from_dict",
    "evaluate",
    "export_training_data",
    "load_config",
    "train",
]
