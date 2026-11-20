"""SIMPLE Score: SImplicity Metric via Preference LEarning.

A reference-free learned metric for text simplification. For each (source, candidate)
pair it returns a quality score ``mu`` in (-1, 1) and the model's uncertainty ``sigma``.

    >>> from simple_score import SIMPLE
    >>> metric = SIMPLE.from_pretrained("large")
    >>> metric.score(["The feline perched upon the mat."], ["The cat sat on the mat."])
"""

from ._version import __version__
from .losses import (
    bradley_terry_loss,
    gaussian_nll_loss,
    mse_loss,
    preference_probability,
    thurstone_loss,
)
from .modeling import SimpleConfig, SimpleModel
from .scorer import SIMPLE, ScoreOutput, load

__all__ = [
    "SIMPLE",
    "ScoreOutput",
    "SimpleConfig",
    "SimpleModel",
    "__version__",
    "bradley_terry_loss",
    "gaussian_nll_loss",
    "load",
    "mse_loss",
    "preference_probability",
    "thurstone_loss",
]
