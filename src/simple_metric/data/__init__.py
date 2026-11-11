"""Generic data handling: read plain files and derive SIMPLE's training signal."""

from .builders import (
    INTER_TYPES,
    KINDS,
    AugmentationOptions,
    DataSpec,
    TrainingData,
    ValidationOptions,
    build_training_data,
    pairs_from_graded,
    pairs_from_parallel,
    pairs_from_preferences,
    pairs_from_ratings,
    rating_targets,
)
from .io import DataFormatError, read_records, write_jsonl

__all__ = [
    "INTER_TYPES",
    "KINDS",
    "AugmentationOptions",
    "DataFormatError",
    "DataSpec",
    "TrainingData",
    "ValidationOptions",
    "build_training_data",
    "pairs_from_graded",
    "pairs_from_parallel",
    "pairs_from_preferences",
    "pairs_from_ratings",
    "rating_targets",
    "read_records",
    "write_jsonl",
]
