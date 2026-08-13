"""
RiffSpace: The Geometry and Evolution of Rock Riffs

A mathematical framework for analyzing rock riffs as points in geometric space.
"""

__version__ = "0.1.0"
__author__ = "RiffSpace Research"

from .riff import Riff, RiffNote, create_example_riff
from .transforms import TransformGroup, Transformation, get_standard_group
from .space import RiffSpace, RiffCollection
from .metrics import get_metric, compare_metrics
from .novelty import NoveltyAnalyzer, NoveltyScore, quantify_derivative
from .pipeline import (
    extract_riff_from_midi,
    extract_riff_from_audio,
    create_synthetic_dataset,
    save_riff_collection,
    load_riff_collection
)

__all__ = [
    # Core
    "Riff",
    "RiffNote",
    "create_example_riff",
    # Transformations
    "TransformGroup",
    "Transformation",
    "get_standard_group",
    # Space
    "RiffSpace",
    "RiffCollection",
    # Metrics
    "get_metric",
    "compare_metrics",
    # Novelty
    "NoveltyAnalyzer",
    "NoveltyScore",
    "quantify_derivative",
    # Pipeline
    "extract_riff_from_midi",
    "extract_riff_from_audio",
    "create_synthetic_dataset",
    "save_riff_collection",
    "load_riff_collection",
]
