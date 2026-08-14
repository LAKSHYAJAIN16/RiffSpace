"""
RiffSpace: Musical Structure Retrieval via Vector Embeddings

A mathematical framework for representing, indexing, and retrieving musical riffs
in vector databases.
"""

__version__ = "0.2.0"
__author__ = "RiffSpace Research"

from .riff import Riff, RiffNote, create_example_riff
from .transforms import TransformGroup, Transformation, get_standard_group
from .space import RiffSpace, RiffCollection
from .metrics import get_metric, compare_metrics
from .novelty import NoveltyAnalyzer, NoveltyScore, quantify_derivative
from .vectordb import (
    RiffVectorizer,
    RiffEmbedding,
    VectorDBAdapter,
    create_riff_vector_index
)
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
    # Vector DB
    "RiffVectorizer",
    "RiffEmbedding",
    "VectorDBAdapter",
    "create_riff_vector_index",
    # Pipeline
    "extract_riff_from_midi",
    "extract_riff_from_audio",
    "create_synthetic_dataset",
    "save_riff_collection",
    "load_riff_collection",
]
