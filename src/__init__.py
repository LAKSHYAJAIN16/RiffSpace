"""
RiffSpace: Musical Structure Retrieval via Vector Embeddings

A mathematical framework for representing, indexing, and retrieving musical content
(riffs and full songs) in vector databases.
"""

__version__ = "0.3.0"
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
from .song import Song, SongSegment, load_song, load_songs_from_directory
from .song_vectorizer import SongVectorizer, SongEmbedding, create_song_vector_index
from .pipeline import (
    extract_riff_from_midi,
    extract_riff_from_audio,
    create_synthetic_dataset,
    save_riff_collection,
    load_riff_collection
)

__all__ = [
    # Core - Riffs
    "Riff",
    "RiffNote",
    "create_example_riff",
    # Core - Songs
    "Song",
    "SongSegment",
    "load_song",
    "load_songs_from_directory",
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
    # Vector DB - Riffs
    "RiffVectorizer",
    "RiffEmbedding",
    "create_riff_vector_index",
    # Vector DB - Songs
    "SongVectorizer",
    "SongEmbedding",
    "create_song_vector_index",
    # Shared
    "VectorDBAdapter",
    # Pipeline
    "extract_riff_from_midi",
    "extract_riff_from_audio",
    "create_synthetic_dataset",
    "save_riff_collection",
    "load_riff_collection",
]
