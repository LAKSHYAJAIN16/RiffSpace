"""
Vector database integration for musical riff embeddings.

Converts riffs to fixed-dimensional vectors suitable for vector databases
(Pinecone, Weaviate, Chroma, FAISS, etc.)
"""

from typing import List, Optional, Dict, Tuple, Any
import numpy as np
from dataclasses import dataclass
from .riff import Riff
from .space import RiffSpace


@dataclass
class RiffEmbedding:
    """A riff encoded as a fixed-dimensional vector."""
    
    riff: Riff
    vector: np.ndarray
    metadata: Dict[str, Any]
    embedding_method: str
    
    def to_dict(self) -> dict:
        """Serialize for vector DB insertion."""
        return {
            "vector": self.vector.tolist(),
            "metadata": {
                **self.riff.metadata,
                "embedding_method": self.embedding_method,
                "riff_length": len(self.riff),
                "total_duration": self.riff.total_duration,
                "tempo": self.riff.tempo
            }
        }


class RiffVectorizer:
    """
    Convert riffs to fixed-dimensional vectors for vector databases.
    
    Problem: Vector DBs require fixed-length vectors, but riffs have variable length.
    Solution: Multiple embedding strategies that preserve musical similarity.
    """
    
    def __init__(
        self,
        method: str = "statistical",
        dimension: int = 128,
        space: Optional[RiffSpace] = None
    ):
        """
        Initialize vectorizer.
        
        Args:
            method: Embedding method ('statistical', 'histogram', 'learned', 'distance')
            dimension: Target vector dimension
            space: RiffSpace for distance-based embeddings
        """
        self.method = method
        self.dimension = dimension
        self.space = space or RiffSpace()
        
        # For distance-based embeddings
        self.reference_riffs: List[Riff] = []
    
    def fit(self, riffs: List[Riff]):
        """
        Fit the vectorizer on a corpus (for distance-based methods).
        
        Args:
            riffs: Corpus of riffs to use as reference points
        """
        if self.method == "distance":
            # Select representative riffs as landmarks
            if len(riffs) <= self.dimension:
                self.reference_riffs = riffs
            else:
                # K-means++ style selection for diversity
                self.reference_riffs = self._select_diverse_riffs(riffs, self.dimension)
        
        return self
    
    def embed(self, riff: Riff) -> RiffEmbedding:
        """
        Convert a riff to a fixed-dimensional vector.
        
        Args:
            riff: Riff to embed
            
        Returns:
            RiffEmbedding with vector and metadata
        """
        if self.method == "statistical":
            vector = self._embed_statistical(riff)
        elif self.method == "histogram":
            vector = self._embed_histogram(riff)
        elif self.method == "distance":
            vector = self._embed_distance(riff)
        elif self.method == "learned":
            vector = self._embed_learned(riff)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return RiffEmbedding(
            riff=riff,
            vector=vector,
            metadata=riff.metadata,
            embedding_method=self.method
        )
    
    def embed_batch(self, riffs: List[Riff]) -> List[RiffEmbedding]:
        """Embed multiple riffs efficiently."""
        return [self.embed(riff) for riff in riffs]
    
    # -------------------------------------------------------------------------
    # Embedding Methods
    # -------------------------------------------------------------------------
    
    def _embed_statistical(self, riff: Riff) -> np.ndarray:
        """
        Statistical features embedding.
        
        Extracts summary statistics that are invariant to riff length:
        - Interval statistics (mean, std, min, max, quartiles)
        - Rhythm statistics (duration patterns)
        - Articulation distribution
        - Higher-order moments
        """
        intervals = riff.get_interval_sequence()
        durations = riff.get_rhythm_sequence()
        
        features = []
        
        # Pitch interval statistics (15 features)
        features.extend([
            np.mean(intervals),
            np.std(intervals),
            np.min(intervals),
            np.max(intervals),
            np.median(intervals),
            np.percentile(intervals, 25),
            np.percentile(intervals, 75),
            np.sum(np.abs(intervals)),  # Total melodic motion
            np.sum(intervals > 0),  # Ascending intervals
            np.sum(intervals < 0),  # Descending intervals
            np.sum(np.abs(intervals) > 3),  # Leaps
            np.sum(np.abs(intervals) <= 2),  # Steps
            np.mean(np.abs(np.diff(intervals))),  # Interval variance
            len(np.unique(intervals)),  # Interval diversity
            np.sum(intervals == 0)  # Repeated notes
        ])
        
        # Rhythm statistics (12 features)
        features.extend([
            np.mean(durations),
            np.std(durations),
            np.min(durations),
            np.max(durations),
            np.median(durations),
            np.sum(durations < 0.5),  # Short notes
            np.sum(durations >= 1.0),  # Long notes
            np.mean(np.abs(np.diff(durations))),  # Rhythmic variation
            len(np.unique(durations)),  # Rhythmic diversity
            riff.total_duration,
            len(riff),  # Number of notes
            len(riff) / riff.total_duration if riff.total_duration > 0 else 0  # Density
        ])
        
        # Articulation distribution (8 features)
        articulations = riff.get_articulation_sequence()
        artic_types = ["normal", "palm-mute", "accent", "bend", "slide", "hammer-on", "pull-off", "harmonic"]
        for artic in artic_types:
            features.append(articulations.count(artic) / len(articulations))
        
        # Pad or truncate to target dimension
        features = np.array(features, dtype=np.float32)
        
        if len(features) < self.dimension:
            # Pad with zeros
            features = np.pad(features, (0, self.dimension - len(features)))
        elif len(features) > self.dimension:
            # PCA-style: keep most important features
            features = features[:self.dimension]
        
        return features
    
    def _embed_histogram(self, riff: Riff) -> np.ndarray:
        """
        Histogram-based embedding.
        
        Creates histograms of intervals and durations, then flattens.
        Captures distribution while being length-invariant.
        """
        intervals = riff.get_interval_sequence()
        durations = riff.get_rhythm_sequence()
        
        # Interval histogram: -12 to +12 semitones
        interval_bins = np.arange(-12, 13, 1)
        interval_hist, _ = np.histogram(intervals, bins=interval_bins, density=True)
        
        # Duration histogram: quantized to common note values
        duration_bins = [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, np.inf]
        duration_hist, _ = np.histogram(durations, bins=duration_bins, density=True)
        
        # 2D histogram: interval vs duration
        interval_duration_hist, _, _ = np.histogram2d(
            intervals,
            durations,
            bins=[interval_bins, duration_bins],
            density=True
        )
        
        # Flatten and concatenate
        features = np.concatenate([
            interval_hist,
            duration_hist,
            interval_duration_hist.flatten()
        ])
        
        # Normalize
        features = features / (np.linalg.norm(features) + 1e-8)
        
        # Resize to target dimension
        if len(features) < self.dimension:
            features = np.pad(features, (0, self.dimension - len(features)))
        else:
            features = features[:self.dimension]
        
        return features.astype(np.float32)
    
    def _embed_distance(self, riff: Riff) -> np.ndarray:
        """
        Distance-based embedding (requires fit()).
        
        Embeds riff as vector of distances to reference riffs:
        v[i] = d(riff, reference_riff[i])
        
        This preserves metric structure and works with any distance function.
        """
        if not self.reference_riffs:
            raise ValueError("Must call fit() before using distance-based embedding")
        
        distances = np.array([
            self.space.distance(riff, ref)
            for ref in self.reference_riffs
        ], dtype=np.float32)
        
        # Normalize distances to [0, 1]
        max_dist = np.max(distances) if np.max(distances) > 0 else 1.0
        distances = distances / max_dist
        
        # Pad if needed
        if len(distances) < self.dimension:
            distances = np.pad(distances, (0, self.dimension - len(distances)))
        
        return distances[:self.dimension]
    
    def _embed_learned(self, riff: Riff) -> np.ndarray:
        """
        Learned embedding using neural network (placeholder).
        
        TODO: Implement with PyTorch/TensorFlow:
        - Encoder network: variable-length riff → fixed vector
        - Train with contrastive learning (similar riffs closer)
        - Use metric learning loss
        """
        raise NotImplementedError(
            "Learned embeddings require neural network training. "
            "Use 'statistical', 'histogram', or 'distance' methods instead."
        )
    
    def _select_diverse_riffs(self, riffs: List[Riff], k: int) -> List[Riff]:
        """
        Select k diverse riffs using greedy farthest-first traversal.
        
        Args:
            riffs: Candidate riffs
            k: Number to select
            
        Returns:
            k diverse riffs
        """
        if len(riffs) <= k:
            return riffs
        
        selected = [riffs[0]]  # Start with first riff
        
        for _ in range(k - 1):
            # Find riff farthest from all selected
            max_min_dist = -1
            farthest_riff = None
            
            for riff in riffs:
                if riff in selected:
                    continue
                
                # Minimum distance to any selected riff
                min_dist = min(self.space.distance(riff, s) for s in selected)
                
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    farthest_riff = riff
            
            if farthest_riff is not None:
                selected.append(farthest_riff)
        
        return selected


class VectorDBAdapter:
    """
    Adapter for common vector databases.
    
    Provides unified interface for Pinecone, Weaviate, Chroma, FAISS, etc.
    """
    
    def __init__(
        self,
        backend: str = "faiss",
        dimension: int = 128,
        **kwargs
    ):
        """
        Initialize vector DB adapter.
        
        Args:
            backend: 'faiss', 'pinecone', 'weaviate', 'chroma', 'milvus'
            dimension: Vector dimension
            **kwargs: Backend-specific configuration
        """
        self.backend = backend
        self.dimension = dimension
        self.index = None
        
        self._initialize_backend(**kwargs)
    
    def _initialize_backend(self, **kwargs):
        """Initialize the chosen vector DB backend."""
        if self.backend == "faiss":
            self._init_faiss(**kwargs)
        elif self.backend == "pinecone":
            self._init_pinecone(**kwargs)
        elif self.backend == "chroma":
            self._init_chroma(**kwargs)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
    
    def _init_faiss(self, metric: str = "l2", **kwargs):
        """Initialize FAISS index."""
        try:
            import faiss
        except ImportError:
            raise ImportError("FAISS not installed. Run: pip install faiss-cpu")
        
        if metric == "l2":
            self.index = faiss.IndexFlatL2(self.dimension)
        elif metric == "cosine":
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product
        else:
            raise ValueError(f"Unknown FAISS metric: {metric}")
        
        self.metadata_store = []  # FAISS doesn't store metadata
    
    def _init_pinecone(self, api_key: str, environment: str, index_name: str, **kwargs):
        """Initialize Pinecone client."""
        try:
            import pinecone
        except ImportError:
            raise ImportError("Pinecone not installed. Run: pip install pinecone-client")
        
        pinecone.init(api_key=api_key, environment=environment)
        self.index = pinecone.Index(index_name)
    
    def _init_chroma(self, collection_name: str = "riffs", **kwargs):
        """Initialize ChromaDB client."""
        try:
            import chromadb
        except ImportError:
            raise ImportError("ChromaDB not installed. Run: pip install chromadb")
        
        client = chromadb.Client()
        self.index = client.get_or_create_collection(name=collection_name)
    
    def insert(self, embeddings: List[RiffEmbedding]):
        """Insert riff embeddings into vector DB."""
        if self.backend == "faiss":
            self._insert_faiss(embeddings)
        elif self.backend == "pinecone":
            self._insert_pinecone(embeddings)
        elif self.backend == "chroma":
            self._insert_chroma(embeddings)
    
    def _insert_faiss(self, embeddings: List[RiffEmbedding]):
        """Insert into FAISS."""
        vectors = np.array([emb.vector for emb in embeddings], dtype=np.float32)
        self.index.add(vectors)
        self.metadata_store.extend([emb.metadata for emb in embeddings])
    
    def _insert_pinecone(self, embeddings: List[RiffEmbedding]):
        """Insert into Pinecone."""
        records = [
            (str(i), emb.vector.tolist(), emb.metadata)
            for i, emb in enumerate(embeddings)
        ]
        self.index.upsert(records)
    
    def _insert_chroma(self, embeddings: List[RiffEmbedding]):
        """Insert into ChromaDB."""
        self.index.add(
            ids=[str(i) for i in range(len(embeddings))],
            embeddings=[emb.vector.tolist() for emb in embeddings],
            metadatas=[emb.metadata for emb in embeddings]
        )
    
    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        filter_metadata: Optional[Dict] = None
    ) -> List[Tuple[int, float, Dict]]:
        """
        Search for similar riffs.
        
        Args:
            query_vector: Query embedding
            k: Number of results
            filter_metadata: Optional metadata filters
            
        Returns:
            List of (index, distance, metadata) tuples
        """
        if self.backend == "faiss":
            return self._search_faiss(query_vector, k)
        elif self.backend == "pinecone":
            return self._search_pinecone(query_vector, k, filter_metadata)
        elif self.backend == "chroma":
            return self._search_chroma(query_vector, k, filter_metadata)
    
    def _search_faiss(self, query_vector: np.ndarray, k: int):
        """Search FAISS."""
        distances, indices = self.index.search(
            query_vector.reshape(1, -1).astype(np.float32),
            k
        )
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.metadata_store):
                results.append((int(idx), float(dist), self.metadata_store[idx]))
        
        return results
    
    def _search_pinecone(self, query_vector: np.ndarray, k: int, filter_metadata: Optional[Dict]):
        """Search Pinecone."""
        response = self.index.query(
            vector=query_vector.tolist(),
            top_k=k,
            filter=filter_metadata,
            include_metadata=True
        )
        
        return [
            (int(match['id']), match['score'], match['metadata'])
            for match in response['matches']
        ]
    
    def _search_chroma(self, query_vector: np.ndarray, k: int, filter_metadata: Optional[Dict]):
        """Search ChromaDB."""
        results = self.index.query(
            query_embeddings=[query_vector.tolist()],
            n_results=k,
            where=filter_metadata
        )
        
        return [
            (i, dist, meta)
            for i, (dist, meta) in enumerate(zip(
                results['distances'][0],
                results['metadatas'][0]
            ))
        ]


def create_riff_vector_index(
    riffs: List[Riff],
    method: str = "statistical",
    backend: str = "faiss",
    dimension: int = 128
) -> Tuple[VectorDBAdapter, RiffVectorizer]:
    """
    Create a vector index from a collection of riffs.
    
    Args:
        riffs: List of riffs to index
        method: Embedding method
        backend: Vector DB backend
        dimension: Vector dimension
        
    Returns:
        (VectorDBAdapter, RiffVectorizer) tuple
    """
    # Initialize vectorizer
    vectorizer = RiffVectorizer(method=method, dimension=dimension)
    
    # Fit if needed
    if method == "distance":
        vectorizer.fit(riffs)
    
    # Embed riffs
    embeddings = vectorizer.embed_batch(riffs)
    
    # Create vector DB
    db = VectorDBAdapter(backend=backend, dimension=dimension)
    db.insert(embeddings)
    
    return db, vectorizer
