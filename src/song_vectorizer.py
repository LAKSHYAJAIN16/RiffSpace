"""
Vectorization for complete songs - fixed-dimensional embeddings for variable-length audio.

Multiple approaches:
1. Statistical aggregation of frame-level features
2. Bag-of-audio-words (clustering audio frames)
3. Learned embeddings (neural networks)
4. Structural embeddings (segment-based)
"""

from typing import List, Optional, Dict
import numpy as np
from dataclasses import dataclass
from .song import Song


@dataclass
class SongEmbedding:
    """A song encoded as a fixed-dimensional vector."""
    
    song: Song
    vector: np.ndarray
    metadata: Dict
    embedding_method: str
    
    def to_dict(self) -> dict:
        """Serialize for vector DB insertion."""
        return {
            "vector": self.vector.tolist(),
            "metadata": {
                **self.song.metadata,
                "embedding_method": self.embedding_method,
                "duration": self.song.duration
            }
        }


class SongVectorizer:
    """
    Convert full songs to fixed-dimensional vectors for vector databases.
    
    Challenge: Songs are 3-5 minutes of audio (millions of samples)
    Solution: Multi-scale feature aggregation
    """
    
    def __init__(
        self,
        method: str = "statistical",
        dimension: int = 512,
        n_mfcc: int = 20
    ):
        """
        Initialize song vectorizer.
        
        Args:
            method: 'statistical', 'bag_of_frames', 'learned', 'openl3'
            dimension: Target vector dimension
            n_mfcc: Number of MFCC coefficients
        """
        self.method = method
        self.dimension = dimension
        self.n_mfcc = n_mfcc
        
        # For bag-of-frames method
        self.codebook = None
        self.n_clusters = min(256, dimension)
    
    def embed(self, song: Song) -> SongEmbedding:
        """
        Convert a song to a fixed-dimensional vector.
        
        Args:
            song: Song object
            
        Returns:
            SongEmbedding with vector and metadata
        """
        if self.method == "statistical":
            vector = self._embed_statistical(song)
        elif self.method == "bag_of_frames":
            vector = self._embed_bag_of_frames(song)
        elif self.method == "openl3":
            vector = self._embed_openl3(song)
        elif self.method == "learned":
            vector = self._embed_learned(song)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return SongEmbedding(
            song=song,
            vector=vector,
            metadata=song.metadata,
            embedding_method=self.method
        )
    
    def embed_batch(self, songs: List[Song]) -> List[SongEmbedding]:
        """Embed multiple songs."""
        return [self.embed(song) for song in songs]
    
    # -------------------------------------------------------------------------
    # Embedding Methods
    # -------------------------------------------------------------------------
    
    def _embed_statistical(self, song: Song) -> np.ndarray:
        """
        Statistical aggregation of audio features.
        
        Extracts frame-level features (MFCCs, chroma, etc.) and computes
        summary statistics (mean, std, min, max, quartiles) over time.
        
        This is the most common approach for audio embeddings.
        """
        features = []
        
        # 1. Global features (6 features)
        global_feats = song.extract_global_features()
        features.extend([
            global_feats['tempo'],
            global_feats['duration'],
            global_feats['energy_mean'],
            global_feats['energy_std'],
            global_feats['zcr_mean'],
            global_feats['spectral_centroid']
        ])
        
        # 2. MFCC statistics (n_mfcc * 5 = 100 features for n_mfcc=20)
        mfcc = song.extract_mfcc(n_mfcc=self.n_mfcc)
        for i in range(self.n_mfcc):
            features.extend([
                np.mean(mfcc[i]),
                np.std(mfcc[i]),
                np.median(mfcc[i]),
                np.min(mfcc[i]),
                np.max(mfcc[i])
            ])
        
        # 3. Chroma statistics (12 * 4 = 48 features)
        chroma = song.extract_chroma()
        for i in range(12):
            features.extend([
                np.mean(chroma[i]),
                np.std(chroma[i]),
                np.max(chroma[i]),
                np.sum(chroma[i] > 0.5)  # Presence of pitch class
            ])
        
        # 4. Spectral contrast statistics (7 * 3 = 21 features)
        spectral_contrast = song.extract_spectral_contrast()
        for i in range(7):
            features.extend([
                np.mean(spectral_contrast[i]),
                np.std(spectral_contrast[i]),
                np.max(spectral_contrast[i])
            ])
        
        # 5. Tonnetz statistics (6 * 2 = 12 features)
        tonnetz = song.extract_tonnetz()
        for i in range(6):
            features.extend([
                np.mean(tonnetz[i]),
                np.std(tonnetz[i])
            ])
        
        # Convert to array
        features = np.array(features, dtype=np.float32)
        
        # Total: 6 + 100 + 48 + 21 + 12 = 187 features
        
        # Pad or truncate to target dimension
        if len(features) < self.dimension:
            features = np.pad(features, (0, self.dimension - len(features)))
        else:
            features = features[:self.dimension]
        
        # Normalize
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        return features
    
    def _embed_bag_of_frames(self, song: Song) -> np.ndarray:
        """
        Bag-of-audio-words embedding.
        
        1. Extract frame-level features (MFCCs)
        2. Quantize frames to a codebook (k-means clustering)
        3. Build histogram of codeword frequencies
        
        This is order-invariant (like bag-of-words for text).
        """
        # Extract MFCCs
        mfcc = song.extract_mfcc(n_mfcc=self.n_mfcc)
        
        # Transpose to (n_frames, n_mfcc)
        frames = mfcc.T
        
        if self.codebook is None:
            raise ValueError("Must call fit() before using bag_of_frames method")
        
        # Quantize frames to nearest codeword
        from sklearn.metrics.pairwise import euclidean_distances
        distances = euclidean_distances(frames, self.codebook)
        assignments = np.argmin(distances, axis=1)
        
        # Build histogram
        histogram, _ = np.histogram(assignments, bins=np.arange(self.n_clusters + 1))
        
        # Normalize
        histogram = histogram.astype(np.float32)
        histogram = histogram / (np.sum(histogram) + 1e-8)
        
        # Pad to dimension
        if len(histogram) < self.dimension:
            histogram = np.pad(histogram, (0, self.dimension - len(histogram)))
        
        return histogram
    
    def _embed_openl3(self, song: Song) -> np.ndarray:
        """
        OpenL3 pre-trained embeddings.
        
        Uses a pre-trained neural network (trained on AudioSet) to extract
        high-level audio representations.
        
        Requires: pip install openl3
        """
        try:
            import openl3
        except ImportError:
            raise ImportError("OpenL3 not installed. Run: pip install openl3")
        
        y = song.audio
        sr = song.sr
        
        # Extract embeddings (512-d by default)
        emb, ts = openl3.get_audio_embedding(y, sr, content_type="music")
        
        # Aggregate over time (mean pooling)
        embedding = np.mean(emb, axis=0).astype(np.float32)
        
        # Resize to target dimension if needed
        if len(embedding) != self.dimension:
            if len(embedding) < self.dimension:
                embedding = np.pad(embedding, (0, self.dimension - len(embedding)))
            else:
                embedding = embedding[:self.dimension]
        
        return embedding
    
    def _embed_learned(self, song: Song) -> np.ndarray:
        """
        Learned embedding using neural network.
        
        Would train a CNN/RNN/Transformer on audio spectrograms.
        Placeholder for future implementation.
        """
        raise NotImplementedError(
            "Learned embeddings require training a neural network. "
            "Use 'statistical', 'bag_of_frames', or 'openl3' methods instead."
        )
    
    def fit(self, songs: List[Song]):
        """
        Fit the vectorizer on a corpus (for bag-of-frames method).
        
        Args:
            songs: Training corpus
        """
        if self.method == "bag_of_frames":
            print(f"Training bag-of-frames codebook with {self.n_clusters} clusters...")
            
            # Extract all MFCCs
            all_frames = []
            for song in songs[:100]:  # Use subset for efficiency
                mfcc = song.extract_mfcc(n_mfcc=self.n_mfcc)
                all_frames.append(mfcc.T)
            
            all_frames = np.vstack(all_frames)
            
            # K-means clustering
            from sklearn.cluster import MiniBatchKMeans
            kmeans = MiniBatchKMeans(
                n_clusters=self.n_clusters,
                random_state=42,
                batch_size=1000
            )
            kmeans.fit(all_frames)
            
            self.codebook = kmeans.cluster_centers_
            print(f"✓ Codebook trained: {self.codebook.shape}")
        
        return self


def create_song_vector_index(
    songs: List[Song],
    method: str = "statistical",
    backend: str = "faiss",
    dimension: int = 512
):
    """
    Create a vector index from a collection of songs.
    
    Args:
        songs: List of songs to index
        method: Embedding method
        backend: Vector DB backend
        dimension: Vector dimension
        
    Returns:
        (VectorDBAdapter, SongVectorizer) tuple
    """
    from .vectordb import VectorDBAdapter
    
    # Initialize vectorizer
    vectorizer = SongVectorizer(method=method, dimension=dimension)
    
    # Fit if needed
    if method == "bag_of_frames":
        vectorizer.fit(songs)
    
    # Embed songs
    print(f"Embedding {len(songs)} songs...")
    embeddings = vectorizer.embed_batch(songs)
    print(f"✓ Embedded {len(embeddings)} songs")
    
    # Create vector DB
    db = VectorDBAdapter(backend=backend, dimension=dimension)
    db.insert(embeddings)
    print(f"✓ Indexed in {backend}")
    
    return db, vectorizer
