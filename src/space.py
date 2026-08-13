"""
Quotient space of riffs under transformation group G.

d([R₁], [R₂]) = inf_{T∈G} D(R₁, T(R₂))
"""

from typing import List, Optional, Callable
import numpy as np
from .riff import Riff
from .transforms import TransformGroup, get_standard_group
from .metrics import get_metric


class RiffSpace:
    """
    The quotient space of riffs under transformation equivalence.
    
    This space treats riffs as equivalent if they differ only by
    transposition, tempo, octave, etc.
    """
    
    def __init__(
        self,
        metric: str = "edit_distance",
        transform_group: Optional[TransformGroup] = None,
        normalize: bool = True
    ):
        """
        Initialize riff space with metric and transformation group.
        
        Args:
            metric: Distance metric name (see metrics.py)
            transform_group: Group of transformations (default: standard group)
            normalize: Whether to normalize riffs before comparison
        """
        self.metric_name = metric
        self.metric_func = get_metric(metric)
        self.transform_group = transform_group or get_standard_group()
        self.normalize = normalize
        
        self.riffs: List[Riff] = []  # Collection of riffs in space
        self._distance_cache = {}  # Cache for computed distances
    
    def add_riff(self, riff: Riff):
        """Add a riff to the space."""
        if self.normalize:
            riff = self.transform_group.normalize(riff)
        self.riffs.append(riff)
    
    def add_riffs(self, riffs: List[Riff]):
        """Add multiple riffs to the space."""
        for riff in riffs:
            self.add_riff(riff)
    
    def distance(self, riff1: Riff, riff2: Riff, use_transforms: bool = True) -> float:
        """
        Compute distance between two riffs.
        
        If use_transforms=True, computes:
            d([R₁], [R₂]) = inf_{T∈G} D(R₁, T(R₂))
        
        Args:
            riff1, riff2: Riffs to compare
            use_transforms: Whether to minimize over transformation group
            
        Returns:
            Distance between riffs (or their equivalence classes)
        """
        # Check cache
        cache_key = (id(riff1), id(riff2), use_transforms)
        if cache_key in self._distance_cache:
            return self._distance_cache[cache_key]
        
        # Normalize if requested
        if self.normalize:
            riff1 = self.transform_group.normalize(riff1)
            riff2 = self.transform_group.normalize(riff2)
        
        if not use_transforms:
            # Direct distance without transformation
            dist = self.metric_func(riff1, riff2)
        else:
            # Minimize over transformations
            transformed_riffs = self.transform_group.apply_all(riff2)
            distances = [self.metric_func(riff1, t_riff) for t_riff in transformed_riffs]
            dist = min(distances)
        
        # Cache result
        self._distance_cache[cache_key] = dist
        return dist
    
    def distance_matrix(self, riffs: Optional[List[Riff]] = None) -> np.ndarray:
        """
        Compute pairwise distance matrix for a collection of riffs.
        
        Args:
            riffs: List of riffs (default: use self.riffs)
            
        Returns:
            Symmetric distance matrix of shape (n, n)
        """
        if riffs is None:
            riffs = self.riffs
        
        n = len(riffs)
        dist_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = self.distance(riffs[i], riffs[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        
        return dist_matrix
    
    def nearest_neighbors(
        self,
        query_riff: Riff,
        k: int = 5,
        riffs: Optional[List[Riff]] = None
    ) -> List[tuple]:
        """
        Find k nearest neighbors to a query riff.
        
        Args:
            query_riff: Riff to find neighbors for
            k: Number of neighbors
            riffs: Search within these riffs (default: self.riffs)
            
        Returns:
            List of (riff, distance) tuples, sorted by distance
        """
        if riffs is None:
            riffs = self.riffs
        
        distances = [(riff, self.distance(query_riff, riff)) for riff in riffs]
        distances.sort(key=lambda x: x[1])
        
        return distances[:k]
    
    def radius_search(
        self,
        center_riff: Riff,
        radius: float,
        riffs: Optional[List[Riff]] = None
    ) -> List[Riff]:
        """
        Find all riffs within a given radius of a center riff.
        
        Args:
            center_riff: Center of search
            radius: Maximum distance
            riffs: Search within these riffs (default: self.riffs)
            
        Returns:
            List of riffs within radius
        """
        if riffs is None:
            riffs = self.riffs
        
        neighbors = []
        for riff in riffs:
            if self.distance(center_riff, riff) <= radius:
                neighbors.append(riff)
        
        return neighbors
    
    def compute_centrality(self, riff: Riff) -> float:
        """
        Compute how central a riff is in the space.
        
        Centrality = average distance to all other riffs (lower = more central)
        
        Args:
            riff: Riff to compute centrality for
            
        Returns:
            Average distance to all riffs in space
        """
        if not self.riffs:
            return 0.0
        
        distances = [self.distance(riff, other) for other in self.riffs if other != riff]
        return np.mean(distances) if distances else 0.0
    
    def local_density(self, riff: Riff, radius: float) -> int:
        """
        Count number of riffs within radius (local density).
        
        Args:
            riff: Center riff
            radius: Search radius
            
        Returns:
            Number of neighbors within radius
        """
        return len(self.radius_search(riff, radius)) - 1  # Exclude self
    
    def clear_cache(self):
        """Clear the distance computation cache."""
        self._distance_cache = {}
    
    def __len__(self) -> int:
        return len(self.riffs)
    
    def __repr__(self) -> str:
        return f"RiffSpace(metric={self.metric_name}, n_riffs={len(self)}, transforms={len(self.transform_group)})"


class RiffCollection:
    """
    A collection of riffs with metadata for temporal/genre analysis.
    """
    
    def __init__(self, space: Optional[RiffSpace] = None):
        """
        Initialize a riff collection.
        
        Args:
            space: RiffSpace to use for distance computations
        """
        self.space = space or RiffSpace()
        self.riffs: List[Riff] = []
    
    def add(self, riff: Riff):
        """Add a riff to the collection."""
        self.riffs.append(riff)
        self.space.add_riff(riff)
    
    def get_by_year(self, year: int) -> List[Riff]:
        """Get all riffs from a specific year."""
        return [r for r in self.riffs if r.metadata.get("year") == year]
    
    def get_by_genre(self, genre: str) -> List[Riff]:
        """Get all riffs from a specific genre."""
        return [r for r in self.riffs if r.metadata.get("genre") == genre]
    
    def get_by_artist(self, artist: str) -> List[Riff]:
        """Get all riffs by a specific artist."""
        return [r for r in self.riffs if r.metadata.get("artist") == artist]
    
    def get_time_range(self, start_year: int, end_year: int) -> List[Riff]:
        """Get riffs within a time range."""
        return [
            r for r in self.riffs
            if start_year <= r.metadata.get("year", 0) <= end_year
        ]
    
    def temporal_sort(self) -> List[Riff]:
        """Sort riffs chronologically by year."""
        return sorted(self.riffs, key=lambda r: r.metadata.get("year", 9999))
    
    def __len__(self) -> int:
        return len(self.riffs)
    
    def __repr__(self) -> str:
        years = [r.metadata.get("year") for r in self.riffs if "year" in r.metadata]
        year_range = f"{min(years)}-{max(years)}" if years else "unknown"
        return f"RiffCollection(n={len(self)}, years={year_range})"
