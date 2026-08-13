"""
Novelty analysis for temporal evolution of rock riffs.

Core metric:
    N(Rₜ) = min_{Rᵢ: tᵢ < t} d(Rₜ, Rᵢ)

A riff's novelty is its distance from ALL riffs released before it.
"""

from typing import List, Optional, Tuple, Dict
import numpy as np
from dataclasses import dataclass
from .riff import Riff
from .space import RiffSpace, RiffCollection


@dataclass
class NoveltyScore:
    """Novelty score for a single riff."""
    
    riff: Riff
    novelty: float  # Distance to nearest prior riff
    nearest_prior: Optional[Riff] = None  # The nearest prior riff
    distance_to_prior: float = 0.0
    year: Optional[int] = None
    
    def __repr__(self) -> str:
        year_str = f"year={self.year}, " if self.year else ""
        return f"NoveltyScore({year_str}novelty={self.novelty:.3f})"


class NoveltyAnalyzer:
    """
    Analyze temporal novelty of riffs.
    
    Measures how innovative a riff is relative to the existing
    musical landscape at the time of its release.
    """
    
    def __init__(self, space: Optional[RiffSpace] = None):
        """
        Initialize novelty analyzer.
        
        Args:
            space: RiffSpace for distance computations
        """
        self.space = space or RiffSpace()
        self.scores: List[NoveltyScore] = []
    
    def compute_novelty(
        self,
        query_riff: Riff,
        prior_riffs: List[Riff],
        return_nearest: bool = True
    ) -> NoveltyScore:
        """
        Compute novelty of a riff relative to prior riffs.
        
        N(Rₜ) = min_{Rᵢ: tᵢ < t} d(Rₜ, Rᵢ)
        
        Args:
            query_riff: Riff to evaluate
            prior_riffs: All riffs released before query_riff
            return_nearest: Whether to identify nearest prior riff
            
        Returns:
            NoveltyScore object
        """
        if not prior_riffs:
            # First riff ever - maximally novel
            return NoveltyScore(
                riff=query_riff,
                novelty=np.inf,
                year=query_riff.metadata.get("year")
            )
        
        # Find minimum distance to any prior riff
        distances = [self.space.distance(query_riff, prior) for prior in prior_riffs]
        min_distance = min(distances)
        
        nearest_prior = None
        if return_nearest:
            nearest_idx = np.argmin(distances)
            nearest_prior = prior_riffs[nearest_idx]
        
        return NoveltyScore(
            riff=query_riff,
            novelty=min_distance,
            nearest_prior=nearest_prior,
            distance_to_prior=min_distance,
            year=query_riff.metadata.get("year")
        )
    
    def analyze_collection(
        self,
        collection: RiffCollection,
        sort_by_time: bool = True
    ) -> List[NoveltyScore]:
        """
        Compute novelty for all riffs in a temporal collection.
        
        Args:
            collection: RiffCollection with temporal metadata
            sort_by_time: Whether to sort by year before analysis
            
        Returns:
            List of NoveltyScore objects
        """
        riffs = collection.riffs
        
        if sort_by_time:
            # Sort by year, handling missing metadata
            riffs = sorted(
                riffs,
                key=lambda r: r.metadata.get("year", 9999)
            )
        
        self.scores = []
        
        for i, riff in enumerate(riffs):
            prior_riffs = riffs[:i]  # All riffs before this one
            score = self.compute_novelty(riff, prior_riffs)
            self.scores.append(score)
        
        return self.scores
    
    def get_temporal_profile(
        self,
        scores: Optional[List[NoveltyScore]] = None,
        aggregate: str = "mean"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get novelty over time profile.
        
        Args:
            scores: NoveltyScores to analyze (default: self.scores)
            aggregate: How to aggregate per year ('mean', 'max', 'median')
            
        Returns:
            (years, novelty_values) arrays
        """
        if scores is None:
            scores = self.scores
        
        # Group by year
        year_novelties: Dict[int, List[float]] = {}
        for score in scores:
            if score.year is None:
                continue
            if np.isinf(score.novelty):
                continue  # Skip infinite novelty (first riff)
            
            if score.year not in year_novelties:
                year_novelties[score.year] = []
            year_novelties[score.year].append(score.novelty)
        
        # Sort years
        years = sorted(year_novelties.keys())
        
        # Aggregate
        if aggregate == "mean":
            novelties = [np.mean(year_novelties[y]) for y in years]
        elif aggregate == "max":
            novelties = [np.max(year_novelties[y]) for y in years]
        elif aggregate == "median":
            novelties = [np.median(year_novelties[y]) for y in years]
        else:
            raise ValueError(f"Unknown aggregation: {aggregate}")
        
        return np.array(years), np.array(novelties)
    
    def find_peak_innovation_periods(
        self,
        window_size: int = 3,
        threshold_percentile: float = 75
    ) -> List[Tuple[int, float]]:
        """
        Identify periods of peak musical innovation.
        
        Args:
            window_size: Years to average over
            threshold_percentile: Percentile above which to flag peaks
            
        Returns:
            List of (year, novelty) for peak periods
        """
        years, novelties = self.get_temporal_profile()
        
        if len(years) < window_size:
            return list(zip(years, novelties))
        
        # Smooth with moving average
        smoothed = np.convolve(
            novelties,
            np.ones(window_size) / window_size,
            mode='valid'
        )
        smoothed_years = years[window_size // 2: -(window_size // 2) + (1 if window_size % 2 == 0 else 0)]
        
        # Find peaks above threshold
        threshold = np.percentile(smoothed, threshold_percentile)
        peaks = [(int(y), float(n)) for y, n in zip(smoothed_years, smoothed) if n >= threshold]
        
        return peaks
    
    def most_novel_riffs(self, k: int = 10) -> List[NoveltyScore]:
        """
        Find the k most novel riffs.
        
        Args:
            k: Number of riffs to return
            
        Returns:
            Top k NoveltyScores sorted by novelty (descending)
        """
        valid_scores = [s for s in self.scores if not np.isinf(s.novelty)]
        valid_scores.sort(key=lambda s: s.novelty, reverse=True)
        return valid_scores[:k]
    
    def least_novel_riffs(self, k: int = 10) -> List[NoveltyScore]:
        """
        Find the k least novel (most derivative) riffs.
        
        Args:
            k: Number of riffs to return
            
        Returns:
            Bottom k NoveltyScores sorted by novelty (ascending)
        """
        valid_scores = [s for s in self.scores if not np.isinf(s.novelty)]
        valid_scores.sort(key=lambda s: s.novelty)
        return valid_scores[:k]
    
    def detect_discontinuities(
        self,
        threshold_std: float = 2.0
    ) -> List[Tuple[int, float]]:
        """
        Detect years with unusually high novelty (genre transitions?).
        
        Args:
            threshold_std: Standard deviations above mean to flag
            
        Returns:
            List of (year, z_score) for discontinuity points
        """
        years, novelties = self.get_temporal_profile()
        
        mean_novelty = np.mean(novelties)
        std_novelty = np.std(novelties)
        
        z_scores = (novelties - mean_novelty) / std_novelty
        
        discontinuities = [
            (int(year), float(z))
            for year, z in zip(years, z_scores)
            if z > threshold_std
        ]
        
        discontinuities.sort(key=lambda x: x[1], reverse=True)
        return discontinuities
    
    def compare_eras(
        self,
        era_ranges: Dict[str, Tuple[int, int]]
    ) -> Dict[str, float]:
        """
        Compare average novelty across different eras.
        
        Args:
            era_ranges: Dict mapping era names to (start_year, end_year)
            
        Returns:
            Dict mapping era names to average novelty
        """
        era_novelties = {}
        
        for era_name, (start, end) in era_ranges.items():
            era_scores = [
                s.novelty for s in self.scores
                if s.year is not None and start <= s.year <= end and not np.isinf(s.novelty)
            ]
            
            if era_scores:
                era_novelties[era_name] = float(np.mean(era_scores))
            else:
                era_novelties[era_name] = 0.0
        
        return era_novelties
    
    def influence_network(
        self,
        threshold: float = 0.5
    ) -> List[Tuple[Riff, Riff, float]]:
        """
        Build influence network: edges from each riff to its nearest prior.
        
        Args:
            threshold: Only include edges with distance < threshold
            
        Returns:
            List of (source_riff, target_riff, distance) tuples
        """
        edges = []
        
        for score in self.scores:
            if score.nearest_prior is not None:
                if score.distance_to_prior < threshold:
                    edges.append((
                        score.nearest_prior,
                        score.riff,
                        score.distance_to_prior
                    ))
        
        return edges
    
    def __repr__(self) -> str:
        if not self.scores:
            return "NoveltyAnalyzer(no scores computed)"
        
        years = [s.year for s in self.scores if s.year]
        year_range = f"{min(years)}-{max(years)}" if years else "unknown"
        return f"NoveltyAnalyzer(n_scores={len(self.scores)}, years={year_range})"


def quantify_derivative(
    riff: Riff,
    corpus: List[Riff],
    space: Optional[RiffSpace] = None,
    threshold: float = 1.0
) -> Tuple[float, List[Riff]]:
    """
    Quantify how derivative a riff is.
    
    Args:
        riff: Riff to evaluate
        corpus: Corpus of existing riffs
        space: RiffSpace for distance computation
        threshold: Distance threshold for "similar" riffs
        
    Returns:
        (derivative_score, similar_riffs)
        derivative_score: 0 = totally novel, 1 = very derivative
    """
    if space is None:
        space = RiffSpace()
    
    if not corpus:
        return 0.0, []
    
    distances = [space.distance(riff, other) for other in corpus]
    min_distance = min(distances)
    
    # Derivative score: inverse of normalized minimum distance
    # 0 distance -> score 1 (very derivative)
    # Large distance -> score 0 (very novel)
    derivative_score = 1.0 - min(min_distance / threshold, 1.0)
    
    # Find similar riffs
    similar_riffs = [
        corpus[i] for i, d in enumerate(distances)
        if d < threshold
    ]
    
    return derivative_score, similar_riffs
