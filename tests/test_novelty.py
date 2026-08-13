"""
Unit tests for novelty analysis.
"""

import pytest
import numpy as np
from src.riff import Riff
from src.space import RiffSpace, RiffCollection
from src.novelty import NoveltyAnalyzer, NoveltyScore, quantify_derivative


@pytest.fixture
def temporal_riffs():
    """Create riffs with temporal metadata."""
    riffs = []
    for year in range(1960, 1971):
        riff = Riff(
            pitch_intervals=[0, 2, 2, -1],
            durations=[0.5, 0.5, 0.5, 0.5],
            metadata={"year": year, "genre": "Rock"}
        )
        riffs.append(riff)
    return riffs


class TestNoveltyScore:
    """Test NoveltyScore dataclass."""
    
    def test_create_score(self):
        riff = Riff([0, 2, 2], [1, 1, 1])
        score = NoveltyScore(riff=riff, novelty=5.0, year=1970)
        assert score.novelty == 5.0
        assert score.year == 1970


class TestNoveltyAnalyzer:
    """Test NoveltyAnalyzer class."""
    
    def test_create_analyzer(self):
        analyzer = NoveltyAnalyzer()
        assert len(analyzer.scores) == 0
    
    def test_compute_novelty_first_riff(self):
        analyzer = NoveltyAnalyzer()
        riff = Riff([0, 2, 2], [1, 1, 1], metadata={"year": 1960})
        
        score = analyzer.compute_novelty(riff, prior_riffs=[])
        assert np.isinf(score.novelty)
    
    def test_compute_novelty_with_priors(self, temporal_riffs):
        analyzer = NoveltyAnalyzer()
        
        query_riff = Riff(
            pitch_intervals=[0, 5, 7, -5],
            durations=[0.5, 0.5, 0.5, 0.5],
            metadata={"year": 1971}
        )
        
        score = analyzer.compute_novelty(query_riff, prior_riffs=temporal_riffs)
        assert score.novelty >= 0.0
        assert not np.isinf(score.novelty)
        assert score.nearest_prior is not None
    
    def test_analyze_collection(self, temporal_riffs):
        collection = RiffCollection()
        for riff in temporal_riffs:
            collection.add(riff)
        
        analyzer = NoveltyAnalyzer(space=collection.space)
        scores = analyzer.analyze_collection(collection)
        
        assert len(scores) == len(temporal_riffs)
        assert np.isinf(scores[0].novelty)  # First riff
        assert all(not np.isinf(s.novelty) for s in scores[1:])
    
    def test_get_temporal_profile(self, temporal_riffs):
        collection = RiffCollection()
        for riff in temporal_riffs:
            collection.add(riff)
        
        analyzer = NoveltyAnalyzer(space=collection.space)
        analyzer.analyze_collection(collection)
        
        years, novelties = analyzer.get_temporal_profile()
        assert len(years) > 0
        assert len(years) == len(novelties)
    
    def test_most_novel_riffs(self, temporal_riffs):
        # Add a very different riff
        novel_riff = Riff(
            pitch_intervals=[0, 12, -12, 7, -7],
            durations=[0.25, 0.25, 0.25, 0.25, 1.0],
            metadata={"year": 1970}
        )
        temporal_riffs.append(novel_riff)
        
        collection = RiffCollection()
        for riff in temporal_riffs:
            collection.add(riff)
        
        analyzer = NoveltyAnalyzer(space=collection.space)
        analyzer.analyze_collection(collection)
        
        most_novel = analyzer.most_novel_riffs(k=3)
        assert len(most_novel) <= 3
        assert most_novel[0].novelty >= most_novel[-1].novelty
    
    def test_least_novel_riffs(self, temporal_riffs):
        collection = RiffCollection()
        for riff in temporal_riffs:
            collection.add(riff)
        
        analyzer = NoveltyAnalyzer(space=collection.space)
        analyzer.analyze_collection(collection)
        
        least_novel = analyzer.least_novel_riffs(k=3)
        assert len(least_novel) <= 3
        assert least_novel[0].novelty <= least_novel[-1].novelty
    
    def test_detect_discontinuities(self, temporal_riffs):
        collection = RiffCollection()
        for riff in temporal_riffs:
            collection.add(riff)
        
        analyzer = NoveltyAnalyzer(space=collection.space)
        analyzer.analyze_collection(collection)
        
        discontinuities = analyzer.detect_discontinuities(threshold_std=1.0)
        assert isinstance(discontinuities, list)
    
    def test_compare_eras(self, temporal_riffs):
        collection = RiffCollection()
        for riff in temporal_riffs:
            collection.add(riff)
        
        analyzer = NoveltyAnalyzer(space=collection.space)
        analyzer.analyze_collection(collection)
        
        eras = {
            "Early": (1960, 1965),
            "Late": (1966, 1970)
        }
        
        era_novelties = analyzer.compare_eras(eras)
        assert len(era_novelties) == 2
        assert "Early" in era_novelties
        assert "Late" in era_novelties


class TestQuantifyDerivative:
    """Test derivative quantification."""
    
    def test_novel_riff(self):
        corpus = [
            Riff([0, 2, 2], [1, 1, 1]),
            Riff([0, 3, 3], [1, 1, 1])
        ]
        
        novel_riff = Riff([0, 12, -5, 7], [1, 1, 1, 1])
        
        score, similar = quantify_derivative(novel_riff, corpus)
        assert 0.0 <= score <= 1.0
    
    def test_derivative_riff(self):
        corpus = [
            Riff([0, 2, 2], [1, 1, 1]),
            Riff([0, 3, 3], [1, 1, 1])
        ]
        
        derivative_riff = Riff([0, 2, 2], [1, 1, 1])
        
        score, similar = quantify_derivative(derivative_riff, corpus, threshold=1.0)
        assert score > 0.5  # Should be quite derivative


if __name__ == "__main__":
    pytest.main([__file__])
