"""
Unit tests for distance metrics.
"""

import pytest
import numpy as np
from src.riff import Riff
from src.metrics import (
    edit_distance,
    dtw_distance,
    euclidean_distance,
    cosine_similarity,
    interval_histogram_distance,
    get_metric,
    compare_metrics
)


@pytest.fixture
def riff1():
    return Riff(
        pitch_intervals=[0, 2, 2, -1, 3],
        durations=[0.5, 0.5, 0.5, 0.5, 1.0]
    )


@pytest.fixture
def riff2():
    return Riff(
        pitch_intervals=[0, 2, 2, -1],
        durations=[0.5, 0.5, 0.5, 1.0]
    )


@pytest.fixture
def identical_riff(riff1):
    return riff1.copy()


class TestEditDistance:
    """Test edit distance metric."""
    
    def test_identical_riffs(self, riff1, identical_riff):
        dist = edit_distance(riff1, identical_riff)
        assert dist == 0.0
    
    def test_different_riffs(self, riff1, riff2):
        dist = edit_distance(riff1, riff2)
        assert dist > 0.0
    
    def test_symmetric(self, riff1, riff2):
        dist1 = edit_distance(riff1, riff2)
        dist2 = edit_distance(riff2, riff1)
        assert np.isclose(dist1, dist2)


class TestDTWDistance:
    """Test Dynamic Time Warping distance."""
    
    def test_identical_riffs(self, riff1, identical_riff):
        dist = dtw_distance(riff1, identical_riff)
        assert np.isclose(dist, 0.0, atol=1e-6)
    
    def test_different_riffs(self, riff1, riff2):
        dist = dtw_distance(riff1, riff2)
        assert dist > 0.0
    
    def test_pitch_only(self, riff1, riff2):
        dist = dtw_distance(riff1, riff2, feature='pitch')
        assert dist >= 0.0
    
    def test_rhythm_only(self, riff1, riff2):
        dist = dtw_distance(riff1, riff2, feature='rhythm')
        assert dist >= 0.0


class TestEuclideanDistance:
    """Test Euclidean distance."""
    
    def test_identical_riffs(self, riff1, identical_riff):
        dist = euclidean_distance(riff1, identical_riff)
        assert np.isclose(dist, 0.0, atol=1e-6)
    
    def test_different_riffs(self, riff1, riff2):
        dist = euclidean_distance(riff1, riff2)
        assert dist > 0.0


class TestCosineDistance:
    """Test cosine similarity distance."""
    
    def test_identical_riffs(self, riff1, identical_riff):
        dist = cosine_similarity(riff1, identical_riff)
        assert np.isclose(dist, 0.0, atol=1e-6)
    
    def test_different_riffs(self, riff1, riff2):
        dist = cosine_similarity(riff1, riff2)
        assert 0.0 <= dist <= 1.0


class TestIntervalHistogram:
    """Test interval histogram distance."""
    
    def test_identical_riffs(self, riff1, identical_riff):
        dist = interval_histogram_distance(riff1, identical_riff)
        assert np.isclose(dist, 0.0, atol=1e-6)
    
    def test_different_riffs(self, riff1, riff2):
        dist = interval_histogram_distance(riff1, riff2)
        assert dist >= 0.0


class TestMetricRegistry:
    """Test metric registry functions."""
    
    def test_get_metric(self):
        metric_func = get_metric("edit_distance")
        assert callable(metric_func)
    
    def test_get_invalid_metric(self):
        with pytest.raises(ValueError):
            get_metric("nonexistent_metric")
    
    def test_compare_metrics(self, riff1, riff2):
        results = compare_metrics(riff1, riff2)
        assert isinstance(results, dict)
        assert len(results) > 0
        assert "edit_distance" in results


if __name__ == "__main__":
    pytest.main([__file__])
