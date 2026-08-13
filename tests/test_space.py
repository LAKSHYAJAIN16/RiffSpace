"""
Unit tests for riff space.
"""

import pytest
import numpy as np
from src.riff import Riff, create_example_riff
from src.space import RiffSpace, RiffCollection


@pytest.fixture
def test_riffs():
    return [
        create_example_riff("smoke_on_the_water"),
        create_example_riff("iron_man"),
        create_example_riff("seven_nation_army")
    ]


class TestRiffSpace:
    """Test RiffSpace class."""
    
    def test_create_space(self):
        space = RiffSpace(metric="edit_distance")
        assert space.metric_name == "edit_distance"
        assert len(space) == 0
    
    def test_add_riff(self, test_riffs):
        space = RiffSpace()
        space.add_riff(test_riffs[0])
        assert len(space) == 1
    
    def test_add_riffs(self, test_riffs):
        space = RiffSpace()
        space.add_riffs(test_riffs)
        assert len(space) == 3
    
    def test_distance(self, test_riffs):
        space = RiffSpace(metric="edit_distance")
        dist = space.distance(test_riffs[0], test_riffs[1])
        assert dist >= 0.0
    
    def test_distance_symmetric(self, test_riffs):
        space = RiffSpace()
        dist1 = space.distance(test_riffs[0], test_riffs[1])
        dist2 = space.distance(test_riffs[1], test_riffs[0])
        assert np.isclose(dist1, dist2)
    
    def test_distance_matrix(self, test_riffs):
        space = RiffSpace()
        space.add_riffs(test_riffs)
        dist_matrix = space.distance_matrix()
        
        assert dist_matrix.shape == (3, 3)
        assert np.allclose(dist_matrix, dist_matrix.T)  # Symmetric
        assert np.allclose(np.diag(dist_matrix), 0.0)  # Zero diagonal
    
    def test_nearest_neighbors(self, test_riffs):
        space = RiffSpace()
        space.add_riffs(test_riffs)
        
        neighbors = space.nearest_neighbors(test_riffs[0], k=2)
        assert len(neighbors) == 2
        assert all(isinstance(n, tuple) and len(n) == 2 for n in neighbors)
    
    def test_radius_search(self, test_riffs):
        space = RiffSpace()
        space.add_riffs(test_riffs)
        
        neighbors = space.radius_search(test_riffs[0], radius=10.0)
        assert isinstance(neighbors, list)
        assert test_riffs[0] in neighbors
    
    def test_compute_centrality(self, test_riffs):
        space = RiffSpace()
        space.add_riffs(test_riffs)
        
        centrality = space.compute_centrality(test_riffs[0])
        assert centrality >= 0.0
    
    def test_local_density(self, test_riffs):
        space = RiffSpace()
        space.add_riffs(test_riffs)
        
        density = space.local_density(test_riffs[0], radius=10.0)
        assert density >= 0


class TestRiffCollection:
    """Test RiffCollection class."""
    
    def test_create_collection(self):
        collection = RiffCollection()
        assert len(collection) == 0
    
    def test_add_riff(self, test_riffs):
        collection = RiffCollection()
        collection.add(test_riffs[0])
        assert len(collection) == 1
    
    def test_get_by_year(self):
        collection = RiffCollection()
        
        riff1970 = Riff(
            pitch_intervals=[0, 2, 2],
            durations=[1, 1, 1],
            metadata={"year": 1970}
        )
        riff1972 = Riff(
            pitch_intervals=[0, 3, 3],
            durations=[1, 1, 1],
            metadata={"year": 1972}
        )
        
        collection.add(riff1970)
        collection.add(riff1972)
        
        riffs_1970 = collection.get_by_year(1970)
        assert len(riffs_1970) == 1
        assert riffs_1970[0].metadata["year"] == 1970
    
    def test_get_by_genre(self):
        collection = RiffCollection()
        
        riff_metal = Riff(
            pitch_intervals=[0, 2, 2],
            durations=[1, 1, 1],
            metadata={"genre": "Metal"}
        )
        riff_punk = Riff(
            pitch_intervals=[0, 3, 3],
            durations=[1, 1, 1],
            metadata={"genre": "Punk"}
        )
        
        collection.add(riff_metal)
        collection.add(riff_punk)
        
        metal_riffs = collection.get_by_genre("Metal")
        assert len(metal_riffs) == 1
        assert metal_riffs[0].metadata["genre"] == "Metal"
    
    def test_temporal_sort(self):
        collection = RiffCollection()
        
        for year in [1980, 1970, 1990]:
            riff = Riff(
                pitch_intervals=[0, 2],
                durations=[1, 1],
                metadata={"year": year}
            )
            collection.add(riff)
        
        sorted_riffs = collection.temporal_sort()
        years = [r.metadata["year"] for r in sorted_riffs]
        assert years == [1970, 1980, 1990]


if __name__ == "__main__":
    pytest.main([__file__])
