"""
Unit tests for riff representation.
"""

import pytest
import numpy as np
from src.riff import Riff, RiffNote, create_example_riff


class TestRiffNote:
    """Test RiffNote class."""
    
    def test_create_note(self):
        note = RiffNote(
            pitch_interval=2.0,
            duration=0.5,
            onset=0.0,
            articulation="palm-mute",
            velocity=0.7
        )
        assert note.pitch_interval == 2.0
        assert note.duration == 0.5
        assert note.articulation == "palm-mute"
    
    def test_invalid_duration(self):
        with pytest.raises(ValueError):
            RiffNote(pitch_interval=0, duration=-1.0, onset=0.0)
    
    def test_invalid_velocity(self):
        with pytest.raises(ValueError):
            RiffNote(pitch_interval=0, duration=1.0, onset=0.0, velocity=1.5)


class TestRiff:
    """Test Riff class."""
    
    def test_create_riff(self):
        riff = Riff(
            pitch_intervals=[0, 2, 2, -1],
            durations=[0.5, 0.5, 0.5, 0.5]
        )
        assert len(riff) == 4
        assert riff.total_duration == 1.5
    
    def test_from_absolute_pitches(self):
        riff = Riff.from_absolute_pitches(
            pitches=[60, 62, 64, 63],
            durations=[0.5, 0.5, 0.5, 0.5]
        )
        expected_intervals = [0, 2, 2, -1]
        assert np.allclose(riff.get_interval_sequence(), expected_intervals)
    
    def test_to_absolute_pitches(self):
        riff = Riff(
            pitch_intervals=[0, 2, 2, -1],
            durations=[0.5, 0.5, 0.5, 0.5]
        )
        pitches = riff.to_absolute_pitches(root_pitch=60)
        assert pitches == [60, 62, 64, 63]
    
    def test_normalize_rhythm(self):
        riff = Riff(
            pitch_intervals=[0, 2, 2],
            durations=[1.0, 1.0, 2.0]
        )
        normalized = riff.normalize_rhythm()
        assert np.isclose(normalized.total_duration, 1.0)
    
    def test_as_array(self):
        riff = Riff(
            pitch_intervals=[0, 2, -1],
            durations=[0.5, 0.5, 0.5]
        )
        arr = riff.as_array()
        assert arr.shape == (3, 4)  # 3 notes, 4 features
    
    def test_copy(self):
        riff = Riff(
            pitch_intervals=[0, 2, 2],
            durations=[0.5, 0.5, 0.5]
        )
        copy = riff.copy()
        assert riff == copy
        assert riff is not copy
    
    def test_to_dict_from_dict(self):
        riff = Riff(
            pitch_intervals=[0, 2, 2],
            durations=[0.5, 0.5, 0.5],
            metadata={"artist": "Test"}
        )
        data = riff.to_dict()
        reconstructed = Riff.from_dict(data)
        assert riff == reconstructed
    
    def test_example_riffs(self):
        smoke = create_example_riff("smoke_on_the_water")
        assert len(smoke) > 0
        assert "Deep Purple" in smoke.metadata.get("artist", "")
        
        iron_man = create_example_riff("iron_man")
        assert len(iron_man) > 0


class TestRiffOperations:
    """Test operations on riffs."""
    
    def test_equality(self):
        riff1 = Riff(
            pitch_intervals=[0, 2, 2],
            durations=[0.5, 0.5, 0.5]
        )
        riff2 = Riff(
            pitch_intervals=[0, 2, 2],
            durations=[0.5, 0.5, 0.5]
        )
        assert riff1 == riff2
    
    def test_inequality(self):
        riff1 = Riff(
            pitch_intervals=[0, 2, 2],
            durations=[0.5, 0.5, 0.5]
        )
        riff2 = Riff(
            pitch_intervals=[0, 3, 2],
            durations=[0.5, 0.5, 0.5]
        )
        assert riff1 != riff2


if __name__ == "__main__":
    pytest.main([__file__])
