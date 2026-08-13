"""
Core riff representation as a sequence of (Δp, Δt, a) tuples.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class RiffNote:
    """A single note in a riff with interval-based representation."""
    
    pitch_interval: float  # Δp: semitones from previous note (0 for first note)
    duration: float  # Δt: note duration in beats
    onset: float  # absolute time position in beats
    articulation: str = "normal"  # palm-mute, accent, bend, slide, hammer-on, etc.
    velocity: float = 0.8  # MIDI-like velocity [0, 1]
    
    def __post_init__(self):
        """Validate note parameters."""
        if self.duration <= 0:
            raise ValueError(f"Duration must be positive, got {self.duration}")
        if not 0 <= self.velocity <= 1:
            raise ValueError(f"Velocity must be in [0, 1], got {self.velocity}")


class Riff:
    """
    A guitar riff represented as a sequence of interval-based notes.
    
    R = {(Δpᵢ, Δtᵢ, aᵢ)}ᵢ₌₁ⁿ
    
    This representation is independent of absolute pitch, making it 
    naturally invariant to transposition.
    """
    
    def __init__(
        self,
        pitch_intervals: List[float],
        durations: List[float],
        articulations: Optional[List[str]] = None,
        velocities: Optional[List[float]] = None,
        onsets: Optional[List[float]] = None,
        tempo: float = 120.0,
        metadata: Optional[dict] = None
    ):
        """
        Initialize a riff from interval-based representation.
        
        Args:
            pitch_intervals: Semitone intervals between consecutive notes
            durations: Note durations in beats
            articulations: Playing techniques per note
            velocities: Note velocities [0, 1]
            onsets: Absolute onset times in beats (auto-computed if None)
            tempo: BPM for tempo-invariant analysis
            metadata: Optional dict with 'artist', 'song', 'year', 'genre', etc.
        """
        n = len(pitch_intervals)
        
        if len(durations) != n:
            raise ValueError(f"Length mismatch: {n} intervals but {len(durations)} durations")
        
        # Default values
        if articulations is None:
            articulations = ["normal"] * n
        if velocities is None:
            velocities = [0.8] * n
        if onsets is None:
            onsets = self._compute_onsets(durations)
            
        if len(articulations) != n or len(velocities) != n or len(onsets) != n:
            raise ValueError("All note attributes must have same length")
        
        self.notes = [
            RiffNote(
                pitch_interval=pitch_intervals[i],
                duration=durations[i],
                onset=onsets[i],
                articulation=articulations[i],
                velocity=velocities[i]
            )
            for i in range(n)
        ]
        
        self.tempo = tempo
        self.metadata = metadata or {}
        
    @staticmethod
    def _compute_onsets(durations: List[float]) -> List[float]:
        """Compute cumulative onset times from durations."""
        onsets = [0.0]
        for dur in durations[:-1]:
            onsets.append(onsets[-1] + dur)
        return onsets
    
    @classmethod
    def from_absolute_pitches(
        cls,
        pitches: List[float],
        durations: List[float],
        **kwargs
    ) -> "Riff":
        """
        Construct a riff from absolute MIDI pitch numbers.
        
        Args:
            pitches: MIDI pitch numbers (e.g., 60 = middle C)
            durations: Note durations in beats
            **kwargs: Additional arguments for __init__
        """
        if not pitches:
            raise ValueError("Cannot create riff from empty pitch sequence")
        
        # Convert to intervals: first note is 0, rest are differences
        intervals = [0.0] + [pitches[i] - pitches[i-1] for i in range(1, len(pitches))]
        
        return cls(pitch_intervals=intervals, durations=durations, **kwargs)
    
    def to_absolute_pitches(self, root_pitch: float = 60.0) -> List[float]:
        """
        Convert interval representation to absolute MIDI pitches.
        
        Args:
            root_pitch: Starting pitch (default: middle C = 60)
            
        Returns:
            List of MIDI pitch numbers
        """
        pitches = [root_pitch]
        for note in self.notes[1:]:
            pitches.append(pitches[-1] + note.pitch_interval)
        return pitches
    
    @property
    def length(self) -> int:
        """Number of notes in the riff."""
        return len(self.notes)
    
    @property
    def total_duration(self) -> float:
        """Total duration of riff in beats."""
        if not self.notes:
            return 0.0
        return self.notes[-1].onset + self.notes[-1].duration
    
    def as_array(self) -> np.ndarray:
        """
        Convert riff to numpy array for distance computations.
        
        Returns:
            Array of shape (n, 4): [pitch_interval, duration, velocity, articulation_code]
        """
        # Map articulations to numeric codes
        articulation_map = {
            "normal": 0.0,
            "palm-mute": 1.0,
            "accent": 2.0,
            "bend": 3.0,
            "slide": 4.0,
            "hammer-on": 5.0,
            "pull-off": 6.0,
            "harmonic": 7.0,
        }
        
        return np.array([
            [
                note.pitch_interval,
                note.duration,
                note.velocity,
                articulation_map.get(note.articulation, 0.0)
            ]
            for note in self.notes
        ])
    
    def get_interval_sequence(self) -> np.ndarray:
        """Extract just the pitch interval sequence."""
        return np.array([note.pitch_interval for note in self.notes])
    
    def get_rhythm_sequence(self) -> np.ndarray:
        """Extract just the rhythm (duration) sequence."""
        return np.array([note.duration for note in self.notes])
    
    def get_articulation_sequence(self) -> List[str]:
        """Extract just the articulation sequence."""
        return [note.articulation for note in self.notes]
    
    def __len__(self) -> int:
        return self.length
    
    def __repr__(self) -> str:
        meta_str = ""
        if self.metadata:
            meta_str = f" ({', '.join(f'{k}={v}' for k, v in self.metadata.items())})"
        return f"Riff(length={self.length}, duration={self.total_duration:.2f}b, tempo={self.tempo}bpm{meta_str})"
    
    def __eq__(self, other: "Riff") -> bool:
        """Check exact equality between riffs."""
        if not isinstance(other, Riff):
            return False
        if len(self) != len(other):
            return False
        return all(
            np.allclose([n1.pitch_interval, n1.duration, n1.velocity],
                       [n2.pitch_interval, n2.duration, n2.velocity]) and
            n1.articulation == n2.articulation
            for n1, n2 in zip(self.notes, other.notes)
        )
    
    def copy(self) -> "Riff":
        """Create a deep copy of the riff."""
        return Riff(
            pitch_intervals=[n.pitch_interval for n in self.notes],
            durations=[n.duration for n in self.notes],
            articulations=[n.articulation for n in self.notes],
            velocities=[n.velocity for n in self.notes],
            onsets=[n.onset for n in self.notes],
            tempo=self.tempo,
            metadata=self.metadata.copy()
        )
    
    def normalize_rhythm(self) -> "Riff":
        """
        Normalize rhythm to unit total duration (tempo-invariant).
        
        Returns:
            New riff with durations scaled to sum to 1.0
        """
        total_dur = self.total_duration
        if total_dur == 0:
            return self.copy()
        
        new_riff = self.copy()
        scale = 1.0 / total_dur
        for note in new_riff.notes:
            note.duration *= scale
            note.onset *= scale
        
        return new_riff
    
    def to_dict(self) -> dict:
        """Serialize riff to dictionary."""
        return {
            "pitch_intervals": [n.pitch_interval for n in self.notes],
            "durations": [n.duration for n in self.notes],
            "articulations": [n.articulation for n in self.notes],
            "velocities": [n.velocity for n in self.notes],
            "onsets": [n.onset for n in self.notes],
            "tempo": self.tempo,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Riff":
        """Deserialize riff from dictionary."""
        return cls(**data)


def create_example_riff(name: str = "smoke_on_the_water") -> Riff:
    """
    Create example riffs for testing.
    
    Args:
        name: Name of the example riff
    """
    examples = {
        "smoke_on_the_water": {
            "pitches": [60, 63, 65, 60, 63, 66, 65],  # G-Bb-C pattern
            "durations": [0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 1.5],
            "articulations": ["palm-mute"] * 7,
            "metadata": {"song": "Smoke on the Water", "artist": "Deep Purple", "year": 1972}
        },
        "iron_man": {
            "pitches": [55, 58, 62, 58, 61, 60],  # Classic Sabbath
            "durations": [1.0, 1.0, 0.5, 0.5, 1.0, 2.0],
            "articulations": ["normal", "accent", "normal", "normal", "accent", "normal"],
            "metadata": {"song": "Iron Man", "artist": "Black Sabbath", "year": 1970}
        },
        "seven_nation_army": {
            "pitches": [64, 64, 67, 64, 62, 60, 58],
            "durations": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0],
            "articulations": ["accent"] + ["normal"] * 6,
            "metadata": {"song": "Seven Nation Army", "artist": "The White Stripes", "year": 2003}
        },
    }
    
    if name not in examples:
        raise ValueError(f"Unknown example: {name}. Available: {list(examples.keys())}")
    
    ex = examples[name]
    return Riff.from_absolute_pitches(**ex)
