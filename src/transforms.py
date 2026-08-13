"""
Transformation group G for riff equivalence classes.

Transformations include:
- Transposition (already handled by interval representation)
- Tempo scaling
- Octave displacement
- Time stretching/compression
"""

from typing import Callable, List
from dataclasses import dataclass
import numpy as np
from .riff import Riff


@dataclass
class Transformation:
    """A single transformation in the group G."""
    
    name: str
    transform_func: Callable[[Riff], Riff]
    
    def __call__(self, riff: Riff) -> Riff:
        """Apply transformation to a riff."""
        return self.transform_func(riff)
    
    def __repr__(self) -> str:
        return f"Transformation({self.name})"


class TransformGroup:
    """
    Group of transformations for riff equivalence.
    
    R₁ ~ R₂ ⟺ R₁ = T(R₂), T ∈ G
    
    This enables us to compare riffs modulo musically-irrelevant variations.
    """
    
    def __init__(self):
        """Initialize the transformation group with standard operations."""
        self.transformations: List[Transformation] = []
        self._register_default_transforms()
    
    def _register_default_transforms(self):
        """Register standard musical transformations."""
        
        # Transposition is already handled by interval representation
        # (intervals are transposition-invariant)
        
        # Identity transform
        self.register(
            "identity",
            lambda r: r.copy()
        )
        
        # Tempo scaling (normalize rhythm)
        self.register(
            "tempo_normalize",
            lambda r: r.normalize_rhythm()
        )
        
        # Octave displacement (shift all intervals by ±12 semitones)
        def octave_up(riff: Riff) -> Riff:
            new_riff = riff.copy()
            new_riff.notes[0].pitch_interval += 12  # shift starting pitch
            return new_riff
        
        def octave_down(riff: Riff) -> Riff:
            new_riff = riff.copy()
            new_riff.notes[0].pitch_interval -= 12
            return new_riff
        
        self.register("octave_up", octave_up)
        self.register("octave_down", octave_down)
        
        # Retrograde (reverse the riff)
        def retrograde(riff: Riff) -> Riff:
            new_intervals = [riff.notes[0].pitch_interval]  # keep first note
            # Reverse and negate intervals
            for i in range(len(riff.notes) - 1, 0, -1):
                new_intervals.append(-riff.notes[i].pitch_interval)
            
            return Riff(
                pitch_intervals=new_intervals,
                durations=[n.duration for n in reversed(riff.notes)],
                articulations=[n.articulation for n in reversed(riff.notes)],
                velocities=[n.velocity for n in reversed(riff.notes)],
                tempo=riff.tempo,
                metadata={**riff.metadata, "transform": "retrograde"}
            )
        
        self.register("retrograde", retrograde)
        
        # Time stretch (double/halve note durations)
        def time_stretch(factor: float):
            def stretch(riff: Riff) -> Riff:
                return Riff(
                    pitch_intervals=[n.pitch_interval for n in riff.notes],
                    durations=[n.duration * factor for n in riff.notes],
                    articulations=[n.articulation for n in riff.notes],
                    velocities=[n.velocity for n in riff.notes],
                    tempo=riff.tempo / factor,
                    metadata={**riff.metadata, "transform": f"stretch_{factor}"}
                )
            return stretch
        
        self.register("double_time", time_stretch(0.5))
        self.register("half_time", time_stretch(2.0))
    
    def register(self, name: str, func: Callable[[Riff], Riff]):
        """Register a new transformation."""
        self.transformations.append(Transformation(name, func))
    
    def apply(self, riff: Riff, transform_name: str) -> Riff:
        """Apply a specific transformation by name."""
        for t in self.transformations:
            if t.name == transform_name:
                return t(riff)
        raise ValueError(f"Unknown transformation: {transform_name}")
    
    def apply_all(self, riff: Riff) -> List[Riff]:
        """
        Generate all transformations of a riff.
        
        Returns:
            List of transformed riffs (equivalence class representatives)
        """
        return [t(riff) for t in self.transformations]
    
    def normalize(self, riff: Riff) -> Riff:
        """
        Apply standard normalization transforms for canonical form.
        
        This creates a canonical representative of the equivalence class
        by applying tempo normalization and other standard transforms.
        """
        normalized = riff.copy()
        
        # Always normalize rhythm for tempo-invariance
        normalized = normalized.normalize_rhythm()
        
        return normalized
    
    def generate_transpositions(self, riff: Riff, semitones: List[int]) -> List[Riff]:
        """
        Generate transpositions by shifting the root note.
        
        Args:
            riff: Input riff
            semitones: List of transposition amounts
            
        Returns:
            List of transposed riffs
        """
        transposed = []
        for shift in semitones:
            new_riff = riff.copy()
            # Shift only affects absolute pitch reconstruction, not intervals
            new_riff.notes[0].pitch_interval += shift
            new_riff.metadata = {
                **new_riff.metadata, 
                "transposition": shift
            }
            transposed.append(new_riff)
        
        return transposed
    
    def __len__(self) -> int:
        return len(self.transformations)
    
    def __repr__(self) -> str:
        names = [t.name for t in self.transformations]
        return f"TransformGroup({', '.join(names)})"


def get_standard_group() -> TransformGroup:
    """
    Get the standard transformation group for riff analysis.
    
    Returns:
        TransformGroup with all standard transforms
    """
    return TransformGroup()


def create_custom_group(transform_names: List[str]) -> TransformGroup:
    """
    Create a custom transformation group with selected transforms.
    
    Args:
        transform_names: List of transformation names to include
        
    Returns:
        TransformGroup with only specified transforms
    """
    full_group = TransformGroup()
    custom_group = TransformGroup()
    custom_group.transformations = []  # Clear defaults
    
    for name in transform_names:
        for t in full_group.transformations:
            if t.name == name:
                custom_group.transformations.append(t)
                break
        else:
            raise ValueError(f"Unknown transformation: {name}")
    
    return custom_group
