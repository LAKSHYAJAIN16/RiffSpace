"""
Data pipeline for processing guitar riffs from MIDI and audio files.
"""

from typing import List, Optional, Tuple
import numpy as np
from pathlib import Path
from .riff import Riff


def extract_riff_from_midi(
    midi_path: str,
    track_idx: int = 0,
    start_bar: Optional[int] = None,
    end_bar: Optional[int] = None,
    quantize: bool = True
) -> Riff:
    """
    Extract a riff from a MIDI file.
    
    Args:
        midi_path: Path to MIDI file
        track_idx: Which track to extract (0 = first track)
        start_bar: Starting bar (None = from beginning)
        end_bar: Ending bar (None = to end)
        quantize: Whether to quantize note timings
        
    Returns:
        Riff object
    """
    try:
        import pretty_midi
    except ImportError:
        raise ImportError("pretty_midi required. Run: pip install pretty_midi")
    
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    
    if not midi_data.instruments:
        raise ValueError(f"No instruments found in {midi_path}")
    
    instrument = midi_data.instruments[track_idx]
    notes = instrument.notes
    
    if not notes:
        raise ValueError(f"No notes found in track {track_idx}")
    
    # Filter by time range if specified
    if start_bar is not None or end_bar is not None:
        # Assuming 4/4 time, 4 beats per bar
        beats_per_bar = 4
        tempo = midi_data.get_tempo_changes()[1][0] if midi_data.get_tempo_changes()[1].size > 0 else 120.0
        
        if start_bar is not None:
            start_time = (start_bar * beats_per_bar) * (60.0 / tempo)
            notes = [n for n in notes if n.start >= start_time]
        
        if end_bar is not None:
            end_time = (end_bar * beats_per_bar) * (60.0 / tempo)
            notes = [n for n in notes if n.end <= end_time]
    
    # Sort by start time
    notes.sort(key=lambda n: n.start)
    
    if not notes:
        raise ValueError("No notes in specified range")
    
    # Convert to riff representation
    pitches = [n.pitch for n in notes]
    
    # Compute durations in beats
    tempo = midi_data.get_tempo_changes()[1][0] if midi_data.get_tempo_changes()[1].size > 0 else 120.0
    durations = [(n.end - n.start) * tempo / 60.0 for n in notes]
    
    if quantize:
        # Quantize to nearest 16th note
        durations = [round(d * 4) / 4 for d in durations]
    
    # Extract velocities
    velocities = [n.velocity / 127.0 for n in notes]  # Normalize to [0, 1]
    
    # Create riff
    riff = Riff.from_absolute_pitches(
        pitches=pitches,
        durations=durations,
        velocities=velocities,
        tempo=tempo,
        metadata={"source": midi_path}
    )
    
    return riff


def extract_riff_from_audio(
    audio_path: str,
    start_time: float = 0.0,
    duration: Optional[float] = None,
    pitch_tracking_method: str = "pyin"
) -> Riff:
    """
    Extract a riff from an audio file using pitch tracking.
    
    WARNING: Audio-to-riff conversion is approximate and experimental.
    For best results, use MIDI sources or manual transcription.
    
    Args:
        audio_path: Path to audio file
        start_time: Start time in seconds
        duration: Duration in seconds (None = full file)
        pitch_tracking_method: Algorithm to use ('pyin', 'crepe')
        
    Returns:
        Riff object (approximate)
    """
    try:
        import librosa
    except ImportError:
        raise ImportError("librosa required. Run: pip install librosa")
    
    # Load audio
    y, sr = librosa.load(audio_path, sr=22050, offset=start_time, duration=duration)
    
    # Pitch tracking
    if pitch_tracking_method == "pyin":
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('E2'),  # Low E on guitar
            fmax=librosa.note_to_hz('E6')   # High E + 2 octaves
        )
    else:
        raise ValueError(f"Unknown pitch tracking method: {pitch_tracking_method}")
    
    # Convert frequencies to MIDI pitches
    pitches = []
    durations = []
    velocities = []
    
    hop_length = 512
    time_step = hop_length / sr
    
    current_pitch = None
    current_duration = 0.0
    current_velocity = 0.0
    
    for i, (freq, voiced, prob) in enumerate(zip(f0, voiced_flag, voiced_probs)):
        if voiced and not np.isnan(freq):
            midi_pitch = librosa.hz_to_midi(freq)
            rounded_pitch = round(midi_pitch)
            
            if current_pitch is None:
                # Start new note
                current_pitch = rounded_pitch
                current_duration = time_step
                current_velocity = prob
            elif abs(rounded_pitch - current_pitch) < 0.5:
                # Continue current note
                current_duration += time_step
                current_velocity = max(current_velocity, prob)
            else:
                # New note - save previous
                if current_duration > 0.05:  # Minimum 50ms
                    pitches.append(current_pitch)
                    # Convert to beats (assuming 120 BPM)
                    durations.append(current_duration * 2.0)
                    velocities.append(current_velocity)
                
                current_pitch = rounded_pitch
                current_duration = time_step
                current_velocity = prob
    
    # Save last note
    if current_pitch is not None and current_duration > 0.05:
        pitches.append(current_pitch)
        durations.append(current_duration * 2.0)
        velocities.append(current_velocity)
    
    if not pitches:
        raise ValueError("No pitches detected in audio")
    
    # Create riff
    riff = Riff.from_absolute_pitches(
        pitches=pitches,
        durations=durations,
        velocities=velocities,
        tempo=120.0,
        metadata={"source": audio_path, "method": "audio_extraction"}
    )
    
    return riff


def batch_process_midi_directory(
    directory: str,
    pattern: str = "*.mid",
    **kwargs
) -> List[Riff]:
    """
    Process all MIDI files in a directory.
    
    Args:
        directory: Directory containing MIDI files
        pattern: Glob pattern for file matching
        **kwargs: Arguments for extract_riff_from_midi
        
    Returns:
        List of Riff objects
    """
    from pathlib import Path
    
    midi_dir = Path(directory)
    midi_files = list(midi_dir.glob(pattern))
    
    riffs = []
    for midi_file in midi_files:
        try:
            riff = extract_riff_from_midi(str(midi_file), **kwargs)
            riff.metadata["filename"] = midi_file.name
            riffs.append(riff)
            print(f"✓ Processed: {midi_file.name}")
        except Exception as e:
            print(f"✗ Failed: {midi_file.name} - {e}")
    
    print(f"\nProcessed {len(riffs)}/{len(midi_files)} files")
    return riffs


def load_dataset_from_csv(
    csv_path: str,
    pitch_column: str = "pitches",
    duration_column: str = "durations",
    metadata_columns: Optional[List[str]] = None
) -> List[Riff]:
    """
    Load riff dataset from CSV file.
    
    CSV format:
    pitches,durations,artist,song,year,genre
    "60,63,65,60","0.5,0.5,1.0,1.0",Deep Purple,Smoke on the Water,1972,Hard Rock
    
    Args:
        csv_path: Path to CSV file
        pitch_column: Column name for pitch sequences
        duration_column: Column name for duration sequences
        metadata_columns: Columns to include as metadata
        
    Returns:
        List of Riff objects
    """
    import pandas as pd
    
    df = pd.read_csv(csv_path)
    
    riffs = []
    for _, row in df.iterrows():
        # Parse sequences
        pitches = [float(p) for p in str(row[pitch_column]).split(',')]
        durations = [float(d) for d in str(row[duration_column]).split(',')]
        
        # Extract metadata
        metadata = {}
        if metadata_columns:
            for col in metadata_columns:
                if col in row:
                    metadata[col] = row[col]
        
        # Create riff
        riff = Riff.from_absolute_pitches(
            pitches=pitches,
            durations=durations,
            metadata=metadata
        )
        riffs.append(riff)
    
    return riffs


def save_riff_collection(
    riffs: List[Riff],
    output_path: str,
    format: str = "json"
):
    """
    Save a collection of riffs to file.
    
    Args:
        riffs: List of Riff objects
        output_path: Output file path
        format: Format ('json', 'pickle')
    """
    import json
    import pickle
    
    if format == "json":
        data = [riff.to_dict() for riff in riffs]
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    elif format == "pickle":
        with open(output_path, 'wb') as f:
            pickle.dump(riffs, f)
    
    else:
        raise ValueError(f"Unknown format: {format}")
    
    print(f"Saved {len(riffs)} riffs to {output_path}")


def load_riff_collection(
    input_path: str,
    format: str = "json"
) -> List[Riff]:
    """
    Load a collection of riffs from file.
    
    Args:
        input_path: Input file path
        format: Format ('json', 'pickle')
        
    Returns:
        List of Riff objects
    """
    import json
    import pickle
    
    if format == "json":
        with open(input_path, 'r') as f:
            data = json.load(f)
        riffs = [Riff.from_dict(d) for d in data]
    
    elif format == "pickle":
        with open(input_path, 'rb') as f:
            riffs = pickle.load(f)
    
    else:
        raise ValueError(f"Unknown format: {format}")
    
    print(f"Loaded {len(riffs)} riffs from {input_path}")
    return riffs


def create_synthetic_dataset(
    n_riffs: int = 100,
    year_range: Tuple[int, int] = (1960, 2025),
    genres: Optional[List[str]] = None
) -> List[Riff]:
    """
    Create a synthetic dataset for testing.
    
    Args:
        n_riffs: Number of riffs to generate
        year_range: Range of years to sample
        genres: List of genres to assign
        
    Returns:
        List of synthetic Riff objects
    """
    if genres is None:
        genres = ["Rock", "Metal", "Punk", "Grunge", "Alternative"]
    
    riffs = []
    
    for i in range(n_riffs):
        # Random parameters
        length = np.random.randint(4, 12)
        
        # Generate pitch intervals (favor small intervals)
        intervals = [0] + [np.random.choice([-2, -1, 0, 1, 2, 3], p=[0.1, 0.2, 0.2, 0.2, 0.2, 0.1]) 
                          for _ in range(length - 1)]
        
        # Generate durations (favor common rhythms)
        durations = [np.random.choice([0.25, 0.5, 1.0, 1.5], p=[0.2, 0.5, 0.2, 0.1]) 
                    for _ in range(length)]
        
        # Metadata
        year = np.random.randint(year_range[0], year_range[1])
        genre = np.random.choice(genres)
        
        riff = Riff(
            pitch_intervals=intervals,
            durations=durations,
            metadata={
                "id": i,
                "year": year,
                "genre": genre,
                "synthetic": True
            }
        )
        riffs.append(riff)
    
    return riffs
