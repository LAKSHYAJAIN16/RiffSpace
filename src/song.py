"""
Full song representation and embedding for vector databases.

Extends the riff framework to handle complete audio files/songs.
"""

from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SongSegment:
    """A temporal segment of a song (verse, chorus, bridge, etc.)"""
    
    start_time: float  # seconds
    end_time: float
    segment_type: Optional[str] = None  # 'verse', 'chorus', 'bridge', 'intro', 'outro'
    features: Optional[np.ndarray] = None


class Song:
    """
    Representation of a complete song.
    
    A song is decomposed into:
    1. Global features (tempo, key, duration, etc.)
    2. Spectral/timbral features (MFCCs, chroma, etc.)
    3. Structural segments (verse, chorus patterns)
    4. Optional: extracted riffs/motifs
    """
    
    def __init__(
        self,
        audio_path: Optional[str] = None,
        sr: int = 22050,
        duration: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize song representation.
        
        Args:
            audio_path: Path to audio file (WAV, MP3, etc.)
            sr: Sample rate for audio processing
            duration: Song duration in seconds
            metadata: Artist, title, album, year, genre, etc.
        """
        self.audio_path = audio_path
        self.sr = sr
        self.duration = duration
        self.metadata = metadata or {}
        
        # Features (computed lazily)
        self._audio = None
        self._tempo = None
        self._key = None
        self._mfcc = None
        self._chroma = None
        self._spectral_contrast = None
        self._tonnetz = None
        
        # Structural analysis
        self.segments: List[SongSegment] = []
        self.beat_times = None
        self.bar_times = None
        
    @property
    def audio(self) -> np.ndarray:
        """Load audio on demand."""
        if self._audio is None and self.audio_path:
            self._load_audio()
        return self._audio
    
    def _load_audio(self):
        """Load audio file using librosa."""
        try:
            import librosa
        except ImportError:
            raise ImportError("librosa required. Run: pip install librosa")
        
        self._audio, self.sr = librosa.load(self.audio_path, sr=self.sr)
        if self.duration is None:
            self.duration = len(self._audio) / self.sr
    
    def extract_global_features(self) -> Dict[str, float]:
        """
        Extract song-level features.
        
        Returns:
            Dict of global features (tempo, key, duration, etc.)
        """
        try:
            import librosa
        except ImportError:
            raise ImportError("librosa required")
        
        y = self.audio
        
        # Tempo
        if self._tempo is None:
            tempo, _ = librosa.beat.beat_track(y=y, sr=self.sr)
            self._tempo = float(tempo)
        
        # Duration
        duration = len(y) / self.sr
        
        # Energy
        rms = librosa.feature.rms(y=y)
        energy_mean = float(np.mean(rms))
        energy_std = float(np.std(rms))
        
        # Zero-crossing rate (noisiness/percussiveness)
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = float(np.mean(zcr))
        
        # Spectral centroid (brightness)
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=self.sr)
        spectral_centroid_mean = float(np.mean(spectral_centroids))
        
        return {
            'tempo': self._tempo,
            'duration': duration,
            'energy_mean': energy_mean,
            'energy_std': energy_std,
            'zcr_mean': zcr_mean,
            'spectral_centroid': spectral_centroid_mean
        }
    
    def extract_mfcc(self, n_mfcc: int = 20) -> np.ndarray:
        """
        Extract Mel-Frequency Cepstral Coefficients.
        
        MFCCs capture timbral texture (what the song "sounds like").
        
        Args:
            n_mfcc: Number of MFCC coefficients
            
        Returns:
            MFCC matrix of shape (n_mfcc, n_frames)
        """
        if self._mfcc is not None:
            return self._mfcc
        
        try:
            import librosa
        except ImportError:
            raise ImportError("librosa required")
        
        y = self.audio
        self._mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=n_mfcc)
        return self._mfcc
    
    def extract_chroma(self) -> np.ndarray:
        """
        Extract chromagram (pitch class profile).
        
        Captures harmonic/melodic content independent of octave.
        
        Returns:
            Chroma matrix of shape (12, n_frames)
        """
        if self._chroma is not None:
            return self._chroma
        
        try:
            import librosa
        except ImportError:
            raise ImportError("librosa required")
        
        y = self.audio
        self._chroma = librosa.feature.chroma_cqt(y=y, sr=self.sr)
        return self._chroma
    
    def extract_spectral_contrast(self) -> np.ndarray:
        """
        Extract spectral contrast (peaks vs valleys in spectrum).
        
        Captures timbral texture and instrument separation.
        
        Returns:
            Spectral contrast matrix of shape (7, n_frames)
        """
        if self._spectral_contrast is not None:
            return self._spectral_contrast
        
        try:
            import librosa
        except ImportError:
            raise ImportError("librosa required")
        
        y = self.audio
        self._spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=self.sr)
        return self._spectral_contrast
    
    def extract_tonnetz(self) -> np.ndarray:
        """
        Extract tonal centroid features (tonnetz).
        
        Captures harmonic relationships between pitches.
        
        Returns:
            Tonnetz matrix of shape (6, n_frames)
        """
        if self._tonnetz is not None:
            return self._tonnetz
        
        try:
            import librosa
        except ImportError:
            raise ImportError("librosa required")
        
        y = self.audio
        self._tonnetz = librosa.feature.tonnetz(y=y, sr=self.sr)
        return self._tonnetz
    
    def segment_structure(self, n_segments: int = 8) -> List[SongSegment]:
        """
        Segment song into structural parts.
        
        Uses spectral clustering to find repeating sections
        (verse, chorus, etc.)
        
        Args:
            n_segments: Target number of segments
            
        Returns:
            List of SongSegment objects
        """
        try:
            import librosa
        except ImportError:
            raise ImportError("librosa required")
        
        y = self.audio
        
        # Compute tempogram for structural analysis
        hop_length = 512
        oenv = librosa.onset.onset_strength(y=y, sr=self.sr, hop_length=hop_length)
        tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=self.sr,
                                               hop_length=hop_length)
        
        # Segment using laplacian segmentation
        boundaries = librosa.segment.agglomerative(tempogram, n_segments)
        boundary_frames = librosa.util.fix_frames(boundaries, x_min=0)
        
        # Convert frames to time
        boundary_times = librosa.frames_to_time(boundary_frames, sr=self.sr, 
                                                 hop_length=hop_length)
        
        # Create segments
        self.segments = []
        for i in range(len(boundary_times) - 1):
            segment = SongSegment(
                start_time=float(boundary_times[i]),
                end_time=float(boundary_times[i + 1])
            )
            self.segments.append(segment)
        
        return self.segments
    
    def extract_all_features(self) -> Dict[str, np.ndarray]:
        """
        Extract all audio features.
        
        Returns:
            Dict containing all computed features
        """
        return {
            'global': self.extract_global_features(),
            'mfcc': self.extract_mfcc(),
            'chroma': self.extract_chroma(),
            'spectral_contrast': self.extract_spectral_contrast(),
            'tonnetz': self.extract_tonnetz()
        }
    
    def __repr__(self) -> str:
        title = self.metadata.get('title', 'Unknown')
        artist = self.metadata.get('artist', 'Unknown')
        dur = f"{self.duration:.1f}s" if self.duration else "unknown"
        return f"Song('{title}' by {artist}, duration={dur})"


def load_song(audio_path: str, **metadata) -> Song:
    """
    Load a song from audio file.
    
    Args:
        audio_path: Path to audio file
        **metadata: Artist, title, year, genre, etc.
        
    Returns:
        Song object
    """
    return Song(audio_path=audio_path, metadata=metadata)


def load_songs_from_directory(
    directory: str,
    pattern: str = "*.mp3",
    extract_metadata_from_filename: bool = True
) -> List[Song]:
    """
    Load all songs from a directory.
    
    Args:
        directory: Directory containing audio files
        pattern: Glob pattern for matching files
        extract_metadata_from_filename: Try to parse artist/title from filename
        
    Returns:
        List of Song objects
    """
    from pathlib import Path
    
    audio_dir = Path(directory)
    audio_files = list(audio_dir.glob(pattern))
    
    songs = []
    for audio_file in audio_files:
        metadata = {'filename': audio_file.name}
        
        if extract_metadata_from_filename:
            # Try to parse "Artist - Title.mp3" format
            name = audio_file.stem
            if ' - ' in name:
                parts = name.split(' - ', 1)
                metadata['artist'] = parts[0].strip()
                metadata['title'] = parts[1].strip()
            else:
                metadata['title'] = name
        
        song = Song(audio_path=str(audio_file), metadata=metadata)
        songs.append(song)
    
    return songs
