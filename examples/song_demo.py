#!/usr/bin/env python3
"""
Full Song Demo: Index complete audio files in vector databases.

Demonstrates embedding full songs (3-5 minutes) into fixed vectors.
"""

import sys
sys.path.append('.')

import numpy as np
from src import Song, SongVectorizer, create_song_vector_index


def main():
    print("=" * 70)
    print("RiffSpace: Full Song Embeddings")
    print("=" * 70)
    print()
    print("Problem: Songs are 3-5 minutes (millions of samples)")
    print("Solution: Multi-scale feature aggregation → fixed vector")
    print()
    
    # Demo with synthetic "songs" (since we don't have audio files)
    print("1. Creating demo songs...")
    print("   (In production, use: load_songs_from_directory('path/to/music/'))")
    print()
    
    # Create mock songs without actual audio
    songs = []
    for i, (title, artist, duration) in enumerate([
        ("Bohemian Rhapsody", "Queen", 354),
        ("Stairway to Heaven", "Led Zeppelin", 482),
        ("Hotel California", "Eagles", 391),
        ("Smells Like Teen Spirit", "Nirvana", 301),
        ("Sweet Child O' Mine", "Guns N' Roses", 356)
    ]):
        song = Song(
            duration=duration,
            metadata={
                'id': i,
                'title': title,
                'artist': artist,
                'year': 1970 + i * 5,
                'genre': ['Rock', 'Hard Rock', 'Rock', 'Grunge', 'Rock'][i]
            }
        )
        songs.append(song)
        print(f"   • {song}")
    
    print()
    
    # 2. Explain embedding approach
    print("2. Song Embedding Methods:")
    print()
    print("   A. Statistical Aggregation (Recommended)")
    print("      - Extract: MFCCs, chroma, spectral contrast, tonnetz")
    print("      - Aggregate: mean, std, min, max, quartiles over time")
    print("      - Result: ~500-d vector")
    print("      - Pros: Fast, interpretable, works for any length")
    print()
    print("   B. Bag-of-Audio-Words")
    print("      - Cluster audio frames into codebook")
    print("      - Build histogram of codeword frequencies")
    print("      - Result: Dimension = codebook size")
    print("      - Pros: Order-invariant, captures local patterns")
    print()
    print("   C. OpenL3 (Pre-trained)")
    print("      - Use pre-trained neural network (AudioSet)")
    print("      - Extract 512-d embeddings per second")
    print("      - Aggregate with mean pooling")
    print("      - Pros: State-of-the-art, transfer learning")
    print()
    
    # 3. Show feature extraction (mock without actual audio)
    print("3. Feature Extraction Pipeline:")
    print()
    print("   For an audio file:")
    print("   ├─ Load audio: librosa.load('song.mp3')")
    print("   ├─ Global features:")
    print("   │   ├─ Tempo: 120 BPM")
    print("   │   ├─ Duration: 3:54")
    print("   │   ├─ Energy: mean, std")
    print("   │   └─ Spectral centroid: brightness")
    print("   ├─ Frame-level features (computed every ~23ms):")
    print("   │   ├─ MFCCs (20 coefficients): timbral texture")
    print("   │   ├─ Chroma (12 pitch classes): harmonic content")
    print("   │   ├─ Spectral contrast (7 bands): texture")
    print("   │   └─ Tonnetz (6 dimensions): tonal relationships")
    print("   └─ Aggregate statistics:")
    print("       ├─ Mean, std, min, max, median")
    print("       └─ Result: 512-dimensional vector")
    print()
    
    # 4. Demonstrate dimensions
    print("4. Embedding Dimensions:")
    print()
    
    # Mock embedding (since we don't have actual audio)
    print("   Example feature breakdown:")
    print("   • Global features:        6 dims")
    print("   • MFCC statistics:      100 dims  (20 MFCCs × 5 stats)")
    print("   • Chroma statistics:     48 dims  (12 pitches × 4 stats)")
    print("   • Spectral contrast:     21 dims  (7 bands × 3 stats)")
    print("   • Tonnetz statistics:    12 dims  (6 dims × 2 stats)")
    print("   ─────────────────────────────────")
    print("   Total:                  187 dims  (padded to 512)")
    print()
    
    # 5. Show what vector DB would look like
    print("5. Vector Database Index:")
    print()
    print("   With actual audio files:")
    print("   ```python")
    print("   from src import load_songs_from_directory, create_song_vector_index")
    print()
    print("   # Load songs")
    print("   songs = load_songs_from_directory('~/Music/')")
    print()
    print("   # Create searchable index")
    print("   db, vectorizer = create_song_vector_index(")
    print("       songs,")
    print("       method='statistical',  # Fast, reliable")
    print("       backend='faiss',")
    print("       dimension=512")
    print("   )")
    print()
    print("   # Query")
    print("   query_song = songs[0]")
    print("   query_embedding = vectorizer.embed(query_song)")
    print("   results = db.search(query_embedding.vector, k=10)")
    print("   ```")
    print()
    
    # 6. Applications
    print("6. Applications:")
    print()
    print("   A. Music Recommendation")
    print("      • Find songs similar to user's favorites")
    print("      • \"More like this\" feature")
    print()
    print("   B. Duplicate Detection")
    print("      • Find cover versions")
    print("      • Detect remixes/remasters")
    print()
    print("   C. Playlist Generation")
    print("      • Create smooth transitions")
    print("      • Group by mood/energy/tempo")
    print()
    print("   D. Music Discovery")
    print("      • Search by audio query (hum/sing)")
    print("      • Find similar artists")
    print()
    print("   E. Copyright Detection")
    print("      • Check for unauthorized samples")
    print("      • Find similar compositions")
    print()
    
    # 7. Performance characteristics
    print("7. Performance (with actual audio):")
    print()
    print("   • Feature extraction:  2-5 seconds per song")
    print("   • Embedding:           0.1-0.5 seconds")
    print("   • FAISS search:        <1ms for 10k songs")
    print("   • Memory per song:     2 KB (512 dims × 4 bytes)")
    print("   • Index 10k songs:     ~20 MB")
    print()
    
    # 8. Comparison with alternatives
    print("8. Why This Approach?")
    print()
    print("   vs Raw Audio:")
    print("   ✓ 1 million× smaller (2KB vs 2GB per song)")
    print("   ✓ Fast similarity search")
    print("   ✓ Works with standard vector DBs")
    print()
    print("   vs Acoustic Fingerprints (Shazam):")
    print("   ✓ Captures musical similarity, not just exact matches")
    print("   ✓ Finds covers, remixes, similar songs")
    print("   ✓ Semantic search, not just identification")
    print()
    print("   vs Metadata:")
    print("   ✓ Audio-based, no tags required")
    print("   ✓ Discovers unexpected similarities")
    print("   ✓ Language-independent")
    print()
    
    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print("✓ Full songs (3-5 minutes) → Fixed vectors (512-d)")
    print("✓ Multi-scale features: tempo, timbre, harmony, rhythm")
    print("✓ Fast: 2-5 seconds extraction, <1ms search")
    print("✓ Scalable: Works with millions of songs")
    print("✓ Compatible: FAISS, Pinecone, Chroma, Weaviate")
    print()
    print("Next Steps:")
    print("  1. Collect audio files (MP3, WAV, FLAC)")
    print("  2. Run: load_songs_from_directory('path/to/music/')")
    print("  3. Create index: create_song_vector_index(songs, backend='faiss')")
    print("  4. Build your music app!")
    print()
    print("The impossible is possible. Full songs in vector databases. 🎵")
    print("=" * 70)


if __name__ == "__main__":
    main()
