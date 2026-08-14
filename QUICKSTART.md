# RiffSpace Quick Start Guide

Get up and running in 5 minutes!

## Installation

```bash
# 1. Install core dependencies
pip install -r requirements.txt

# 2. Install FAISS for vector search
pip install faiss-cpu

# 3. Install Streamlit for UI demo
pip install streamlit

# Or install everything at once
pip install -e ".[vectordb,ui]"
```

## Run the Web UI Demo

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser!

## Quick Code Examples

### 1. Search Riffs

```python
from src import create_example_riff, create_riff_vector_index

# Load famous riffs
riffs = [
    create_example_riff("smoke_on_the_water"),
    create_example_riff("iron_man"),
    create_example_riff("seven_nation_army")
]

# Create searchable index
db, vectorizer = create_riff_vector_index(
    riffs,
    backend='faiss',
    dimension=128
)

# Search
query = riffs[0]
query_emb = vectorizer.embed(query)
results = db.search(query_emb.vector, k=5)

print("Similar riffs:", results)
```

### 2. Analyze Songs

```python
from src import load_songs_from_directory, create_song_vector_index

# Load your music library
songs = load_songs_from_directory('~/Music/', pattern='*.mp3')

# Create searchable index
db, vectorizer = create_song_vector_index(
    songs,
    method='statistical',
    backend='faiss',
    dimension=512
)

# Find similar songs
query_song = songs[0]
query_emb = vectorizer.embed(query_song)
results = db.search(query_emb.vector, k=10)

for idx, distance, metadata in results:
    print(f"{metadata['title']} - {metadata['artist']}: {distance:.2f}")
```

### 3. Create Custom Riff

```python
from src import Riff

# Create your own riff
my_riff = Riff(
    pitch_intervals=[0, 2, 2, -1, 3, -2],  # Melody
    durations=[0.5, 0.5, 0.5, 0.5, 1.0, 1.0],  # Rhythm
    articulations=['palm-mute', 'palm-mute', 'accent', 'normal', 'accent', 'normal']
)

print(my_riff)
# Riff(length=6, duration=3.50b, tempo=120.0bpm)
```

## Run CLI Demos

```bash
# Riff demo
python examples/demo.py

# Vector DB demo
python examples/vectordb_demo.py

# Song analysis demo
python examples/song_demo.py
```

## Run Tests

```bash
pytest tests/ -v
```

## Using the Makefile (Unix/Mac/WSL)

```bash
make help          # Show all commands
make install       # Install dependencies
make ui            # Run web UI
make demo          # Run all CLI demos
make test          # Run tests
```

## Project Structure

```
riffspace/
├── app.py              # 🌐 Web UI (Streamlit)
├── src/                # 📦 Core library
│   ├── riff.py        # Riff representation
│   ├── song.py        # Full song support
│   ├── vectordb.py    # Vector DB integration
│   └── ...
├── examples/           # 💻 CLI demos
├── tests/             # 🧪 Unit tests
└── README.md          # 📚 Full documentation
```

## What Can I Do?

### For Riffs (4-20 notes)
- ✅ Search similar guitar riffs
- ✅ Detect plagiarism
- ✅ Track musical innovation over time
- ✅ Build influence networks

### For Full Songs (3-5 minutes)
- ✅ Music recommendations
- ✅ Find cover versions
- ✅ Playlist generation
- ✅ Audio search (Shazam-like)

## Next Steps

1. **Try the Web UI**: `streamlit run app.py`
2. **Read the docs**: `README.md`, `VECTOR_DB_GUIDE.md`
3. **Explore examples**: Check `examples/` directory
4. **Build your app**: Use the API in your own project

## Need Help?

- 📚 Full docs: `README.md`
- 🎓 Tutorial: `GETTING_STARTED.md`
- 🏗️ Architecture: `ARCHITECTURE.md`
- 🗄️ Vector DB guide: `VECTOR_DB_GUIDE.md`

---

**The impossible is now possible. Music in vector databases.** 🎸🎵
