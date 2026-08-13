# Getting Started with RiffSpace

Welcome to **RiffSpace**! This guide will help you get up and running quickly.

## Installation

### 1. Clone or Download

```bash
cd lemma-forge
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
python examples/demo.py
```

You should see output analyzing example riffs!

## Quick Tour (5 minutes)

### Create Your First Riff

```python
from src import Riff

# Classic rock riff: 0-2-2 pattern (think "Smoke on the Water")
riff = Riff(
    pitch_intervals=[0, 2, 2, -1],     # Melody: up 2, up 2, down 1
    durations=[0.5, 0.5, 0.5, 0.5]     # All quarter notes
)

print(riff)
# Output: Riff(length=4, duration=1.50b, tempo=120.0bpm)
```

### Compare Two Riffs

```python
from src import create_example_riff, compare_metrics

smoke = create_example_riff("smoke_on_the_water")
iron_man = create_example_riff("iron_man")

distances = compare_metrics(smoke, iron_man)
print(distances)
# Output: {'edit_distance': 6.5, 'dtw': 5.56, ...}
```

### Build Riff Space

```python
from src import RiffSpace

space = RiffSpace(metric='edit_distance')
space.add_riffs([smoke, iron_man])

# How far apart are they?
distance = space.distance(smoke, iron_man)
print(f"Distance: {distance:.2f}")
```

### Analyze Novelty

```python
from src import RiffCollection, NoveltyAnalyzer, create_synthetic_dataset

# Generate synthetic dataset
riffs = create_synthetic_dataset(n_riffs=50, year_range=(1960, 2020))

# Build collection
collection = RiffCollection()
for riff in riffs:
    collection.add(riff)

# Analyze novelty
analyzer = NoveltyAnalyzer(space=collection.space)
scores = analyzer.analyze_collection(collection)

# When was rock most innovative?
years, novelties = analyzer.get_temporal_profile()
print(f"Peak innovation: {years[novelties.argmax()]}")
```

## Key Concepts

### 1. Interval-Based Representation

Riffs are represented as **intervals** between notes, not absolute pitches:

```python
# Instead of: [60, 62, 64, 63] (C, D, E, D)
# We use:     [0, 2, 2, -1]     (start, up 2, up 2, down 1)
```

This makes riffs **transposition-invariant**: "Smoke on the Water" in E and F# are the same!

### 2. Equivalence Classes

Two riffs are "equivalent" if they differ only by:
- Transposition (different key)
- Tempo (faster/slower)
- Octave (higher/lower)

Distance between riffs minimizes over these transformations.

### 3. Novelty Score

A riff's novelty is its distance from **all prior riffs**:

```
N(Rₜ) = min_{Rᵢ: tᵢ < t} d(Rₜ, Rᵢ)
```

High novelty = innovative, low novelty = derivative.

## Common Tasks

### Load MIDI Files

```python
from src.pipeline import extract_riff_from_midi

riff = extract_riff_from_midi('path/to/riff.mid', track_idx=0)
```

### Save Your Work

```python
from src.pipeline import save_riff_collection

save_riff_collection(my_riffs, 'my_collection.json')
```

### Find Similar Riffs

```python
space = RiffSpace()
space.add_riffs(all_my_riffs)

# Find 5 nearest neighbors
neighbors = space.nearest_neighbors(query_riff, k=5)

for neighbor, distance in neighbors:
    print(f"{neighbor.metadata['song']}: {distance:.2f}")
```

### Visualize (requires umap-learn)

```python
from src.visualization import plot_riff_space_2d

fig = plot_riff_space_2d(collection, method='umap', color_by='year')
plt.show()
```

## Example Workflows

### Workflow 1: Analyze Your Guitar Riff Collection

```python
from src import *

# 1. Extract riffs from MIDI files
from src.pipeline import batch_process_midi_directory
riffs = batch_process_midi_directory('~/Music/guitar_riffs/')

# 2. Build collection with metadata
collection = RiffCollection()
for riff in riffs:
    collection.add(riff)

# 3. Compute novelty
analyzer = NoveltyAnalyzer(space=collection.space)
scores = analyzer.analyze_collection(collection)

# 4. Find most innovative riffs
print("Top 10 most novel riffs:")
for score in analyzer.most_novel_riffs(k=10):
    print(f"{score.riff.metadata['song']}: {score.novelty:.2f}")
```

### Workflow 2: Compare Genres

```python
# Get riffs by genre
metal_riffs = collection.get_by_genre('Metal')
punk_riffs = collection.get_by_genre('Punk')

# Compute average distances
space = RiffSpace(metric='dtw')

metal_space = RiffSpace(metric='dtw')
metal_space.add_riffs(metal_riffs)

punk_space = RiffSpace(metric='dtw')
punk_space.add_riffs(punk_riffs)

# Are genres distinct?
within_metal = np.mean(metal_space.distance_matrix()[np.triu_indices(len(metal_riffs), k=1)])
within_punk = np.mean(punk_space.distance_matrix()[np.triu_indices(len(punk_riffs), k=1)])

print(f"Within-genre distance (Metal): {within_metal:.2f}")
print(f"Within-genre distance (Punk): {within_punk:.2f}")
```

### Workflow 3: Track Your Own Riff's Novelty

```python
# Create your riff
my_riff = Riff(
    pitch_intervals=[0, 3, 5, -2, 7],
    durations=[0.5, 0.5, 0.5, 0.5, 1.0],
    metadata={'song': 'My New Riff', 'year': 2026}
)

# Compare to existing corpus
from src.novelty import quantify_derivative

derivative_score, similar_riffs = quantify_derivative(
    my_riff,
    corpus=all_my_riffs,
    threshold=2.0
)

if derivative_score > 0.7:
    print("Warning: Your riff is quite similar to existing riffs!")
    print("Similar riffs:")
    for riff in similar_riffs:
        print(f"  - {riff.metadata['song']}")
else:
    print("✓ Your riff is novel!")
```

## Next Steps

1. **Explore the notebook**: `notebooks/01_quick_start.ipynb`
2. **Read the theory**: `docs/mathematical_framework.md`
3. **Check the tests**: `tests/` for more examples
4. **Run the demo**: `python examples/demo.py`

## Troubleshooting

### "POT library required"

Optional transport metric requires POT:
```bash
pip install POT
```

### "UMAP not installed"

For visualizations:
```bash
pip install umap-learn
```

### "pretty_midi required"

For MIDI processing:
```bash
pip install pretty_midi
```

### "librosa required"

For audio extraction:
```bash
pip install librosa
```

## Getting Help

- Check the examples: `examples/README.md`
- Read the docs: `docs/mathematical_framework.md`
- Look at tests: `tests/test_*.py`
- Open an issue on GitHub

## Research Questions to Explore

Once you're comfortable with the basics, try answering:

1. **When was rock most innovative?** Plot N(t) from 1960-2025
2. **Do genres form distinct manifolds?** Compare within vs. between-genre distances
3. **Can you detect influence?** Build influence networks from similarity
4. **Are famous riffs central?** Compute centrality scores
5. **What makes a riff memorable?** Correlate features with popularity

Happy riff analyzing! 🎸
