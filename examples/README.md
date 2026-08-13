# RiffSpace Examples

This directory contains example scripts and demonstrations of RiffSpace functionality.

## Quick Demo

Run the basic demonstration:

```bash
python examples/demo.py
```

This script demonstrates:
- Creating and comparing riffs
- Computing distances with multiple metrics
- Building riff space
- Temporal novelty analysis
- Era comparison

## Example Output

```
============================================================
RiffSpace: The Geometry and Evolution of Rock Riffs
============================================================

1. Loading example riffs...
   • Smoke on the Water (Deep Purple, 1972)
   • Iron Man (Black Sabbath, 1970)
   • Seven Nation Army (The White Stripes, 2003)

2. Computing distances...
   • edit_distance: 6.500
   • dtw: 5.562
   • euclidean: 5.928

... and more!
```

## Creating Custom Riffs

### From Interval Representation

```python
from src import Riff

riff = Riff(
    pitch_intervals=[0, 2, 2, -1, 3, -2],  # Melodic pattern
    durations=[0.5, 0.5, 0.5, 0.5, 1.0, 1.0],  # Rhythmic pattern
    articulations=['palm-mute', 'palm-mute', 'accent', 'normal', 'accent', 'normal']
)
```

### From MIDI Pitches

```python
riff = Riff.from_absolute_pitches(
    pitches=[60, 62, 64, 63],  # Middle C, D, E, D
    durations=[0.5, 0.5, 0.5, 0.5]
)
```

### From MIDI File

```python
from src import extract_riff_from_midi

riff = extract_riff_from_midi('path/to/riff.mid', track_idx=0)
```

## Computing Distances

```python
from src import compare_metrics, RiffSpace

# Compare with all metrics
distances = compare_metrics(riff1, riff2)

# Or use a specific metric in riff space
space = RiffSpace(metric='dtw')
distance = space.distance(riff1, riff2)
```

## Novelty Analysis

```python
from src import RiffCollection, NoveltyAnalyzer

# Build collection
collection = RiffCollection()
collection.add(riff1)
collection.add(riff2)

# Analyze novelty
analyzer = NoveltyAnalyzer(space=collection.space)
scores = analyzer.analyze_collection(collection)

# Find most novel riffs
top_novel = analyzer.most_novel_riffs(k=10)
```

## Visualization

See `notebooks/01_quick_start.ipynb` for visualization examples:

```python
from src.visualization import plot_novelty_timeline, plot_riff_space_2d

# Plot novelty over time
fig = plot_novelty_timeline(analyzer)

# Visualize riff space in 2D
fig = plot_riff_space_2d(collection, method='umap', color_by='year')
```

## Real Data Pipeline

### Batch Process MIDI Files

```python
from src.pipeline import batch_process_midi_directory

riffs = batch_process_midi_directory('path/to/midi/files')
```

### Save/Load Collections

```python
from src.pipeline import save_riff_collection, load_riff_collection

# Save
save_riff_collection(riffs, 'my_riffs.json')

# Load
riffs = load_riff_collection('my_riffs.json')
```

## Advanced: Custom Metrics

```python
from src.metrics import get_metric
from src.space import RiffSpace

# Use built-in metrics
space = RiffSpace(metric='dtw')

# Or define your own
def my_metric(riff1, riff2):
    # Your custom distance function
    return distance

space = RiffSpace()
space.metric_func = my_metric
```

## Next Steps

1. **Load your own data**: Use MIDI files from your collection
2. **Experiment**: Try different metrics and transformations
3. **Visualize**: Create plots and embeddings (see notebooks/)
4. **Research**: Answer the big questions about rock evolution!

## Questions?

See:
- Main README: `../README.md`
- Mathematical framework: `../docs/mathematical_framework.md`
- Jupyter notebooks: `../notebooks/`
- Tests: `../tests/`
