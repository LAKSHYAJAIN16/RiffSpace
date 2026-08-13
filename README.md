# RiffSpace: The Geometry and Evolution of Rock Riffs

**A mathematical framework for analyzing rock riffs as points in geometric space**

## Core Concept

RiffSpace constructs a mathematical space where every guitar riff is a point, and distance corresponds to structural musical similarity. This enables quantitative analysis of:

- Musical evolution over time
- Genre boundaries and relationships
- Influence detection without metadata
- Innovation and novelty quantification

## Mathematical Foundation

### Riff Representation

Each riff is represented as a sequence:

```
R = {(Δpᵢ, Δtᵢ, aᵢ)}ᵢ₌₁ⁿ
```

Where:
- `Δpᵢ` = pitch interval (semitones)
- `Δtᵢ` = rhythmic duration/onset
- `aᵢ` = articulation/accent information

### Equivalence Classes

Riffs are considered equivalent under transformations:

```
R₁ ~ R₂ ⟺ R₁ = T(R₂), T ∈ G
```

Where `G` includes:
- Transposition (key changes)
- Tempo scaling
- Octave displacement

### Distance Metric

The distance between riff equivalence classes:

```
d([R₁], [R₂]) = inf_{T∈G} D(R₁, T(R₂))
```

### Novelty Score

A riff's novelty is its distance from all prior riffs:

```
N(Rₜ) = min_{Rᵢ: tᵢ < t} d(Rₜ, Rᵢ)
```

## Research Questions

1. **Do genres form geometrically distinct manifolds?**
2. **Does riff complexity increase or decrease through rock history?**
3. **Can you identify musical "missing links" between genres?**
4. **Are highly influential riffs unusually central in riff-space?**
5. **Can the system recover known influence relationships without artist metadata?**
6. **Does musical innovation correspond to movement into low-density regions?**
7. **Can you quantify how derivative a new riff is?**
8. **When was rock music most mathematically innovative?**

## Project Structure

```
riffspace/
├── src/
│   ├── riff.py              # Core riff representation
│   ├── transforms.py        # Transformation group G
│   ├── metrics.py           # Distance functions
│   ├── space.py            # Quotient space operations
│   ├── novelty.py          # Novelty computation
│   ├── visualization.py    # Plotting and analysis
│   └── pipeline.py         # Data processing
├── data/
│   ├── raw/                # Original MIDI/audio files
│   ├── processed/          # Extracted riff features
│   └── metadata/           # Release dates, genres, artists
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_metric_comparison.ipynb
│   ├── 03_temporal_analysis.ipynb
│   └── 04_genre_geometry.ipynb
├── tests/
│   └── test_*.py
└── docs/
    └── mathematical_framework.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from src.riff import Riff
from src.metrics import edit_distance
from src.space import RiffSpace

# Create riffs
riff1 = Riff(pitch_intervals=[0, 2, 2, -1], 
             durations=[0.5, 0.5, 0.5, 0.5],
             articulations=['palm-mute', 'palm-mute', 'accent', 'normal'])

riff2 = Riff(pitch_intervals=[0, 2, 2, -1, 3], 
             durations=[0.5, 0.5, 0.5, 0.5, 1.0])

# Initialize space
space = RiffSpace(metric='edit_distance')

# Compute distance
distance = space.distance(riff1, riff2)
print(f"Distance: {distance}")
```

## Core Features

- **Multiple Distance Metrics**: Edit distance, DTW, optimal transport, learned embeddings
- **Transformation-Invariant**: Automatic normalization under transposition, tempo, octave
- **Temporal Analysis**: Track novelty and evolution over time
- **Visualization**: t-SNE, UMAP, and custom projections of riff-space
- **Genre Analysis**: Clustering and manifold detection

## Planned Paper

**"The Geometry of Rock: Measuring Structural Novelty and Evolution in Guitar Riffs"**

Target venues: ISMIR, ICML, Nature Scientific Reports

## License

MIT

## Citation

```bibtex
@software{riffspace2026,
  title={RiffSpace: The Geometry and Evolution of Rock Riffs},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/riffspace}
}
```
