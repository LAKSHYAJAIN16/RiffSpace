# RiffSpace Project Structure

Complete overview of the RiffSpace codebase.

## Directory Tree

```
lemma-forge/  (RiffSpace)
├── src/                          # Core implementation
│   ├── __init__.py              # Package exports
│   ├── riff.py                  # Riff representation (R = {(Δp, Δt, a)})
│   ├── transforms.py            # Transformation group G
│   ├── space.py                 # Quotient space 𝓡 = Riffs/G
│   ├── metrics.py               # Distance functions d([R₁], [R₂])
│   ├── novelty.py               # Novelty analysis N(Rₜ)
│   ├── visualization.py         # Plotting and dimensionality reduction
│   └── pipeline.py              # Data loading/processing
│
├── tests/                        # Unit tests
│   ├── __init__.py
│   ├── test_riff.py             # Riff representation tests
│   ├── test_metrics.py          # Distance metric tests
│   ├── test_space.py            # Riff space tests
│   └── test_novelty.py          # Novelty analysis tests
│
├── notebooks/                    # Jupyter notebooks
│   └── 01_quick_start.ipynb     # Interactive tutorial
│
├── examples/                     # Example scripts
│   ├── README.md                # Examples documentation
│   └── demo.py                  # Quick demonstration
│
├── data/                         # Data storage
│   ├── raw/                     # Original MIDI/audio files
│   ├── processed/               # Extracted riff features
│   └── metadata/                # Release dates, genres, artists
│
├── docs/                         # Documentation
│   └── mathematical_framework.md # Rigorous math foundation
│
├── README.md                     # Project overview
├── GETTING_STARTED.md           # Quick start guide
├── PROJECT_STRUCTURE.md         # This file
├── requirements.txt             # Python dependencies
└── .gitignore                   # Git ignore rules
```

## Core Modules

### `src/riff.py`

**Purpose**: Interval-based riff representation

**Key Classes**:
- `RiffNote`: Single note with (Δp, Δt, a)
- `Riff`: Sequence of notes with metadata

**Key Functions**:
- `Riff.from_absolute_pitches()`: Convert MIDI pitches to intervals
- `Riff.normalize_rhythm()`: Tempo-invariant normalization
- `create_example_riff()`: Load famous riffs

**Mathematical Concept**: R = {(Δpᵢ, Δtᵢ, aᵢ)}ᵢ₌₁ⁿ

---

### `src/transforms.py`

**Purpose**: Musical transformations for equivalence classes

**Key Classes**:
- `Transformation`: Single transformation T ∈ G
- `TransformGroup`: Collection of transformations

**Built-in Transforms**:
- Identity
- Tempo normalization
- Octave shifts (±12 semitones)
- Retrograde (reverse)
- Time stretch (double/half time)

**Mathematical Concept**: R₁ ~ R₂ ⟺ ∃T ∈ G : R₁ = T(R₂)

---

### `src/space.py`

**Purpose**: Quotient space and distance computations

**Key Classes**:
- `RiffSpace`: Space of equivalence classes [R]
- `RiffCollection`: Temporal collection with metadata

**Key Methods**:
- `distance([R₁], [R₂])`: inf_{T∈G} D(R₁, T(R₂))
- `distance_matrix()`: Pairwise distances
- `nearest_neighbors()`: k-NN search
- `compute_centrality()`: Average distance to all riffs

**Mathematical Concept**: 𝓡 = (Set of riffs) / G

---

### `src/metrics.py`

**Purpose**: Distance functions between riffs

**Implemented Metrics**:
1. **Edit Distance**: Weighted Levenshtein with musical costs
2. **DTW**: Dynamic Time Warping for alignment
3. **Optimal Transport**: Earth Mover's Distance (requires POT)
4. **Euclidean**: Simple L2 distance (with padding)
5. **Cosine**: Cosine similarity converted to distance
6. **Interval Histogram**: Chi-squared on interval distributions

**Key Function**: `compare_metrics(R₁, R₂)` - Compare all metrics

**Mathematical Concept**: D : Riff × Riff → ℝ₊

---

### `src/novelty.py`

**Purpose**: Temporal novelty analysis

**Key Classes**:
- `NoveltyScore`: Result for a single riff
- `NoveltyAnalyzer`: Full temporal analysis

**Key Methods**:
- `compute_novelty(Rₜ, prior_riffs)`: N(Rₜ) = min d(Rₜ, Rᵢ)
- `get_temporal_profile()`: N(t) over time
- `find_peak_innovation_periods()`: Detect discontinuities
- `most_novel_riffs()`: Top k innovative riffs
- `influence_network()`: Build similarity graph

**Mathematical Concept**: N(Rₜ) = min_{Rᵢ:tᵢ<t} d([Rₜ], [Rᵢ])

---

### `src/visualization.py`

**Purpose**: Plotting and dimensionality reduction

**Key Functions**:
- `plot_riff_space_2d()`: UMAP/t-SNE projection
- `plot_novelty_timeline()`: N(t) over time
- `plot_genre_comparison()`: Within vs. between distances
- `plot_influence_network()`: NetworkX graph visualization
- `plot_era_comparison()`: Bar chart by decade

**Dependencies**: matplotlib, seaborn, umap-learn, networkx

---

### `src/pipeline.py`

**Purpose**: Data loading and preprocessing

**Key Functions**:
- `extract_riff_from_midi()`: MIDI → Riff
- `extract_riff_from_audio()`: Audio → Riff (experimental)
- `batch_process_midi_directory()`: Process folder
- `save_riff_collection()`: Serialize to JSON/pickle
- `load_riff_collection()`: Deserialize
- `create_synthetic_dataset()`: Generate test data

**Dependencies**: pretty_midi, librosa, mido

---

## Test Coverage

### `tests/test_riff.py`

Tests for:
- Riff creation and validation
- Interval conversion
- Normalization
- Serialization

### `tests/test_metrics.py`

Tests for:
- All distance metrics
- Metric properties (symmetry, identity)
- Edge cases (empty riffs, identical riffs)

### `tests/test_space.py`

Tests for:
- Space creation and operations
- Distance matrix computation
- Nearest neighbor search
- Collection filtering (by year, genre)

### `tests/test_novelty.py`

Tests for:
- Novelty computation
- Temporal analysis
- Peak detection
- Era comparison

**Run tests**: `pytest tests/ -v`

---

## Documentation

### `README.md`

- Project overview
- Mathematical motivation
- Research questions
- Installation
- Quick start

### `GETTING_STARTED.md`

- 5-minute tutorial
- Common tasks
- Example workflows
- Troubleshooting

### `docs/mathematical_framework.md`

- Rigorous definitions
- Formal proofs
- Metric properties
- Expected paper structure

### `examples/README.md`

- Code examples
- Custom riff creation
- Pipeline usage
- Advanced topics

---

## Notebooks

### `notebooks/01_quick_start.ipynb`

Interactive tutorial covering:
1. Creating riffs
2. Distance metrics
3. Transformations
4. Novelty analysis
5. Genre geometry
6. Era comparison
7. Influence networks

**Run**: `jupyter notebook notebooks/01_quick_start.ipynb`

---

## Data Organization

### `data/raw/`

Original source files:
- MIDI files (`.mid`)
- Audio files (`.wav`, `.mp3`)
- Transcriptions

### `data/processed/`

Extracted features:
- Riff collections (JSON, pickle)
- Distance matrices (`.npy`)
- Precomputed embeddings

### `data/metadata/`

Additional information:
- Release dates CSV
- Genre labels
- Artist information
- Known influences

---

## Configuration

### `requirements.txt`

Core dependencies:
- `numpy`, `scipy`, `pandas`: Scientific computing
- `music21`, `librosa`, `pretty_midi`: Music processing
- `scikit-learn`, `umap-learn`: ML and dimensionality reduction
- `matplotlib`, `seaborn`, `plotly`: Visualization
- `pytest`: Testing

Optional:
- `POT`: Optimal transport metric
- `torch`: Deep learning metrics

---

## Development Workflow

### Adding a New Metric

1. Implement in `src/metrics.py`:
```python
def my_metric(riff1: Riff, riff2: Riff) -> float:
    # Your implementation
    return distance

METRICS["my_metric"] = my_metric
```

2. Add tests in `tests/test_metrics.py`
3. Update documentation

### Adding a New Transformation

1. Implement in `src/transforms.py`:
```python
def my_transform(riff: Riff) -> Riff:
    # Your implementation
    return transformed_riff

transform_group.register("my_transform", my_transform)
```

2. Add tests
3. Update `TransformGroup` docstring

### Adding a New Analysis

1. Create new file in `src/` (e.g., `complexity.py`)
2. Import in `src/__init__.py`
3. Add tests in `tests/`
4. Create notebook example

---

## Code Style

- **Formatting**: PEP 8, max line length 100
- **Type hints**: Use for function signatures
- **Docstrings**: Google style
- **Naming**: 
  - Classes: `PascalCase`
  - Functions: `snake_case`
  - Constants: `UPPER_CASE`

---

## Performance Notes

### Bottlenecks

1. **Distance matrix**: O(n² × |G| × C(D))
   - Cache results
   - Parallelize with `joblib`

2. **DTW**: O(n × m) per comparison
   - Use Sakoe-Chiba band
   - Consider FastDTW approximation

3. **Optimal Transport**: Expensive for large riffs
   - Use Sinkhorn (entropic regularization)
   - Approximate with smaller grids

### Optimization Strategies

- **Memoization**: Cache distance computations
- **Approximate NN**: Use Annoy, FAISS for large datasets
- **Batch processing**: Vectorize where possible
- **GPU acceleration**: For learned metrics (PyTorch)

---

## Research Roadmap

### Phase 1: Foundation (✓ Complete)
- Core representation
- Multiple metrics
- Novelty analysis
- Basic visualization

### Phase 2: Data Collection
- Compile rock riff corpus (1960-2025)
- Extract from MIDI/transcriptions
- Annotate metadata (dates, genres)

### Phase 3: Analysis
- Compute full novelty timeline
- Genre manifold analysis
- Influence network construction
- Statistical validation

### Phase 4: Paper
- Write mathematical framework
- Create figures and visualizations
- Run statistical tests
- Submit to ISMIR/ICML

### Phase 5: Extensions
- Learned distance metrics
- Cross-genre analysis
- Predictive modeling
- Interactive web tool

---

## Contributing

1. Fork the repository
2. Create feature branch
3. Add tests
4. Update documentation
5. Submit pull request

---

## License

MIT License - See LICENSE file

---

## Contact

For questions or collaboration:
- GitHub Issues
- Email: [your-email]
- Website: [project-site]

---

Last updated: 2026-08-13
