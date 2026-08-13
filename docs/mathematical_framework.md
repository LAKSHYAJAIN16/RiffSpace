# Mathematical Framework for RiffSpace

## Overview

RiffSpace constructs a mathematical space where every guitar riff is a point, and distance corresponds to structural musical similarity. This document provides the rigorous mathematical foundation.

## 1. Riff Representation

### 1.1 Basic Definition

A **riff** is a finite sequence of musical events:

```
R = {(Δpᵢ, Δtᵢ, aᵢ)}ᵢ₌₁ⁿ
```

Where:
- **Δpᵢ ∈ ℝ**: Pitch interval in semitones from previous note (Δp₁ = 0 or absolute root)
- **Δtᵢ ∈ ℝ₊**: Duration in beats
- **aᵢ ∈ A**: Articulation from a finite set A = {normal, palm-mute, accent, bend, ...}

### 1.2 Properties

**Length**: |R| = n (number of notes)

**Total Duration**: T(R) = Σᵢ Δtᵢ

**Interval Sequence**: π(R) = (Δp₁, Δp₂, ..., Δpₙ)

**Rhythm Sequence**: τ(R) = (Δt₁, Δt₂, ..., Δtₙ)

**Articulation Sequence**: α(R) = (a₁, a₂, ..., aₙ)

### 1.3 Why Interval Representation?

Traditional absolute pitch representation: R = {(pᵢ, tᵢ, aᵢ)} is **not transposition-invariant**.

Interval representation naturally satisfies:
```
Transpose(R, k) ≅ R  (up to equivalence)
```

This means "Smoke on the Water" in E and "Smoke on the Water" in F# are the same riff.

## 2. Transformation Group G

### 2.1 Group Structure

Define a transformation group **G** that preserves musical identity:

```
G = {Tₖ, Sλ, Oₘ, Rᵣ}
```

Where:
- **Tₖ**: Transposition by k semitones (handled by intervals)
- **Sλ**: Tempo scaling by factor λ
- **Oₘ**: Octave displacement by m octaves  
- **Rᵣ**: Retrograde (reverse)

### 2.2 Equivalence Relation

Two riffs are **equivalent** if related by a transformation:

```
R₁ ~ R₂  ⟺  ∃T ∈ G such that R₁ = T(R₂)
```

This defines **equivalence classes**:

```
[R] = {T(R) : T ∈ G}
```

### 2.3 Quotient Space

The space of riffs **modulo** transformations:

```
𝓡 = (Set of all riffs) / G
```

Each point in 𝓡 is an equivalence class [R], not a single riff.

## 3. Distance Metrics

### 3.1 Base Distance Function

A **base distance** D : Riff × Riff → ℝ₊ satisfies:
1. D(R₁, R₂) ≥ 0 (non-negative)
2. D(R, R) = 0 (identity)
3. D(R₁, R₂) = D(R₂, R₁) (symmetric)

Note: May not satisfy triangle inequality (pseudo-metric).

### 3.2 Quotient Space Distance

The distance between **equivalence classes**:

```
d([R₁], [R₂]) = inf_{T∈G} D(R₁, T(R₂))
```

This minimizes distance over all transformations of R₂.

**Computational approach** (finite approximation):
```
d([R₁], [R₂]) ≈ min_{T∈G_finite} D(R₁, T(R₂))
```

where G_finite is a finite subset of G.

### 3.3 Specific Metrics

#### Edit Distance (Levenshtein with Costs)

```
D_edit(R₁, R₂) = min number of operations to transform R₁ into R₂
```

Operations:
- Insert note: cost = 1
- Delete note: cost = 1
- Substitute note (i, j): cost = wₚ|Δpᵢ - Δpⱼ| + wₜ|Δtᵢ - Δtⱼ| + wₐ δ(aᵢ, aⱼ)

where wₚ, wₜ, wₐ are weights.

#### Dynamic Time Warping (DTW)

For sequences x = (x₁, ..., xₙ) and y = (y₁, ..., yₘ):

```
DTW(x, y) = min_{alignment γ} Σ_{(i,j)∈γ} ||xᵢ - yⱼ||²
```

Subject to:
- γ starts at (1, 1), ends at (n, m)
- γ is monotonic: if (i, j) ∈ γ and (i', j') ∈ γ with (i, j) < (i', j'), then i ≤ i' and j ≤ j'

#### Optimal Transport (Earth Mover's Distance)

Represent riffs as discrete distributions:
```
μ_R = (1/n) Σᵢ δ_{xᵢ}
```

where xᵢ = (Δpᵢ, Δtᵢ, encode(aᵢ)) ∈ ℝᵈ.

Optimal transport distance:
```
W(μ_R₁, μ_R₂) = min_{π ∈ Π(μ_R₁, μ_R₂)} ∫ c(x, y) dπ(x, y)
```

where Π(μ_R₁, μ_R₂) is the set of couplings and c(x, y) is the cost function (typically Euclidean).

## 4. Novelty Score

### 4.1 Definition

For a riff Rₜ released at time t, its **novelty** is:

```
N(Rₜ) = min_{Rᵢ : tᵢ < t} d([Rₜ], [Rᵢ])
```

The minimum distance to **all prior riffs**.

### 4.2 Interpretation

- **N(Rₜ) = ∞**: First riff ever (no prior riffs)
- **N(Rₜ) = 0**: Exact copy of a previous riff
- **Large N(Rₜ)**: Highly innovative, structurally unlike anything before
- **Small N(Rₜ)**: Derivative, similar to existing riffs

### 4.3 Temporal Evolution

The novelty function over time:

```
N : [t_start, t_end] → ℝ₊
```

Can be analyzed for:
- **Mean novelty per year**: E[N(Rₜ) | year(t) = y]
- **Innovation peaks**: Local maxima of smoothed N(t)
- **Discontinuities**: Sudden jumps indicating genre transitions

## 5. Genre Manifolds

### 5.1 Hypothesis

**Do genres form geometrically distinct submanifolds in 𝓡?**

Let Gᵢ ⊂ 𝓡 be the set of all riffs in genre i.

**Within-genre distance**:
```
D_within(Gᵢ) = E[d(R₁, R₂) | R₁, R₂ ∈ Gᵢ]
```

**Between-genre distance**:
```
D_between(Gᵢ, Gⱼ) = E[d(R₁, R₂) | R₁ ∈ Gᵢ, R₂ ∈ Gⱼ]
```

**Manifold hypothesis**: If Gᵢ forms a manifold, then:
1. D_within(Gᵢ) < D_between(Gᵢ, Gⱼ) for i ≠ j
2. Local structure: Riffs within Gᵢ form clusters in low-dimensional embeddings

### 5.2 Dimensionality Reduction

Map 𝓡 to ℝ² or ℝ³ for visualization:

**UMAP**: Preserves local topology
**t-SNE**: Emphasizes local clusters
**MDS**: Preserves pairwise distances (approximately)

## 6. Influence Networks

### 6.1 Definition

Construct a directed graph G = (V, E) where:
- **V**: Set of riffs
- **E**: Edge (Rᵢ, Rⱼ) exists if:
  - tᵢ < tⱼ (Rᵢ released before Rⱼ)
  - d([Rᵢ], [Rⱼ]) < θ (similarity threshold)
  - Rᵢ = arg min_{Rₖ:tₖ<tⱼ} d([Rₖ], [Rⱼ]) (nearest prior)

### 6.2 Network Properties

- **Centrality**: Riffs with high in-degree are "influential"
- **Communities**: Detect via Louvain, label propagation
- **Temporal flow**: Analyze how influence propagates through time

## 7. Research Questions (Formalized)

### Q1: Genre Separation

**Null hypothesis**: Genres are indistinguishable in riff space.

**Test**: Two-sample test (e.g., permutation test) on distance distributions:
```
H₀: D_within(Gᵢ) = D_between(Gᵢ, Gⱼ)
```

### Q2: Temporal Complexity

**Hypothesis**: Riff complexity changes over time.

**Metric**: Average novelty E[N(Rₜ) | year(t) = y]

**Test**: Regression, trend analysis, change-point detection

### Q3: Innovation Detection

**Hypothesis**: Major genre transitions correspond to novelty peaks.

**Test**: Correlate N(t) with known historical transitions (e.g., punk 1976, grunge 1991)

### Q4: Influence Recovery

**Hypothesis**: Influence network recovers known artist relationships without metadata.

**Validation**: Compare computed influence edges with documented influences

## 8. Implementation Notes

### 8.1 Computational Complexity

- **Distance matrix**: O(n² × |G_finite| × C(D))
  - n: number of riffs
  - |G_finite|: number of transformations
  - C(D): cost of base distance D

- **Novelty computation**: O(n² × |G_finite| × C(D))
  - Must compare each riff to all prior riffs

### 8.2 Optimizations

1. **Caching**: Store computed d([R₁], [R₂])
2. **Approximate search**: Use k-NN with approximate methods (Annoy, FAISS)
3. **Coarse-graining**: Pre-filter with fast lower bounds before exact computation
4. **Parallelization**: Distance computations are embarrassingly parallel

### 8.3 Learned Metrics

Replace hand-crafted D with learned distance:

**Siamese network**:
```
D_learned(R₁, R₂) = ||f_θ(R₁) - f_θ(R₂)||
```

where f_θ : Riff → ℝᵈ is a neural encoder.

**Training**: Supervised (similar/dissimilar pairs) or contrastive (triplet loss)

## 9. Expected Paper Structure

### Title
"The Geometry of Rock: Measuring Structural Novelty and Evolution in Guitar Riffs"

### Abstract
- Define riff space and quotient metric
- Introduce novelty score N(Rₜ)
- Key findings: when was rock most innovative? genre manifolds?

### Sections
1. **Introduction**: Motivation, related work
2. **Mathematical Framework**: Sections 1-3 of this doc
3. **Metrics**: Section 3.3, experiments comparing metrics
4. **Novelty Analysis**: Section 4, temporal plots
5. **Genre Geometry**: Section 5, manifold visualization
6. **Influence Networks**: Section 6, recovery of known influences
7. **Discussion**: Interpretation, limitations
8. **Conclusion**: Contributions, future work

### Potential Venues
- **ISMIR**: International Society for Music Information Retrieval
- **ICML**: If emphasizing ML methods
- **Nature Scientific Reports**: If findings are culturally significant

## References

- Levenshtein, V. (1966). Binary codes capable of correcting deletions, insertions, and reversals.
- Sakoe, H., & Chiba, S. (1978). Dynamic programming algorithm optimization for spoken word recognition.
- Cuturi, M. (2013). Sinkhorn distances: Lightspeed computation of optimal transport.
- McInnes, L., et al. (2018). UMAP: Uniform Manifold Approximation and Projection.
