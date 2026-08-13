"""
Visualization tools for riff space analysis.

Creates:
- t-SNE / UMAP projections
- Temporal evolution plots
- Genre manifold visualizations
- Influence networks
"""

from typing import List, Optional, Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .riff import Riff
from .space import RiffSpace, RiffCollection
from .novelty import NoveltyAnalyzer, NoveltyScore


def plot_riff_space_2d(
    collection: RiffCollection,
    method: str = "umap",
    color_by: str = "year",
    figsize: Tuple[int, int] = (12, 8),
    **kwargs
) -> plt.Figure:
    """
    Project riff space to 2D and visualize.
    
    Args:
        collection: RiffCollection to visualize
        method: Dimensionality reduction ('umap', 'tsne', 'pca', 'mds')
        color_by: Metadata field for coloring ('year', 'genre', 'artist')
        figsize: Figure size
        **kwargs: Additional arguments for reduction algorithm
        
    Returns:
        matplotlib Figure
    """
    if len(collection) < 3:
        raise ValueError("Need at least 3 riffs for visualization")
    
    # Compute distance matrix
    dist_matrix = collection.space.distance_matrix()
    
    # Dimensionality reduction
    if method == "umap":
        try:
            import umap
            reducer = umap.UMAP(
                metric='precomputed',
                n_neighbors=min(15, len(collection) - 1),
                **kwargs
            )
            embedding = reducer.fit_transform(dist_matrix)
        except ImportError:
            raise ImportError("UMAP not installed. Run: pip install umap-learn")
    
    elif method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(
            metric='precomputed',
            n_components=2,
            **kwargs
        )
        embedding = reducer.fit_transform(dist_matrix)
    
    elif method == "pca":
        from sklearn.decomposition import PCA
        from sklearn.manifold import MDS
        # PCA needs feature vectors, use MDS for distance matrix
        reducer = MDS(n_components=2, dissimilarity='precomputed', **kwargs)
        embedding = reducer.fit_transform(dist_matrix)
    
    elif method == "mds":
        from sklearn.manifold import MDS
        reducer = MDS(n_components=2, dissimilarity='precomputed', **kwargs)
        embedding = reducer.fit_transform(dist_matrix)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Extract color values
    colors = []
    color_map = {}
    for riff in collection.riffs:
        value = riff.metadata.get(color_by, "unknown")
        colors.append(value)
        if value not in color_map:
            color_map[value] = len(color_map)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    if color_by == "year" and all(isinstance(c, int) for c in colors if c != "unknown"):
        # Continuous colormap for years
        numeric_colors = [c if c != "unknown" else np.nan for c in colors]
        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=numeric_colors,
            cmap='viridis',
            s=100,
            alpha=0.7
        )
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Year', rotation=270, labelpad=20)
    else:
        # Discrete colormap for genres/artists
        numeric_colors = [color_map[c] for c in colors]
        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=numeric_colors,
            cmap='tab10',
            s=100,
            alpha=0.7
        )
        
        # Legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=plt.cm.tab10(color_map[label] / 10),
                      markersize=10, label=label)
            for label in sorted(color_map.keys())
        ]
        ax.legend(handles=legend_elements, title=color_by.capitalize(),
                 bbox_to_anchor=(1.15, 1), loc='upper left')
    
    ax.set_xlabel(f'{method.upper()} Dimension 1')
    ax.set_ylabel(f'{method.upper()} Dimension 2')
    ax.set_title(f'Riff Space Projection ({method.upper()})')
    plt.tight_layout()
    
    return fig


def plot_novelty_timeline(
    analyzer: NoveltyAnalyzer,
    aggregate: str = "mean",
    smoothing_window: Optional[int] = 3,
    figsize: Tuple[int, int] = (14, 6),
    highlight_peaks: bool = True
) -> plt.Figure:
    """
    Plot novelty over time.
    
    Args:
        analyzer: NoveltyAnalyzer with computed scores
        aggregate: How to aggregate per year
        smoothing_window: Window for moving average (None = no smoothing)
        figsize: Figure size
        highlight_peaks: Whether to highlight innovation peaks
        
    Returns:
        matplotlib Figure
    """
    years, novelties = analyzer.get_temporal_profile(aggregate=aggregate)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Original data
    ax.plot(years, novelties, 'o-', alpha=0.6, label='Raw', linewidth=2)
    
    # Smoothed data
    if smoothing_window and len(years) >= smoothing_window:
        smoothed = np.convolve(
            novelties,
            np.ones(smoothing_window) / smoothing_window,
            mode='valid'
        )
        smooth_years = years[smoothing_window // 2: -(smoothing_window // 2) + 
                            (1 if smoothing_window % 2 == 0 else 0)]
        ax.plot(smooth_years, smoothed, 'r-', linewidth=3, 
               label=f'{smoothing_window}-year moving average')
    
    # Highlight peaks
    if highlight_peaks:
        peaks = analyzer.find_peak_innovation_periods()
        if peaks:
            peak_years, peak_novelties = zip(*peaks)
            ax.scatter(peak_years, peak_novelties, s=200, c='red', 
                      marker='*', zorder=5, label='Innovation peaks')
    
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Novelty Score', fontsize=12)
    ax.set_title('Musical Innovation Over Time', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig


def plot_genre_comparison(
    collection: RiffCollection,
    genre_field: str = "genre",
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Compare riff distributions across genres.
    
    Args:
        collection: RiffCollection with genre metadata
        genre_field: Metadata field containing genre info
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    # Compute distance matrix
    dist_matrix = collection.space.distance_matrix()
    
    # Group by genre
    genres = {}
    for i, riff in enumerate(collection.riffs):
        genre = riff.metadata.get(genre_field, "Unknown")
        if genre not in genres:
            genres[genre] = []
        genres[genre].append(i)
    
    # Compute within-genre and between-genre distances
    within_genre_dists = {}
    between_genre_dists = {}
    
    for genre1, indices1 in genres.items():
        # Within-genre
        if len(indices1) > 1:
            within = [dist_matrix[i, j] for i in indices1 for j in indices1 if i < j]
            within_genre_dists[genre1] = within
        
        # Between-genre
        for genre2, indices2 in genres.items():
            if genre1 < genre2:  # Only compute once per pair
                between = [dist_matrix[i, j] for i in indices1 for j in indices2]
                between_genre_dists[f"{genre1}-{genre2}"] = between
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Within-genre distances
    ax1.boxplot(
        within_genre_dists.values(),
        labels=within_genre_dists.keys(),
        patch_artist=True
    )
    ax1.set_xlabel('Genre')
    ax1.set_ylabel('Pairwise Distance')
    ax1.set_title('Within-Genre Cohesion')
    ax1.tick_params(axis='x', rotation=45)
    
    # Between-genre distances
    if between_genre_dists:
        between_labels = list(between_genre_dists.keys())
        between_means = [np.mean(dists) for dists in between_genre_dists.values()]
        ax2.barh(range(len(between_labels)), between_means)
        ax2.set_yticks(range(len(between_labels)))
        ax2.set_yticklabels(between_labels, fontsize=9)
        ax2.set_xlabel('Mean Distance')
        ax2.set_title('Between-Genre Separation')
    
    plt.tight_layout()
    return fig


def plot_influence_network(
    analyzer: NoveltyAnalyzer,
    threshold: float = 0.5,
    layout: str = "spring",
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Visualize influence relationships between riffs.
    
    Args:
        analyzer: NoveltyAnalyzer with computed scores
        threshold: Distance threshold for edges
        layout: Graph layout algorithm ('spring', 'kamada_kawai')
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("NetworkX required. Run: pip install networkx")
    
    # Build graph
    G = nx.DiGraph()
    
    edges = analyzer.influence_network(threshold=threshold)
    
    # Add nodes and edges
    node_to_idx = {}
    for i, score in enumerate(analyzer.scores):
        if score.riff not in node_to_idx:
            node_to_idx[score.riff] = i
            label = score.riff.metadata.get("song", f"Riff{i}")
            G.add_node(i, label=label, year=score.year)
    
    for source, target, dist in edges:
        if source in node_to_idx and target in node_to_idx:
            G.add_edge(
                node_to_idx[source],
                node_to_idx[target],
                weight=1.0 - dist  # Higher weight = more similar
            )
    
    # Layout
    if layout == "spring":
        pos = nx.spring_layout(G, k=1, iterations=50)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.circular_layout(G)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw nodes colored by year
    years = [G.nodes[n].get('year', 0) for n in G.nodes()]
    nx.draw_networkx_nodes(
        G, pos, node_color=years, cmap='viridis',
        node_size=500, alpha=0.8, ax=ax
    )
    
    # Draw edges
    nx.draw_networkx_edges(
        G, pos, alpha=0.3, arrows=True,
        arrowsize=20, edge_color='gray', ax=ax
    )
    
    # Labels
    labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
    
    ax.set_title('Riff Influence Network', fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    
    return fig


def plot_era_comparison(
    analyzer: NoveltyAnalyzer,
    era_ranges: Dict[str, Tuple[int, int]],
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Compare novelty across different musical eras.
    
    Args:
        analyzer: NoveltyAnalyzer with computed scores
        era_ranges: Dict mapping era names to (start_year, end_year)
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    era_novelties = analyzer.compare_eras(era_ranges)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    eras = list(era_novelties.keys())
    novelties = list(era_novelties.values())
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(eras)))
    bars = ax.bar(eras, novelties, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar, value in zip(bars, novelties):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
               f'{value:.2f}',
               ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Era', fontsize=12)
    ax.set_ylabel('Average Novelty Score', fontsize=12)
    ax.set_title('Musical Innovation by Era', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    
    return fig


def save_figures(figures: Dict[str, plt.Figure], output_dir: str = "figures"):
    """
    Save multiple figures to files.
    
    Args:
        figures: Dict mapping filenames to Figure objects
        output_dir: Output directory
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for name, fig in figures.items():
        filepath = os.path.join(output_dir, f"{name}.png")
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
