"""
Distance metrics for riff space.

Implements multiple distance functions:
- Edit distance (Levenshtein with custom costs)
- Dynamic Time Warping (DTW)
- Optimal Transport
- Euclidean/Manhattan on feature representations
"""

import numpy as np
from typing import Callable, Optional, Tuple
from .riff import Riff


def edit_distance(
    riff1: Riff,
    riff2: Riff,
    weights: Optional[Tuple[float, float, float]] = None
) -> float:
    """
    Weighted edit distance between two riffs.
    
    Computes Levenshtein distance with custom costs for pitch, rhythm,
    and articulation differences.
    
    Args:
        riff1, riff2: Riffs to compare
        weights: (pitch_weight, rhythm_weight, articulation_weight)
        
    Returns:
        Edit distance (lower = more similar)
    """
    if weights is None:
        weights = (1.0, 0.5, 0.3)  # Prioritize pitch > rhythm > articulation
    
    wp, wr, wa = weights
    
    n, m = len(riff1), len(riff2)
    
    # DP table
    dp = np.zeros((n + 1, m + 1))
    
    # Initialize base cases (insertion/deletion costs)
    for i in range(n + 1):
        dp[i, 0] = i
    for j in range(m + 1):
        dp[0, j] = j
    
    # Fill DP table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            note1 = riff1.notes[i - 1]
            note2 = riff2.notes[j - 1]
            
            # Substitution cost based on note differences
            pitch_diff = abs(note1.pitch_interval - note2.pitch_interval)
            rhythm_diff = abs(note1.duration - note2.duration)
            artic_diff = 0.0 if note1.articulation == note2.articulation else 1.0
            
            subst_cost = wp * pitch_diff + wr * rhythm_diff + wa * artic_diff
            
            dp[i, j] = min(
                dp[i - 1, j] + 1.0,  # deletion
                dp[i, j - 1] + 1.0,  # insertion
                dp[i - 1, j - 1] + subst_cost  # substitution
            )
    
    return dp[n, m]


def dtw_distance(
    riff1: Riff,
    riff2: Riff,
    feature: str = "all",
    window: Optional[int] = None
) -> float:
    """
    Dynamic Time Warping distance between riffs.
    
    Allows flexible alignment of sequences, useful for riffs with timing variations.
    
    Args:
        riff1, riff2: Riffs to compare
        feature: Which features to use ('all', 'pitch', 'rhythm')
        window: Sakoe-Chiba band constraint (None = no constraint)
        
    Returns:
        DTW distance
    """
    # Extract feature vectors
    if feature == "pitch":
        seq1 = riff1.get_interval_sequence().reshape(-1, 1)
        seq2 = riff2.get_interval_sequence().reshape(-1, 1)
    elif feature == "rhythm":
        seq1 = riff1.get_rhythm_sequence().reshape(-1, 1)
        seq2 = riff2.get_rhythm_sequence().reshape(-1, 1)
    else:  # 'all'
        arr1 = riff1.as_array()[:, :3]  # pitch, duration, velocity
        arr2 = riff2.as_array()[:, :3]
        seq1 = arr1
        seq2 = arr2
    
    n, m = len(seq1), len(seq2)
    
    # Initialize cost matrix with infinity
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    
    # Fill cost matrix
    for i in range(1, n + 1):
        # Apply window constraint if specified
        j_start = max(1, i - window if window else 1)
        j_end = min(m + 1, i + window + 1 if window else m + 1)
        
        for j in range(j_start, j_end):
            # Euclidean distance between feature vectors
            dist = np.linalg.norm(seq1[i - 1] - seq2[j - 1])
            
            cost[i, j] = dist + min(
                cost[i - 1, j],      # insertion
                cost[i, j - 1],      # deletion
                cost[i - 1, j - 1]   # match
            )
    
    return cost[n, m]


def optimal_transport_distance(
    riff1: Riff,
    riff2: Riff,
    reg: float = 0.1
) -> float:
    """
    Earth Mover's Distance (Wasserstein) between riffs using optimal transport.
    
    Treats riffs as distributions over feature space and computes
    optimal cost to transform one into the other.
    
    Args:
        riff1, riff2: Riffs to compare
        reg: Entropic regularization parameter (smaller = more accurate, slower)
        
    Returns:
        Optimal transport distance
    """
    try:
        import ot  # Python Optimal Transport library
    except ImportError:
        raise ImportError("POT library required. Install: pip install POT")
    
    # Get feature representations
    features1 = riff1.as_array()[:, :3]  # pitch, duration, velocity
    features2 = riff2.as_array()[:, :3]
    
    # Uniform distributions over notes
    weights1 = np.ones(len(riff1)) / len(riff1)
    weights2 = np.ones(len(riff2)) / len(riff2)
    
    # Pairwise cost matrix (Euclidean distance)
    cost_matrix = ot.dist(features1, features2, metric='euclidean')
    
    # Solve optimal transport with entropic regularization (faster)
    transport_cost = ot.sinkhorn2(weights1, weights2, cost_matrix, reg)
    
    return float(transport_cost)


def euclidean_distance(riff1: Riff, riff2: Riff) -> float:
    """
    Euclidean distance between riff feature vectors.
    
    Simple but requires riffs of same length. Pads shorter riff with zeros.
    """
    arr1 = riff1.as_array()
    arr2 = riff2.as_array()
    
    # Pad shorter sequence
    max_len = max(len(arr1), len(arr2))
    if len(arr1) < max_len:
        arr1 = np.vstack([arr1, np.zeros((max_len - len(arr1), arr1.shape[1]))])
    if len(arr2) < max_len:
        arr2 = np.vstack([arr2, np.zeros((max_len - len(arr2), arr2.shape[1]))])
    
    return float(np.linalg.norm(arr1 - arr2))


def cosine_similarity(riff1: Riff, riff2: Riff) -> float:
    """
    Cosine similarity between riff feature vectors.
    
    Returns value in [0, 1] where 1 = identical, 0 = orthogonal.
    Convert to distance by: d = 1 - similarity
    """
    arr1 = riff1.as_array().flatten()
    arr2 = riff2.as_array().flatten()
    
    # Pad to same length
    max_len = max(len(arr1), len(arr2))
    arr1 = np.pad(arr1, (0, max_len - len(arr1)))
    arr2 = np.pad(arr2, (0, max_len - len(arr2)))
    
    # Compute cosine similarity
    dot_product = np.dot(arr1, arr2)
    norm_product = np.linalg.norm(arr1) * np.linalg.norm(arr2)
    
    if norm_product == 0:
        return 0.0
    
    similarity = dot_product / norm_product
    return float(1.0 - similarity)  # Convert to distance


def interval_histogram_distance(riff1: Riff, riff2: Riff) -> float:
    """
    Distance based on pitch interval histograms.
    
    Good for capturing melodic character while being order-invariant.
    """
    # Create histograms over interval range [-12, 12] semitones
    bins = np.arange(-12, 13, 1)
    
    intervals1 = riff1.get_interval_sequence()
    intervals2 = riff2.get_interval_sequence()
    
    hist1, _ = np.histogram(intervals1, bins=bins, density=True)
    hist2, _ = np.histogram(intervals2, bins=bins, density=True)
    
    # Chi-squared distance
    epsilon = 1e-10
    chi2 = np.sum((hist1 - hist2) ** 2 / (hist1 + hist2 + epsilon))
    
    return float(chi2)


# Metric registry for easy access
METRICS = {
    "edit_distance": edit_distance,
    "dtw": dtw_distance,
    "optimal_transport": optimal_transport_distance,
    "euclidean": euclidean_distance,
    "cosine": cosine_similarity,
    "interval_histogram": interval_histogram_distance,
}


def get_metric(name: str) -> Callable[[Riff, Riff], float]:
    """
    Get a distance metric function by name.
    
    Args:
        name: Metric name (see METRICS dict)
        
    Returns:
        Distance function
    """
    if name not in METRICS:
        raise ValueError(f"Unknown metric: {name}. Available: {list(METRICS.keys())}")
    return METRICS[name]


def compare_metrics(riff1: Riff, riff2: Riff) -> dict:
    """
    Compare two riffs using all available metrics.
    
    Returns:
        Dictionary of {metric_name: distance}
    """
    results = {}
    
    for name, metric_func in METRICS.items():
        try:
            results[name] = metric_func(riff1, riff2)
        except Exception as e:
            results[name] = f"Error: {str(e)}"
    
    return results
