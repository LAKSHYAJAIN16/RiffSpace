#!/usr/bin/env python3
"""
RiffSpace Demo: Quick demonstration of core functionality.

Run: python examples/demo.py
"""

import sys
sys.path.append('.')

from src import (
    create_example_riff,
    RiffSpace,
    RiffCollection,
    NoveltyAnalyzer,
    compare_metrics,
    create_synthetic_dataset
)


def main():
    print("=" * 60)
    print("RiffSpace: The Geometry and Evolution of Rock Riffs")
    print("=" * 60)
    print()
    
    # 1. Create example riffs
    print("1. Loading example riffs...")
    smoke = create_example_riff("smoke_on_the_water")
    iron_man = create_example_riff("iron_man")
    seven_nation = create_example_riff("seven_nation_army")
    
    print(f"   • {smoke}")
    print(f"   • {iron_man}")
    print(f"   • {seven_nation}")
    print()
    
    # 2. Compare with different metrics
    print("2. Computing distances between 'Smoke on the Water' and 'Iron Man'...")
    distances = compare_metrics(smoke, iron_man)
    
    for metric, dist in distances.items():
        if isinstance(dist, (int, float)):
            print(f"   • {metric:25s}: {dist:.3f}")
        else:
            print(f"   • {metric:25s}: {dist}")
    print()
    
    # 3. Create riff space
    print("3. Building riff space...")
    space = RiffSpace(metric='edit_distance', normalize=True)
    space.add_riffs([smoke, iron_man, seven_nation])
    print(f"   • {space}")
    print()
    
    # 4. Distance matrix
    print("4. Computing distance matrix...")
    dist_matrix = space.distance_matrix()
    print("   Distance matrix:")
    print(f"   {dist_matrix}")
    print()
    
    # 5. Nearest neighbors
    print("5. Finding nearest neighbors to 'Smoke on the Water'...")
    neighbors = space.nearest_neighbors(smoke, k=2)
    for i, (neighbor, dist) in enumerate(neighbors, 1):
        song = neighbor.metadata.get('song', 'Unknown')
        print(f"   {i}. {song} (distance: {dist:.3f})")
    print()
    
    # 6. Temporal analysis with synthetic data
    print("6. Analyzing temporal novelty with synthetic dataset...")
    print("   Generating 100 synthetic riffs from 1960-2025...")
    synthetic_riffs = create_synthetic_dataset(n_riffs=100, year_range=(1960, 2025))
    
    collection = RiffCollection(space=RiffSpace(metric='dtw'))
    for riff in synthetic_riffs:
        collection.add(riff)
    
    print(f"   • {collection}")
    print()
    
    # 7. Novelty analysis
    print("7. Computing novelty scores...")
    analyzer = NoveltyAnalyzer(space=collection.space)
    scores = analyzer.analyze_collection(collection)
    
    print(f"   • Computed novelty for {len(scores)} riffs")
    print()
    
    # 8. Most novel riffs
    print("8. Most innovative riffs:")
    for i, score in enumerate(analyzer.most_novel_riffs(k=5), 1):
        print(f"   {i}. Year {score.year}: novelty={score.novelty:.3f}")
    print()
    
    # 9. Innovation peaks
    print("9. Detecting innovation peaks...")
    peaks = analyzer.find_peak_innovation_periods()
    peak_years = [year for year, _ in peaks[:5]]
    print(f"   • Peak innovation years: {peak_years}")
    print()
    
    # 10. Era comparison
    print("10. Comparing musical eras...")
    eras = {
        '1960s': (1960, 1969),
        '1970s': (1970, 1979),
        '1980s': (1980, 1989),
        '1990s': (1990, 1999),
        '2000s': (2000, 2009),
        '2010s': (2010, 2019),
        '2020s': (2020, 2025)
    }
    
    era_novelties = analyzer.compare_eras(eras)
    print("   Average novelty by era:")
    for era, novelty in sorted(era_novelties.items()):
        bar = '█' * int(novelty * 20)
        print(f"   • {era}: {bar} {novelty:.3f}")
    print()
    
    # 11. Summary
    print("=" * 60)
    print("✓ Demo complete!")
    print()
    print("Next steps:")
    print("  1. Load real MIDI files from your riff collection")
    print("  2. Experiment with different distance metrics")
    print("  3. Visualize the results (see notebooks/)")
    print("  4. Write your paper on the geometry of rock!")
    print("=" * 60)


if __name__ == "__main__":
    main()
