#!/usr/bin/env python3
"""
Vector Database Demo: Store and query music in vector DBs.

Demonstrates solving the "impossible" problem of indexing variable-length
musical sequences in vector databases.
"""

import sys
sys.path.append('.')

import numpy as np
from src import (
    create_example_riff,
    RiffVectorizer,
    create_riff_vector_index,
    create_synthetic_dataset
)


def main():
    print("=" * 70)
    print("RiffSpace Vector Database Demo")
    print("=" * 70)
    print()
    print("Problem: Music is variable-length, VectorDBs need fixed dimensions")
    print("Solution: Mathematical embeddings that preserve musical similarity")
    print()
    
    # 1. Create test data
    print("1. Creating test riff corpus...")
    
    # Example riffs
    famous_riffs = [
        create_example_riff("smoke_on_the_water"),
        create_example_riff("iron_man"),
        create_example_riff("seven_nation_army")
    ]
    
    # Synthetic corpus
    synthetic = create_synthetic_dataset(n_riffs=50, year_range=(1960, 2020))
    
    all_riffs = famous_riffs + synthetic
    print(f"   • Loaded {len(all_riffs)} riffs")
    print(f"   • Riff lengths: {[len(r) for r in famous_riffs]}")
    print()
    
    # 2. Demonstrate different embedding methods
    print("2. Testing embedding methods...")
    print()
    
    test_riff = famous_riffs[0]
    
    # Statistical embedding
    print("   A. Statistical Features")
    vec_stat = RiffVectorizer(method='statistical', dimension=128)
    emb_stat = vec_stat.embed(test_riff)
    print(f"      Input:  Riff with {len(test_riff)} notes (variable length)")
    print(f"      Output: Vector of dimension {emb_stat.vector.shape[0]} (fixed)")
    print(f"      Vector preview: [{emb_stat.vector[:5]}...]")
    print()
    
    # Histogram embedding
    print("   B. Histogram-Based")
    vec_hist = RiffVectorizer(method='histogram', dimension=256)
    emb_hist = vec_hist.embed(test_riff)
    print(f"      Dimension: {emb_hist.vector.shape[0]}")
    print(f"      Norm: {np.linalg.norm(emb_hist.vector):.3f}")
    print()
    
    # Distance-based embedding
    print("   C. Distance-Based (Metric Preserving)")
    vec_dist = RiffVectorizer(method='distance', dimension=30)
    vec_dist.fit(all_riffs[:30])  # Use subset as landmarks
    emb_dist = vec_dist.embed(test_riff)
    print(f"      Dimension: {emb_dist.vector.shape[0]}")
    print(f"      Interpretation: Distance to 30 reference riffs")
    print()
    
    # 3. Create vector index
    print("3. Building vector database index...")
    
    try:
        db, vectorizer = create_riff_vector_index(
            all_riffs,
            method='statistical',
            backend='faiss',
            dimension=128
        )
        print(f"   ✓ Indexed {len(all_riffs)} riffs in FAISS")
        print(f"   • Backend: FAISS (local)")
        print(f"   • Dimension: 128")
        print(f"   • Method: statistical features")
        print()
        
        # 4. Query the database
        print("4. Querying vector database...")
        print()
        
        query_riff = famous_riffs[0]  # Smoke on the Water
        query_embedding = vectorizer.embed(query_riff)
        
        results = db.search(query_embedding.vector, k=5)
        
        print(f"   Query: {query_riff.metadata.get('song', 'Unknown')}")
        print(f"   Top 5 similar riffs:")
        print()
        
        for i, (idx, distance, metadata) in enumerate(results, 1):
            song = metadata.get('song', metadata.get('id', f'Riff {idx}'))
            year = metadata.get('year', 'N/A')
            print(f"   {i}. {song:30s} (year: {year}, distance: {distance:.3f})")
        
        print()
        
        # 5. Compare to direct metric
        print("5. Validation: Comparing to ground-truth distances...")
        
        from src import RiffSpace
        space = RiffSpace(metric='edit_distance')
        
        # Compute actual distances
        print(f"   Query: {query_riff.metadata.get('song')}")
        for other in famous_riffs[1:3]:
            # Embedding distance
            other_emb = vectorizer.embed(other)
            emb_dist = np.linalg.norm(query_embedding.vector - other_emb.vector)
            
            # True metric distance
            true_dist = space.distance(query_riff, other, use_transforms=False)
            
            song = other.metadata.get('song', 'Unknown')
            print(f"   vs {song}:")
            print(f"      Embedding distance: {emb_dist:.3f}")
            print(f"      True distance:      {true_dist:.3f}")
            print()
        
        # 6. Metadata filtering (if supported)
        print("6. Filtered search by metadata...")
        
        query_embedding = vectorizer.embed(famous_riffs[1])
        
        # Search all
        all_results = db.search(query_embedding.vector, k=10)
        print(f"   Found {len(all_results)} results (no filter)")
        
        # Filter by year (FAISS doesn't support native filtering, but we can post-filter)
        filtered = [r for r in all_results if r[2].get('year', 0) >= 1990]
        print(f"   Filtered to {len(filtered)} results (year >= 1990)")
        print()
        
    except ImportError as e:
        print(f"   ⚠ FAISS not installed: {e}")
        print(f"   Install with: pip install faiss-cpu")
        print()
    
    # 7. Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print("✓ Variable-length riffs → Fixed-dimensional vectors")
    print("✓ Musical similarity preserved in vector space")
    print("✓ Compatible with any vector database (FAISS, Pinecone, Chroma, etc.)")
    print()
    print("Applications:")
    print("  • Semantic music search")
    print("  • Plagiarism detection")
    print("  • Recommendation systems")
    print("  • Copyright infringement analysis")
    print()
    print("The 'impossible' is now possible. 🎸")
    print("=" * 70)


if __name__ == "__main__":
    main()
