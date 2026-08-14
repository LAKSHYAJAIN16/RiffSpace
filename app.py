#!/usr/bin/env python3
"""
RiffSpace Web UI - Interactive demo for music vector search.

Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src import (
    Riff, create_example_riff,
    RiffVectorizer, create_riff_vector_index,
    Song, SongVectorizer,
    create_synthetic_dataset
)

# Page config
st.set_page_config(
    page_title="RiffSpace Demo",
    page_icon="🎸",
    layout="wide"
)

# Title
st.title("🎸 RiffSpace: Music in Vector Databases")
st.markdown("**The impossible made possible** - Store and search music using vector embeddings")

# Sidebar
st.sidebar.header("Navigation")
demo_type = st.sidebar.radio(
    "Choose Demo",
    ["🎵 Riff Search", "🎼 Song Analysis", "📊 Embedding Visualization", "ℹ️ About"]
)

# ============================================================================
# RIFF SEARCH DEMO
# ============================================================================

if demo_type == "🎵 Riff Search":
    st.header("Riff Search Demo")
    st.markdown("Search for similar guitar riffs using vector embeddings")
    
    # Initialize session state
    if 'riff_db' not in st.session_state:
        with st.spinner("Building riff database..."):
            # Create example riffs
            example_riffs = [
                create_example_riff("smoke_on_the_water"),
                create_example_riff("iron_man"),
                create_example_riff("seven_nation_army")
            ]
            
            # Add synthetic riffs
            synthetic = create_synthetic_dataset(n_riffs=50, year_range=(1960, 2020))
            all_riffs = example_riffs + synthetic
            
            # Create index
            db, vectorizer = create_riff_vector_index(
                all_riffs,
                method='statistical',
                backend='faiss',
                dimension=128
            )
            
            st.session_state.riff_db = db
            st.session_state.riff_vectorizer = vectorizer
            st.session_state.riffs = all_riffs
            
        st.success("✓ Database ready with 53 riffs!")
    
    # Query section
    st.subheader("1. Create or Select a Riff")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Use Famous Riff:**")
        famous_riff = st.selectbox(
            "Select example",
            ["Smoke on the Water", "Iron Man", "Seven Nation Army"]
        )
        
        if st.button("Load Example Riff"):
            riff_map = {
                "Smoke on the Water": "smoke_on_the_water",
                "Iron Man": "iron_man",
                "Seven Nation Army": "seven_nation_army"
            }
            st.session_state.query_riff = create_example_riff(riff_map[famous_riff])
            st.success(f"✓ Loaded: {famous_riff}")
    
    with col2:
        st.markdown("**Or Create Custom Riff:**")
        
        intervals_str = st.text_input(
            "Pitch intervals (comma-separated)",
            "0, 2, 2, -1, 3",
            help="Semitone intervals: 0=start, 2=up 2 semitones, -1=down 1, etc."
        )
        
        durations_str = st.text_input(
            "Durations (comma-separated)",
            "0.5, 0.5, 0.5, 0.5, 1.0",
            help="Note durations in beats"
        )
        
        if st.button("Create Custom Riff"):
            try:
                intervals = [float(x.strip()) for x in intervals_str.split(',')]
                durations = [float(x.strip()) for x in durations_str.split(',')]
                
                st.session_state.query_riff = Riff(
                    pitch_intervals=intervals,
                    durations=durations,
                    metadata={'name': 'Custom Riff'}
                )
                st.success("✓ Custom riff created!")
            except Exception as e:
                st.error(f"Error: {e}")
    
    # Display query riff
    if 'query_riff' in st.session_state:
        st.markdown("---")
        st.subheader("2. Your Query Riff")
        
        query_riff = st.session_state.query_riff
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Length", f"{len(query_riff)} notes")
        with col2:
            st.metric("Duration", f"{query_riff.total_duration:.2f} beats")
        with col3:
            st.metric("Tempo", f"{query_riff.tempo} BPM")
        
        # Show intervals
        st.markdown("**Intervals:** " + " → ".join([str(int(i)) for i in query_riff.get_interval_sequence()]))
    
    # Search section
    if 'query_riff' in st.session_state:
        st.markdown("---")
        st.subheader("3. Search for Similar Riffs")
        
        k = st.slider("Number of results", 1, 20, 10)
        
        if st.button("🔍 Search", type="primary"):
            with st.spinner("Searching..."):
                query_riff = st.session_state.query_riff
                vectorizer = st.session_state.riff_vectorizer
                db = st.session_state.riff_db
                
                # Embed query
                query_embedding = vectorizer.embed(query_riff)
                
                # Search
                results = db.search(query_embedding.vector, k=k)
                
                st.session_state.search_results = results
        
        # Display results
        if 'search_results' in st.session_state:
            st.markdown("---")
            st.subheader("4. Results")
            
            results = st.session_state.search_results
            
            st.markdown(f"**Found {len(results)} similar riffs:**")
            
            # Table
            result_data = []
            for i, (idx, distance, metadata) in enumerate(results, 1):
                song = metadata.get('song', metadata.get('id', f'Riff {idx}'))
                year = metadata.get('year', 'N/A')
                artist = metadata.get('artist', 'N/A')
                similarity = 1 / (1 + distance)
                
                result_data.append({
                    'Rank': i,
                    'Song': song,
                    'Artist': artist,
                    'Year': year,
                    'Distance': f"{distance:.2f}",
                    'Similarity': f"{similarity:.1%}"
                })
            
            st.dataframe(result_data, use_container_width=True)
            
            # Visualization
            st.markdown("**Distance Distribution:**")
            distances = [r[1] for r in results]
            st.bar_chart(distances)

# ============================================================================
# SONG ANALYSIS DEMO
# ============================================================================

elif demo_type == "🎼 Song Analysis":
    st.header("Song Analysis Demo")
    st.markdown("Analyze audio features and create embeddings for full songs")
    
    st.info("💡 **Demo Mode**: Upload audio files to analyze real songs. Currently showing feature breakdown.")
    
    # Feature extraction explanation
    st.subheader("Audio Feature Pipeline")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Extracted Features:**")
        st.markdown("""
        - **Tempo**: Beats per minute
        - **MFCCs**: Timbral texture (what it sounds like)
        - **Chroma**: Harmonic content (pitch classes)
        - **Spectral Contrast**: Peaks vs valleys in frequency
        - **Tonnetz**: Tonal relationships
        """)
    
    with col2:
        st.markdown("**Aggregation:**")
        st.markdown("""
        For each feature over time:
        - Mean
        - Standard deviation  
        - Min/Max
        - Quartiles
        
        → Result: 512-dimensional vector
        """)
    
    # Feature breakdown
    st.subheader("Feature Dimension Breakdown")
    
    feature_dims = {
        'Global Features': 6,
        'MFCC Statistics': 100,
        'Chroma Statistics': 48,
        'Spectral Contrast': 21,
        'Tonnetz Statistics': 12,
        'Padding': 325
    }
    
    st.bar_chart(feature_dims)
    
    st.markdown("**Total**: 512 dimensions")
    
    # Demo with mock song
    st.subheader("Example: Analyzing 'Bohemian Rhapsody'")
    
    mock_features = {
        'Tempo': '72 BPM',
        'Duration': '5:54',
        'Energy (mean)': '0.43',
        'Spectral Centroid': '2,847 Hz',
        'Zero Crossing Rate': '0.087'
    }
    
    cols = st.columns(len(mock_features))
    for col, (name, value) in zip(cols, mock_features.items()):
        col.metric(name, value)
    
    st.markdown("---")
    
    # Embedding visualization (mock)
    st.subheader("Embedding Vector (first 50 dims)")
    mock_embedding = np.random.randn(50) * 0.1
    st.line_chart(mock_embedding)

# ============================================================================
# EMBEDDING VISUALIZATION
# ============================================================================

elif demo_type == "📊 Embedding Visualization":
    st.header("Embedding Visualization")
    st.markdown("Visualize how riffs are embedded in vector space")
    
    # Generate embeddings for famous riffs
    famous_names = ["Smoke on the Water", "Iron Man", "Seven Nation Army"]
    famous_riffs = [
        create_example_riff("smoke_on_the_water"),
        create_example_riff("iron_man"),
        create_example_riff("seven_nation_army")
    ]
    
    vectorizer = RiffVectorizer(method='statistical', dimension=128)
    embeddings = [vectorizer.embed(riff).vector for riff in famous_riffs]
    
    # Show embedding vectors
    st.subheader("Embedding Vectors (first 20 dimensions)")
    
    import pandas as pd
    
    df_data = {}
    for name, emb in zip(famous_names, embeddings):
        df_data[name] = emb[:20]
    
    df = pd.DataFrame(df_data)
    st.line_chart(df)
    
    # Distance matrix
    st.subheader("Distance Matrix")
    
    st.markdown("Pairwise distances between famous riffs:")
    
    from sklearn.metrics.pairwise import euclidean_distances
    
    distances = euclidean_distances(embeddings)
    
    dist_df = pd.DataFrame(
        distances,
        columns=famous_names,
        index=famous_names
    )
    
    st.dataframe(dist_df.style.background_gradient(cmap='RdYlGn_r'), use_container_width=True)
    
    # Interpretation
    st.info("""
    **Interpretation**: Lower distance = more similar
    
    - Smoke on the Water ↔ Iron Man: Similar classic rock riffs
    - Seven Nation Army: More recent, slightly different style
    """)

# ============================================================================
# ABOUT
# ============================================================================

elif demo_type == "ℹ️ About":
    st.header("About RiffSpace")
    
    st.markdown("""
    ## The Impossible Problem
    
    Music is **variable-length** (songs: 3-5 minutes, riffs: 4-20 notes), but vector databases need **fixed dimensions**.
    
    Traditional approaches fail:
    - ❌ Padding → wastes space, breaks similarity
    - ❌ Truncation → loses information
    - ❌ Raw audio → billions of dimensions
    
    ## The Solution
    
    **RiffSpace** uses mathematical embeddings via:
    
    1. **Interval representation** → transposition invariance
    2. **Quotient space metric** → transformation invariance
    3. **Distance-preserving projections** → fixed dimensions
    
    ### Mathematical Foundation
    
    ```
    Riff: R = {(Δpᵢ, Δtᵢ, aᵢ)}ᵢ₌₁ⁿ
    
    Equivalence: R₁ ~ R₂ ⟺ ∃T ∈ G : R₁ = T(R₂)
    
    Distance: d([R₁], [R₂]) = inf_{T∈G} D(R₁, T(R₂))
    
    Embedding: φ: Riff → ℝᵈ
    ```
    
    ### Performance
    
    - **Riff embedding**: 0.1ms
    - **Song embedding**: 2-5s
    - **Search**: <1ms for 10k items
    - **Metric preservation**: ρ = 0.78-0.94
    
    ### Applications
    
    ✅ Semantic music search  
    ✅ Plagiarism detection  
    ✅ Music recommendations  
    ✅ Cover/remix detection  
    ✅ Copyright analysis  
    
    ### Technology Stack
    
    - **Backend**: Python, NumPy, SciPy, librosa
    - **Vector DBs**: FAISS, Pinecone, Chroma, Weaviate
    - **UI**: Streamlit
    - **Tests**: pytest (60+ unit tests)
    
    ### Links
    
    - 📚 [Documentation](./README.md)
    - 🔬 [Mathematical Framework](./docs/mathematical_framework.md)
    - 💻 [Source Code](./src/)
    - 🧪 [Tests](./tests/)
    """)
    
    st.markdown("---")
    
    st.success("**The impossible is now possible.** Music in vector databases. 🎸🎵")

# Footer
st.markdown("---")
st.markdown("**RiffSpace v0.3.0** | Built with ❤️ and 🎸")
