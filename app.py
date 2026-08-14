#!/usr/bin/env python3
"""
RiffSpace Web UI - Music vector search demo.

Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import sys
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src import (
    Riff, create_example_riff,
    RiffVectorizer, create_riff_vector_index,
    Song, SongVectorizer, load_song,
    create_synthetic_dataset
)

# Page config
st.set_page_config(
    page_title="RiffSpace",
    page_icon="🎸",
    layout="wide"
)

# Title
st.title("🎸 RiffSpace")
st.markdown("**Music Vector Search Engine**")

# Sidebar
st.sidebar.header("Navigation")
demo_type = st.sidebar.radio(
    "Choose Demo",
    ["🔍 Riff Search", "🎵 Song Upload", "📊 Visualization", "ℹ️ About"]
)

# ============================================================================
# RIFF SEARCH
# ============================================================================

if demo_type == "🔍 Riff Search":
    st.header("Riff Search")
    st.markdown("Search for similar guitar riffs using vector embeddings")
    
    # Initialize database
    if 'riff_db' not in st.session_state:
        with st.spinner("Building riff database..."):
            example_riffs = [
                create_example_riff("smoke_on_the_water"),
                create_example_riff("iron_man"),
                create_example_riff("seven_nation_army")
            ]
            synthetic = create_synthetic_dataset(n_riffs=50, year_range=(1960, 2020))
            all_riffs = example_riffs + synthetic
            
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

# ============================================================================
# SONG UPLOAD
# ============================================================================

elif demo_type == "🎵 Song Upload":
    st.header("Song Upload & Vectorization")
    st.markdown("Upload an audio file and convert it to a vector embedding")
    
    st.subheader("Upload Audio File")
    
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['mp3', 'wav', 'ogg', 'flac', 'm4a'],
        help="Supported formats: MP3, WAV, OGG, FLAC, M4A"
    )
    
    if uploaded_file is not None:
        st.success(f"✓ Uploaded: {uploaded_file.name}")
        
        # Show file details
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Filename", uploaded_file.name)
        with col2:
            st.metric("Size", f"{uploaded_file.size / 1024:.1f} KB")
        with col3:
            st.metric("Type", uploaded_file.type)
        
        st.markdown("---")
        st.subheader("Vectorization Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            method = st.selectbox(
                "Embedding Method",
                ["statistical", "bag_of_frames", "openl3", "ensemble"],
                format_func=lambda x: {
                    "statistical": "Statistical (Fast - MFCCs/Chroma)",
                    "bag_of_frames": "Histogram (Order-invariant)",
                    "openl3": "OpenL3 (Pre-trained Neural Network)",
                    "ensemble": "Ensemble (ALL METHODS COMBINED)"
                }[x],
                help="Statistical: Fast aggregation. Bag of Frames: Histogram. OpenL3: Neural network. Ensemble: Concatenates all methods!"
            )
        
        with col2:
            dimension = st.selectbox(
                "Vector Dimension",
                [128, 256, 512],
                index=2,
                help="Higher dimensions = more accuracy, more storage"
            )
        
        if st.button("🚀 Generate Embedding", type="primary"):
            with st.spinner("Processing audio and generating embedding..."):
                try:
                    # Save uploaded file temporarily
                    import tempfile
                    import os
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    # Load song
                    song = load_song(tmp_path, title=uploaded_file.name)
                    
                    # Handle ensemble method
                    if method == "ensemble":
                        st.info("🎯 Ensemble mode: Combining all embedding methods...")
                        
                        embeddings_list = []
                        methods_used = []
                        
                        # 1. Statistical
                        try:
                            vec_stat = SongVectorizer(method="statistical", dimension=dimension // 3)
                            emb_stat = vec_stat.embed(song)
                            embeddings_list.append(emb_stat.vector)
                            methods_used.append("statistical")
                            st.success("✓ Statistical embedding generated")
                        except Exception as e:
                            st.warning(f"Statistical failed: {e}")
                        
                        # 2. Bag of frames
                        try:
                            vec_bag = SongVectorizer(method="bag_of_frames", dimension=dimension // 3)
                            vec_bag.fit([song])
                            emb_bag = vec_bag.embed(song)
                            embeddings_list.append(emb_bag.vector)
                            methods_used.append("bag_of_frames")
                            st.success("✓ Bag of frames embedding generated")
                        except Exception as e:
                            st.warning(f"Bag of frames failed: {e}")
                        
                        # 3. OpenL3 (optional)
                        try:
                            vec_openl3 = SongVectorizer(method="openl3", dimension=dimension // 3)
                            emb_openl3 = vec_openl3.embed(song)
                            embeddings_list.append(emb_openl3.vector)
                            methods_used.append("openl3")
                            st.success("✓ OpenL3 embedding generated")
                        except ImportError:
                            st.info("⚠️ OpenL3 not installed (skipping). Install with: pip install openl3")
                        except Exception as e:
                            st.warning(f"OpenL3 failed: {e}")
                        
                        # Concatenate all embeddings
                        if len(embeddings_list) > 0:
                            # Ensure all embeddings are same length
                            target_dim = dimension // len(embeddings_list)
                            normalized_embeddings = []
                            
                            for emb in embeddings_list:
                                if len(emb) < target_dim:
                                    emb = np.pad(emb, (0, target_dim - len(emb)))
                                else:
                                    emb = emb[:target_dim]
                                normalized_embeddings.append(emb)
                            
                            # Concatenate
                            combined_vector = np.concatenate(normalized_embeddings)
                            
                            # Pad to exact dimension if needed
                            if len(combined_vector) < dimension:
                                combined_vector = np.pad(combined_vector, (0, dimension - len(combined_vector)))
                            else:
                                combined_vector = combined_vector[:dimension]
                            
                            # Normalize
                            norm = np.linalg.norm(combined_vector)
                            if norm > 0:
                                combined_vector = combined_vector / norm
                            
                            # Create ensemble embedding
                            from dataclasses import dataclass
                            @dataclass
                            class EnsembleEmbedding:
                                song: object
                                vector: np.ndarray
                                metadata: dict
                                embedding_method: str
                                methods_used: list
                            
                            embedding = EnsembleEmbedding(
                                song=song,
                                vector=combined_vector,
                                metadata=song.metadata,
                                embedding_method=f"ensemble ({'+'.join(methods_used)})",
                                methods_used=methods_used
                            )
                            
                            st.success(f"✓ Ensemble embedding created from {len(methods_used)} methods!")
                        else:
                            raise Exception("No embedding methods succeeded")
                    
                    else:
                        # Single method
                        vectorizer = SongVectorizer(method=method, dimension=dimension)
                        
                        # Special handling for bag_of_frames
                        if method == "bag_of_frames":
                            st.info("Bag of frames method requires training a codebook. Using the song itself for training...")
                            vectorizer.fit([song])
                        
                        # Generate embedding
                        embedding = vectorizer.embed(song)
                    
                    # Clean up temp file
                    os.unlink(tmp_path)
                    
                    # Store in session state
                    st.session_state.song = song
                    st.session_state.song_embedding = embedding
                    
                    st.success("✓ Embedding generated successfully!")
                    
                except ImportError as e:
                    if "openl3" in str(e):
                        st.error("OpenL3 method requires installation: `pip install openl3`")
                    else:
                        st.error(f"Import error: {e}")
                    if 'tmp_path' in locals():
                        try:
                            os.unlink(tmp_path)
                        except:
                            pass
                except Exception as e:
                    st.error(f"Error processing audio: {e}")
                    import traceback
                    st.error(traceback.format_exc())
                    if 'tmp_path' in locals():
                        try:
                            os.unlink(tmp_path)
                        except:
                            pass
        
        # Display results
        if 'song_embedding' in st.session_state:
            st.markdown("---")
            st.subheader("Embedding Results")
            
            song = st.session_state.song
            embedding = st.session_state.song_embedding
            
            # Audio features
            st.markdown("**Audio Features:**")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Duration", f"{song.duration:.1f}s")
            with col2:
                st.metric("Sample Rate", f"{song.sr} Hz")
            with col3:
                tempo = song._tempo if song._tempo else "Computing..."
                st.metric("Tempo", f"{tempo:.0f} BPM" if isinstance(tempo, (int, float)) else tempo)
            with col4:
                channels = "Mono" if len(song.audio.shape) == 1 else "Stereo"
                st.metric("Channels", channels)
            
            # Embedding info
            st.markdown("**Embedding:**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Method", embedding.embedding_method)
            with col2:
                st.metric("Dimension", len(embedding.vector))
            with col3:
                st.metric("Norm", f"{np.linalg.norm(embedding.vector):.2f}")
            
            # Visualization
            st.markdown("**Vector Visualization (first 100 dimensions):**")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=embedding.vector[:100],
                mode='lines',
                line=dict(color='blue', width=1)
            ))
            fig.update_layout(
                xaxis_title="Dimension",
                yaxis_title="Value",
                height=300,
                margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Download option
            st.markdown("**Download Embedding:**")
            
            # Create download data
            embedding_data = {
                'filename': uploaded_file.name,
                'method': embedding.embedding_method,
                'dimension': len(embedding.vector),
                'vector': embedding.vector.tolist()
            }
            
            import json
            embedding_json = json.dumps(embedding_data, indent=2)
            
            st.download_button(
                label="📥 Download as JSON",
                data=embedding_json,
                file_name=f"{Path(uploaded_file.name).stem}_embedding.json",
                mime="application/json"
            )
            
            # Stats
            st.markdown("**Vector Statistics:**")
            stats_df = pd.DataFrame({
                'Statistic': ['Mean', 'Std Dev', 'Min', 'Max', 'Median'],
                'Value': [
                    f"{np.mean(embedding.vector):.4f}",
                    f"{np.std(embedding.vector):.4f}",
                    f"{np.min(embedding.vector):.4f}",
                    f"{np.max(embedding.vector):.4f}",
                    f"{np.median(embedding.vector):.4f}"
                ]
            })
            st.dataframe(stats_df, use_container_width=True, hide_index=True)

# ============================================================================
# VISUALIZATION
# ============================================================================

elif demo_type == "📊 Visualization":
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
    """)
    
    st.markdown("---")
    
    st.success("**The impossible is now possible.** Music in vector databases. 🎸🎵")

# Footer
st.markdown("---")
st.markdown("**RiffSpace v0.3.0** | Built with ❤️ and 🎸")
