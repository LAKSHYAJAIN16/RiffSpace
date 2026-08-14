#!/usr/bin/env python3
"""
RiffSpace Web UI - Minimalistic music vector search demo.

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
    create_synthetic_dataset
)

# Page config
st.set_page_config(
    page_title="RiffSpace",
    page_icon="🎸",
    layout="wide"
)

# Minimalistic CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background: #FFFFFF;
    }
    
    .main .block-container {
        padding-top: 3rem;
        max-width: 1200px;
    }
    
    h1 {
        color: #000000;
        font-weight: 700;
        font-size: 2.5rem !important;
        letter-spacing: -0.02em;
    }
    
    h2 {
        color: #000000;
        font-weight: 600;
        font-size: 1.75rem !important;
        margin-top: 2.5rem;
    }
    
    h3 {
        color: #333333;
        font-weight: 600;
        font-size: 1.25rem !important;
    }
    
    [data-testid="stSidebar"] {
        background: #F8F9FA;
        border-right: 1px solid #E9ECEF;
    }
    
    .stButton > button {
        background: #000000;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.6rem 1.5rem;
        font-weight: 500;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        background: #333333;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 600;
        color: #000000;
    }
    
    [data-testid="stMetricLabel"] {
        color: #666666;
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    hr {
        border: none;
        border-top: 1px solid #E9ECEF;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 style='text-align: center;'>RiffSpace</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666; font-size: 1.1rem;'>Music vector search engine</p>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### Navigation")
    demo_type = st.radio(
        "nav",
        ["Riff Search", "Song Analysis", "Visualization", "About"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### Stats")
    st.markdown("**DB:** FAISS  \n**Dim:** 128/512  \n**Speed:** <1ms")

# Riff Search
if demo_type == "Riff Search":
    st.markdown("## Riff Search")
    st.markdown("Search for similar guitar riffs using vector embeddings")
    
    if 'riff_db' not in st.session_state:
        with st.spinner("Building database..."):
            progress_bar = st.progress(0)
            
            progress_bar.progress(33)
            example_riffs = [
                create_example_riff("smoke_on_the_water"),
                create_example_riff("iron_man"),
                create_example_riff("seven_nation_army")
            ]
            
            progress_bar.progress(66)
            synthetic = create_synthetic_dataset(n_riffs=50, year_range=(1960, 2020))
            all_riffs = example_riffs + synthetic
            
            db, vectorizer = create_riff_vector_index(
                all_riffs, method='statistical',
                backend='faiss', dimension=128
            )
            progress_bar.progress(100)
            
            st.session_state.riff_db = db
            st.session_state.riff_vectorizer = vectorizer
            st.session_state.riffs = all_riffs
        st.success("Database ready (53 riffs)")
    
    st.markdown("### Choose Riff")
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("**Famous Riffs**")
        famous_riff = st.selectbox(
            "riff",
            ["Smoke on the Water", "Iron Man", "Seven Nation Army"],
            label_visibility="collapsed"
        )
        
        if st.button("Load", use_container_width=True):
            riff_map = {
                "Smoke on the Water": "smoke_on_the_water",
                "Iron Man": "iron_man",
                "Seven Nation Army": "seven_nation_army"
            }
            st.session_state.query_riff = create_example_riff(riff_map[famous_riff])
            st.success(f"Loaded: {famous_riff}")
    
    with col2:
        st.markdown("**Custom Riff**")
        intervals_str = st.text_input(
            "Intervals", "0, 2, 2, -1, 3",
            label_visibility="collapsed",
            placeholder="Pitch intervals (e.g. 0, 2, 2, -1, 3)"
        )
        
        if st.button("Create", use_container_width=True):
            try:
                intervals = [float(x.strip()) for x in intervals_str.split(',')]
                durations = [0.5] * len(intervals)
                st.session_state.query_riff = Riff(
                    pitch_intervals=intervals,
                    durations=durations,
                    metadata={'name': 'Custom'}
                )
                st.success("Created")
            except Exception as e:
                st.error(f"Error: {e}")
    
    if 'query_riff' in st.session_state:
        st.markdown("---")
        st.markdown("### Query Riff")
        
        query_riff = st.session_state.query_riff
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Notes", len(query_riff))
        col2.metric("Duration", f"{query_riff.total_duration:.1f}b")
        col3.metric("Tempo", query_riff.tempo)
        col4.metric("Unique", len(set(query_riff.get_interval_sequence())))
        
        intervals = query_riff.get_interval_sequence()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(intervals))), y=intervals,
            mode='lines+markers',
            marker=dict(size=8, color='#000'),
            line=dict(color='#000', width=2)
        ))
        fig.update_layout(
            xaxis_title="Position", yaxis_title="Interval",
            height=200, margin=dict(l=20, r=20, t=20, b=20),
            plot_bgcolor='white', paper_bgcolor='white',
            font=dict(family='Inter', size=11)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    if 'query_riff' in st.session_state:
        st.markdown("---")
        st.markdown("### Search")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            k = st.slider("Results", 1, 20, 10)
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            search = st.button("Search", type="primary", use_container_width=True)
        
        if search:
            with st.spinner("Searching..."):
                query_emb = st.session_state.riff_vectorizer.embed(st.session_state.query_riff)
                results = st.session_state.riff_db.search(query_emb.vector, k=k)
                st.session_state.search_results = results
        
        if 'search_results' in st.session_state:
            st.markdown("---")
            st.markdown("### Results")
            
            data = []
            for i, (idx, dist, meta) in enumerate(st.session_state.search_results, 1):
                data.append({
                    '#': i,
                    'Song': str(meta.get('song', f'Riff {idx}'))[:40],
                    'Artist': meta.get('artist', 'Unknown'),
                    'Year': meta.get('year', 'N/A'),
                    'Distance': f"{dist:.2f}",
                    'Match': f"{(1/(1+dist)):.1%}"
                })
            
            st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)

# Song Analysis
elif demo_type == "Song Analysis":
    st.markdown("## Song Analysis")
    st.markdown("How audio files become searchable vectors")
    
    st.info("Demo mode: showing feature pipeline")
    
    st.markdown("### Features")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Extracted**
        - Tempo (BPM)
        - MFCCs (timbre)
        - Chroma (harmony)
        - Spectral contrast
        - Tonnetz (tonal)
        """)
    with col2:
        st.markdown("""
        **Aggregated**
        - Mean & std
        - Min & max
        - Quartiles
        
        → 512 dimensions
        """)
    
    st.markdown("### Dimensions")
    
    fig = go.Figure(data=[go.Bar(
        x=['Global', 'MFCC', 'Chroma', 'Spectral', 'Tonnetz', 'Pad'],
        y=[6, 100, 48, 21, 12, 325],
        marker_color='#000'
    )])
    fig.update_layout(
        height=250, margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor='white', paper_bgcolor='white',
        font=dict(family='Inter', size=11)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Example")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Tempo", "72", "BPM")
    col2.metric("Duration", "5:54")
    col3.metric("Energy", "0.43")
    col4.metric("Spectral", "2.8k", "Hz")
    col5.metric("ZCR", "0.09")

# Visualization
elif demo_type == "Visualization":
    st.markdown("## Visualization")
    st.markdown("Riffs in vector space")
    
    with st.spinner("Computing..."):
        riffs = [
            create_example_riff("smoke_on_the_water"),
            create_example_riff("iron_man"),
            create_example_riff("seven_nation_army")
        ]
        names = ["Smoke on the Water", "Iron Man", "Seven Nation Army"]
        vectorizer = RiffVectorizer(method='statistical', dimension=128)
        embeddings = [vectorizer.embed(r).vector for r in riffs]
    
    st.markdown("### Embeddings (30 dims)")
    
    fig = go.Figure()
    for name, emb in zip(names, embeddings):
        fig.add_trace(go.Scatter(
            x=list(range(30)), y=emb[:30],
            mode='lines', name=name, line=dict(width=2)
        ))
    fig.update_layout(
        height=300, margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor='white', paper_bgcolor='white',
        font=dict(family='Inter', size=11)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Distance Matrix")
    
    from sklearn.metrics.pairwise import euclidean_distances
    distances = euclidean_distances(embeddings)
    
    fig = go.Figure(data=go.Heatmap(
        z=distances, x=names, y=names,
        colorscale='Greys',
        text=np.round(distances, 2),
        texttemplate='%{text}'
    ))
    fig.update_layout(
        height=300, margin=dict(l=20, r=20, t=20, b=20),
        font=dict(family='Inter', size=11)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("Lower = more similar")

# About
elif demo_type == "About":
    st.markdown("## About")
    
    st.markdown("### Problem")
    st.markdown("Music is variable-length, but vector DBs need fixed dimensions")
    
    col1, col2, col3 = st.columns(3)
    col1.markdown("**Padding**  \nWastes space")
    col2.markdown("**Truncation**  \nLoses data")
    col3.markdown("**Raw audio**  \nToo large")
    
    st.markdown("---")
    st.markdown("### Solution")
    
    col1, col2, col3 = st.columns(3)
    col1.markdown("**Intervals**  \nRelative pitch")
    col2.markdown("**Equivalence**  \nTransformations")
    col3.markdown("**Projection**  \nFixed dims")
    
    st.markdown("---")
    st.markdown("### Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Riff", "0.1ms")
    col2.metric("Song", "2-5s")
    col3.metric("Search", "<1ms")
    col4.metric("Accuracy", "0.94")
    
    st.markdown("---")
    st.markdown("### Applications")
    
    st.markdown("""
    - Music search
    - Plagiarism detection
    - Recommendations
    - Cover detection
    - Copyright analysis
    """)

# Footer
st.markdown("---")
st.markdown("<div style='text-align:center;color:#999;font-size:0.85rem'>RiffSpace v0.3.0</div>", unsafe_allow_html=True)
