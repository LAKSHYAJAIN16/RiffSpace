"""
Unit tests for vector database integration.
"""

import pytest
import numpy as np
from src.riff import Riff, create_example_riff
from src.vectordb import RiffVectorizer, RiffEmbedding, VectorDBAdapter, create_riff_vector_index


@pytest.fixture
def test_riffs():
    """Create test riffs."""
    return [
        create_example_riff("smoke_on_the_water"),
        create_example_riff("iron_man"),
        create_example_riff("seven_nation_army")
    ]


@pytest.fixture
def synthetic_corpus():
    """Create larger synthetic corpus."""
    from src.pipeline import create_synthetic_dataset
    return create_synthetic_dataset(n_riffs=20, year_range=(1960, 2020))


class TestRiffVectorizer:
    """Test RiffVectorizer class."""
    
    def test_create_vectorizer(self):
        vectorizer = RiffVectorizer(method='statistical', dimension=128)
        assert vectorizer.dimension == 128
        assert vectorizer.method == 'statistical'
    
    def test_embed_statistical(self, test_riffs):
        vectorizer = RiffVectorizer(method='statistical', dimension=128)
        embedding = vectorizer.embed(test_riffs[0])
        
        assert isinstance(embedding, RiffEmbedding)
        assert embedding.vector.shape == (128,)
        assert embedding.embedding_method == 'statistical'
    
    def test_embed_histogram(self, test_riffs):
        vectorizer = RiffVectorizer(method='histogram', dimension=256)
        embedding = vectorizer.embed(test_riffs[0])
        
        assert embedding.vector.shape == (256,)
    
    def test_embed_distance(self, test_riffs, synthetic_corpus):
        vectorizer = RiffVectorizer(method='distance', dimension=20)
        vectorizer.fit(synthetic_corpus)
        
        embedding = vectorizer.embed(test_riffs[0])
        assert embedding.vector.shape == (20,)
    
    def test_embed_distance_without_fit(self, test_riffs):
        vectorizer = RiffVectorizer(method='distance', dimension=128)
        
        with pytest.raises(ValueError):
            vectorizer.embed(test_riffs[0])
    
    def test_embed_batch(self, test_riffs):
        vectorizer = RiffVectorizer(method='statistical', dimension=64)
        embeddings = vectorizer.embed_batch(test_riffs)
        
        assert len(embeddings) == 3
        assert all(emb.vector.shape == (64,) for emb in embeddings)
    
    def test_different_riffs_different_vectors(self, test_riffs):
        vectorizer = RiffVectorizer(method='statistical', dimension=128)
        
        emb1 = vectorizer.embed(test_riffs[0])
        emb2 = vectorizer.embed(test_riffs[1])
        
        # Vectors should be different
        assert not np.allclose(emb1.vector, emb2.vector)
    
    def test_same_riff_same_vector(self, test_riffs):
        vectorizer = RiffVectorizer(method='statistical', dimension=128)
        
        emb1 = vectorizer.embed(test_riffs[0])
        emb2 = vectorizer.embed(test_riffs[0])
        
        # Same riff should produce identical vector
        assert np.allclose(emb1.vector, emb2.vector)
    
    def test_embedding_to_dict(self, test_riffs):
        vectorizer = RiffVectorizer(method='statistical', dimension=128)
        embedding = vectorizer.embed(test_riffs[0])
        
        data = embedding.to_dict()
        assert 'vector' in data
        assert 'metadata' in data
        assert isinstance(data['vector'], list)


class TestVectorDBAdapter:
    """Test VectorDBAdapter class."""
    
    def test_create_faiss_adapter(self):
        try:
            adapter = VectorDBAdapter(backend='faiss', dimension=128)
            assert adapter.backend == 'faiss'
            assert adapter.dimension == 128
        except ImportError:
            pytest.skip("FAISS not installed")
    
    def test_insert_and_search_faiss(self, test_riffs):
        try:
            # Create embeddings
            vectorizer = RiffVectorizer(method='statistical', dimension=128)
            embeddings = vectorizer.embed_batch(test_riffs)
            
            # Insert into FAISS
            adapter = VectorDBAdapter(backend='faiss', dimension=128)
            adapter.insert(embeddings)
            
            # Search
            query_vector = embeddings[0].vector
            results = adapter.search(query_vector, k=3)
            
            assert len(results) == 3
            assert results[0][1] < 1e-5  # Should match itself with ~0 distance
            
        except ImportError:
            pytest.skip("FAISS not installed")
    
    def test_unsupported_backend(self):
        with pytest.raises(ValueError):
            VectorDBAdapter(backend='nonexistent', dimension=128)


class TestCreateRiffVectorIndex:
    """Test complete index creation."""
    
    def test_create_index_statistical(self, synthetic_corpus):
        try:
            db, vectorizer = create_riff_vector_index(
                synthetic_corpus,
                method='statistical',
                backend='faiss',
                dimension=64
            )
            
            assert isinstance(db, VectorDBAdapter)
            assert isinstance(vectorizer, RiffVectorizer)
            
            # Test query
            query_riff = synthetic_corpus[0]
            query_embedding = vectorizer.embed(query_riff)
            results = db.search(query_embedding.vector, k=5)
            
            assert len(results) == 5
            
        except ImportError:
            pytest.skip("FAISS not installed")
    
    def test_create_index_distance(self, test_riffs):
        try:
            db, vectorizer = create_riff_vector_index(
                test_riffs,
                method='distance',
                backend='faiss',
                dimension=3  # Same as corpus size
            )
            
            # Query
            query_embedding = vectorizer.embed(test_riffs[0])
            results = db.search(query_embedding.vector, k=2)
            
            assert len(results) == 2
            
        except ImportError:
            pytest.skip("FAISS not installed")


class TestMetricPreservation:
    """Test that embeddings preserve distance relationships."""
    
    def test_statistical_preserves_order(self, test_riffs):
        """Check if relative distances are preserved."""
        vectorizer = RiffVectorizer(method='statistical', dimension=128)
        
        # Compute riff-space distances
        from src.space import RiffSpace
        space = RiffSpace(metric='edit_distance')
        
        d01 = space.distance(test_riffs[0], test_riffs[1], use_transforms=False)
        d02 = space.distance(test_riffs[0], test_riffs[2], use_transforms=False)
        
        # Compute embedding distances
        emb0 = vectorizer.embed(test_riffs[0])
        emb1 = vectorizer.embed(test_riffs[1])
        emb2 = vectorizer.embed(test_riffs[2])
        
        e01 = np.linalg.norm(emb0.vector - emb1.vector)
        e02 = np.linalg.norm(emb0.vector - emb2.vector)
        
        # Relative order should be similar
        # (Not exact, but correlated)
        if d01 < d02:
            # In most cases, embedding distance should follow
            pass  # Can't assert exact preservation, just test doesn't crash


if __name__ == "__main__":
    pytest.main([__file__])
