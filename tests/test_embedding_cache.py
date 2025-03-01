"""Tests for the embedding cache."""

import os
import torch
import numpy as np
import pytest
from transformers import AutoTokenizer

from scone.tokenization.n_gram_extractor import NGramExtractor
from scone.inference.embedding_cache import EmbeddingCache


@pytest.fixture
def tokenizer():
    """Create a tokenizer for testing."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@pytest.fixture
def n_gram_extractor(tokenizer):
    """Create an n-gram extractor for testing."""
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "A journey of a thousand miles begins with a single step.",
        "To be or not to be, that is the question.",
    ]
    
    # Tokenize texts
    tokenized_texts = [tokenizer.encode(text) for text in texts]
    
    # Create n-gram extractor
    n_gram_extractor = NGramExtractor(max_n=3)
    n_gram_extractor.fit(tokenized_texts, min_freq=1, max_f_grams=100)
    
    return n_gram_extractor


@pytest.fixture
def embeddings(n_gram_extractor):
    """Create dummy embeddings for testing."""
    embedding_dim = 768
    embeddings = {}
    
    for i, f_gram in enumerate(n_gram_extractor.f_grams):
        embeddings[i] = torch.randn(embedding_dim)
    
    return embeddings


@pytest.fixture
def embedding_cache(n_gram_extractor, tmp_path):
    """Create an embedding cache for testing."""
    embedding_dim = 768
    cache_dir = tmp_path / "cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    return EmbeddingCache(
        n_gram_extractor=n_gram_extractor,
        embedding_dim=embedding_dim,
        cache_dir=cache_dir,
    )


def test_embedding_cache_init(embedding_cache, n_gram_extractor):
    """Test embedding cache initialization."""
    assert embedding_cache.n_gram_extractor is n_gram_extractor
    assert embedding_cache.embedding_dim == 768
    assert os.path.exists(embedding_cache.cache_dir)
    assert embedding_cache.embeddings == {}
    assert embedding_cache.memory_mapped_embeddings is None


def test_embedding_cache_cache_embeddings(embedding_cache, embeddings):
    """Test embedding cache cache_embeddings method."""
    # Cache embeddings
    embedding_cache.cache_embeddings(embeddings)
    
    # Check if embeddings are cached
    assert len(embedding_cache.embeddings) == len(embeddings)
    
    # Check if embeddings are correct
    for f_gram_id, embedding in embeddings.items():
        assert f_gram_id in embedding_cache.embeddings
        assert torch.allclose(embedding_cache.embeddings[f_gram_id], embedding)


def test_embedding_cache_get_embeddings(embedding_cache, embeddings):
    """Test embedding cache get_embeddings method."""
    # Cache embeddings
    embedding_cache.cache_embeddings(embeddings)
    
    # Get embeddings for a subset of f-gram IDs
    f_gram_ids = list(embeddings.keys())[:5]
    retrieved_embeddings = embedding_cache.get_embeddings(f_gram_ids)
    
    # Check if retrieved embeddings are correct
    assert len(retrieved_embeddings) == len(f_gram_ids)
    for i, f_gram_id in enumerate(f_gram_ids):
        assert torch.allclose(retrieved_embeddings[i], embeddings[f_gram_id])


def test_embedding_cache_get_token_embeddings(embedding_cache, embeddings, n_gram_extractor):
    """Test embedding cache get_token_embeddings method."""
    # Cache embeddings
    embedding_cache.cache_embeddings(embeddings)
    
    # Get token embeddings for a token sequence
    token_ids = [1, 2, 3, 4, 5]  # Dummy token IDs
    token_embeddings = embedding_cache.get_token_embeddings(token_ids)
    
    # Check if token embeddings are returned
    assert isinstance(token_embeddings, dict)
    
    # The actual content depends on the n-gram extractor's f-grams
    # and how they match with the token IDs, so we can't check specific values


def test_embedding_cache_save_load(embedding_cache, embeddings, tmp_path):
    """Test embedding cache save and load."""
    # Cache embeddings
    embedding_cache.cache_embeddings(embeddings)
    
    # Save cache
    cache_path = tmp_path / "embeddings.cache"
    embedding_cache.save(cache_path)
    
    # Check if cache file exists
    assert os.path.exists(cache_path)
    
    # Load cache
    loaded_cache = EmbeddingCache.load(
        cache_path,
        n_gram_extractor=embedding_cache.n_gram_extractor,
        cache_dir=embedding_cache.cache_dir,
    )
    
    # Check if loaded cache has the same embeddings
    assert len(loaded_cache.embeddings) == len(embedding_cache.embeddings)
    for f_gram_id, embedding in embedding_cache.embeddings.items():
        assert f_gram_id in loaded_cache.embeddings
        assert torch.allclose(loaded_cache.embeddings[f_gram_id], embedding)


def test_embedding_cache_memory_map(embedding_cache, embeddings, tmp_path):
    """Test embedding cache with memory mapping."""
    # Create a memory-mapped cache
    memory_mapped_cache = EmbeddingCache(
        n_gram_extractor=embedding_cache.n_gram_extractor,
        embedding_dim=embedding_cache.embedding_dim,
        cache_dir=embedding_cache.cache_dir,
        use_memory_map=True,
    )
    
    # Cache embeddings
    memory_mapped_cache.cache_embeddings(embeddings)
    
    # Check if memory-mapped array is created
    assert memory_mapped_cache.memory_mapped_embeddings is not None
    assert isinstance(memory_mapped_cache.memory_mapped_embeddings, np.ndarray)
    assert memory_mapped_cache.memory_mapped_embeddings.shape == (len(embeddings), embedding_cache.embedding_dim)
    
    # Save cache
    cache_path = tmp_path / "memory_mapped_embeddings.cache"
    memory_mapped_cache.save(cache_path)
    
    # Check if cache file exists
    assert os.path.exists(cache_path)
    
    # Load cache
    loaded_cache = EmbeddingCache.load(
        cache_path,
        n_gram_extractor=memory_mapped_cache.n_gram_extractor,
        cache_dir=memory_mapped_cache.cache_dir,
    )
    
    # Check if loaded cache has memory mapping
    assert loaded_cache.use_memory_map is True
    assert loaded_cache.memory_mapped_embeddings is not None
    assert isinstance(loaded_cache.memory_mapped_embeddings, np.ndarray)
    assert loaded_cache.memory_mapped_embeddings.shape == (len(embeddings), memory_mapped_cache.embedding_dim)
    
    # Get embeddings
    f_gram_ids = list(embeddings.keys())[:5]
    retrieved_embeddings = loaded_cache.get_embeddings(f_gram_ids)
    
    # Check if retrieved embeddings are correct
    assert len(retrieved_embeddings) == len(f_gram_ids)
    for i, f_gram_id in enumerate(f_gram_ids):
        # Convert to numpy for comparison since memory-mapped arrays are numpy arrays
        original = embeddings[f_gram_id].numpy()
        retrieved = retrieved_embeddings[i].numpy()
        assert np.allclose(original, retrieved) 