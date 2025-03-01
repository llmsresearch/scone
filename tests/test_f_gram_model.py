"""Tests for the f-gram model."""

import torch
import pytest
from transformers import AutoTokenizer

from scone.models.f_gram_model import FGramConfig, FGramModel
from scone.tokenization.n_gram_extractor import NGramExtractor
from scone.tokenization.f_gram_tokenizer import FGramTokenizer


@pytest.fixture
def tokenizer():
    """Create a tokenizer for testing."""
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
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
def f_gram_tokenizer(tokenizer, n_gram_extractor):
    """Create an f-gram tokenizer for testing."""
    return FGramTokenizer(n_gram_extractor=n_gram_extractor, tokenizer=tokenizer)


@pytest.fixture
def f_gram_model():
    """Create an f-gram model for testing."""
    config = FGramConfig(
        vocab_size=30522,  # bert-base-uncased vocab size
        hidden_size=768,
        num_hidden_layers=2,
        num_attention_heads=12,
        intermediate_size=3072,
    )
    
    return FGramModel(config)


def test_f_gram_model_init(f_gram_model):
    """Test f-gram model initialization."""
    assert isinstance(f_gram_model, FGramModel)
    assert f_gram_model.config.vocab_size == 30522
    assert f_gram_model.config.hidden_size == 768
    assert f_gram_model.config.num_hidden_layers == 2
    assert f_gram_model.config.num_attention_heads == 12


def test_f_gram_model_forward(f_gram_model, tokenizer, f_gram_tokenizer):
    """Test f-gram model forward pass."""
    # Create input data
    text = "The quick brown fox jumps over the lazy dog."
    input_ids = tokenizer.encode(text, return_tensors="pt")
    
    # Get f-gram IDs
    f_gram_ids = f_gram_tokenizer.encode(text, return_tensors="pt")["f_gram_ids"]
    
    # Forward pass
    outputs = f_gram_model(
        input_ids=input_ids,
        f_gram_ids=f_gram_ids,
    )
    
    # Check outputs
    assert "last_hidden_state" in outputs
    assert "f_gram_embeddings" in outputs
    
    # Check shapes
    assert outputs["last_hidden_state"].shape == (1, input_ids.shape[1], f_gram_model.config.hidden_size)
    assert outputs["f_gram_embeddings"].shape == (1, f_gram_ids.shape[1], f_gram_model.config.hidden_size)


def test_f_gram_model_get_f_gram_embeddings(f_gram_model, tokenizer, f_gram_tokenizer):
    """Test f-gram model get_f_gram_embeddings method."""
    # Create input data
    text = "The quick brown fox jumps over the lazy dog."
    input_ids = tokenizer.encode(text, return_tensors="pt")
    
    # Get f-gram IDs
    f_gram_ids = f_gram_tokenizer.encode(text, return_tensors="pt")["f_gram_ids"]
    
    # Get f-gram embeddings
    f_gram_embeddings = f_gram_model.get_f_gram_embeddings(
        input_ids=input_ids,
        f_gram_ids=f_gram_ids,
    )
    
    # Check shape
    assert f_gram_embeddings.shape == (1, f_gram_ids.shape[1], f_gram_model.config.hidden_size)


def test_f_gram_model_save_load(f_gram_model, tmp_path):
    """Test f-gram model save and load."""
    # Save model
    model_path = tmp_path / "f_gram_model"
    f_gram_model.save_pretrained(model_path)
    
    # Load model
    loaded_model = FGramModel.from_pretrained(model_path)
    
    # Check if loaded model has the same config
    assert loaded_model.config.vocab_size == f_gram_model.config.vocab_size
    assert loaded_model.config.hidden_size == f_gram_model.config.hidden_size
    assert loaded_model.config.num_hidden_layers == f_gram_model.config.num_hidden_layers
    assert loaded_model.config.num_attention_heads == f_gram_model.config.num_attention_heads
    
    # Check if models have the same parameters
    for p1, p2 in zip(f_gram_model.parameters(), loaded_model.parameters()):
        assert torch.allclose(p1, p2) 