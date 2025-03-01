"""Tests for the SCONE language model."""

import torch
import pytest
from transformers import AutoTokenizer

from scone.models.f_gram_model import FGramConfig, FGramModel
from scone.models.language_model import SconeConfig, SconeLanguageModel
from scone.tokenization.n_gram_extractor import NGramExtractor
from scone.tokenization.f_gram_tokenizer import FGramTokenizer


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
def f_gram_tokenizer(tokenizer, n_gram_extractor):
    """Create an f-gram tokenizer for testing."""
    return FGramTokenizer(n_gram_extractor=n_gram_extractor, tokenizer=tokenizer)


@pytest.fixture
def f_gram_config():
    """Create an f-gram model config for testing."""
    return FGramConfig(
        vocab_size=50257,  # gpt2 vocab size
        hidden_size=768,
        num_hidden_layers=2,
        num_attention_heads=12,
        intermediate_size=3072,
    )


@pytest.fixture
def f_gram_model(f_gram_config):
    """Create an f-gram model for testing."""
    return FGramModel(f_gram_config)


@pytest.fixture
def scone_config(f_gram_config):
    """Create a SCONE model config for testing."""
    return SconeConfig(
        vocab_size=50257,  # gpt2 vocab size
        hidden_size=768,
        num_hidden_layers=2,
        num_attention_heads=12,
        intermediate_size=3072,
        f_gram_model_config=f_gram_config,
        use_f_gram_embeddings=True,
    )


@pytest.fixture
def scone_model(scone_config):
    """Create a SCONE language model for testing."""
    return SconeLanguageModel(scone_config)


def test_scone_model_init(scone_model):
    """Test SCONE model initialization."""
    assert isinstance(scone_model, SconeLanguageModel)
    assert scone_model.config.vocab_size == 50257
    assert scone_model.config.hidden_size == 768
    assert scone_model.config.num_hidden_layers == 2
    assert scone_model.config.num_attention_heads == 12
    assert scone_model.config.use_f_gram_embeddings is True


def test_scone_model_forward(scone_model, tokenizer, f_gram_tokenizer):
    """Test SCONE model forward pass."""
    # Create input data
    text = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(text, return_tensors="pt")
    
    # Get f-gram IDs
    f_gram_inputs = f_gram_tokenizer.encode(text, return_tensors="pt")
    
    # Combine inputs
    combined_inputs = {**inputs, **f_gram_inputs}
    
    # Forward pass
    outputs = scone_model(**combined_inputs)
    
    # Check outputs
    assert "loss" in outputs
    assert "logits" in outputs
    assert "hidden_states" in outputs
    
    # Check shapes
    assert outputs["logits"].shape == (1, inputs["input_ids"].shape[1], scone_model.config.vocab_size)


def test_scone_model_generate(scone_model, tokenizer):
    """Test SCONE model generate method."""
    # Create input data
    text = "The quick brown fox"
    input_ids = tokenizer.encode(text, return_tensors="pt")
    
    # Generate text
    output_ids = scone_model.generate(
        input_ids=input_ids,
        max_length=20,
        do_sample=False,
    )
    
    # Check output
    assert output_ids.shape[1] > input_ids.shape[1]
    assert output_ids.shape[1] <= 20
    
    # Decode output
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    assert text in output_text


def test_scone_model_with_f_gram_embeddings(scone_model, tokenizer, f_gram_tokenizer):
    """Test SCONE model with f-gram embeddings."""
    # Create input data
    text = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(text, return_tensors="pt")
    
    # Get f-gram IDs and embeddings
    f_gram_inputs = f_gram_tokenizer.encode(text, return_tensors="pt")
    f_gram_ids = f_gram_inputs["f_gram_ids"]
    
    # Create dummy f-gram embeddings
    f_gram_embeddings = torch.randn(1, f_gram_ids.shape[1], scone_model.config.hidden_size)
    
    # Forward pass with f-gram embeddings
    outputs = scone_model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        f_gram_ids=f_gram_ids,
        f_gram_embeddings=f_gram_embeddings,
    )
    
    # Check outputs
    assert "loss" in outputs
    assert "logits" in outputs
    assert "hidden_states" in outputs
    
    # Check shapes
    assert outputs["logits"].shape == (1, inputs["input_ids"].shape[1], scone_model.config.vocab_size)


def test_scone_model_save_load(scone_model, tmp_path):
    """Test SCONE model save and load."""
    # Save model
    model_path = tmp_path / "scone_model"
    scone_model.save_pretrained(model_path)
    
    # Load model
    loaded_model = SconeLanguageModel.from_pretrained(model_path)
    
    # Check if loaded model has the same config
    assert loaded_model.config.vocab_size == scone_model.config.vocab_size
    assert loaded_model.config.hidden_size == scone_model.config.hidden_size
    assert loaded_model.config.num_hidden_layers == scone_model.config.num_hidden_layers
    assert loaded_model.config.num_attention_heads == scone_model.config.num_attention_heads
    assert loaded_model.config.use_f_gram_embeddings == scone_model.config.use_f_gram_embeddings
    
    # Check if models have the same parameters
    for p1, p2 in zip(scone_model.parameters(), loaded_model.parameters()):
        assert torch.allclose(p1, p2) 