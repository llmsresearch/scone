#!/usr/bin/env python
"""Simple example of using SCONE for text generation."""

import os
import torch
from transformers import AutoTokenizer

from scone.models.f_gram_model import FGramConfig, FGramModel
from scone.models.language_model import SconeConfig, SconeLanguageModel
from scone.tokenization.f_gram_tokenizer import FGramTokenizer
from scone.tokenization.n_gram_extractor import NGramExtractor
from scone.inference.engine import SconeInferenceEngine
from scone.inference.embedding_cache import EmbeddingCache
from scone.data.preprocessing import extract_f_grams, precompute_f_gram_embeddings


def main():
    """Run a simple example of SCONE text generation."""
    print("SCONE Simple Example")
    print("===================")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Sample texts for demonstration
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "A journey of a thousand miles begins with a single step.",
        "To be or not to be, that is the question.",
        "All that glitters is not gold.",
        "The early bird catches the worm.",
        "Actions speak louder than words.",
        "You can't teach an old dog new tricks.",
        "The pen is mightier than the sword.",
        "Where there's a will, there's a way.",
        "Knowledge is power.",
    ]
    
    print("\nStep 1: Load tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    print("\nStep 2: Extract f-grams")
    n_gram_extractor = extract_f_grams(
        texts=texts,
        tokenizer=tokenizer,
        max_n=3,
        min_freq=1,  # Low threshold for this small example
        max_f_grams=1000,
        verbose=True,
    )
    
    print(f"\nExtracted {len(n_gram_extractor.f_grams)} f-grams")
    print("Sample f-grams:")
    for i, f_gram in enumerate(list(n_gram_extractor.f_grams)[:5]):
        print(f"  {i+1}. {tokenizer.decode(f_gram)}")
    
    print("\nStep 3: Create f-gram tokenizer")
    f_gram_tokenizer = FGramTokenizer(
        n_gram_extractor=n_gram_extractor,
        tokenizer=tokenizer,
    )
    
    print("\nStep 4: Create f-gram model")
    f_gram_config = FGramConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=768,
        num_hidden_layers=6,
        num_attention_heads=12,
    )
    
    f_gram_model = FGramModel(f_gram_config)
    f_gram_model.to(device)
    
    print("\nStep 5: Create SCONE language model")
    scone_config = SconeConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=768,
        num_hidden_layers=6,
        num_attention_heads=12,
        f_gram_model_config=f_gram_config,
        use_f_gram_embeddings=True,
    )
    
    model = SconeLanguageModel(scone_config)
    model.to(device)
    
    print("\nStep 6: Precompute f-gram embeddings")
    embeddings = precompute_f_gram_embeddings(
        n_gram_extractor=n_gram_extractor,
        f_gram_model=f_gram_model,
        tokenizer=tokenizer,
        batch_size=4,
        device=device,
        verbose=True,
    )
    
    print(f"\nPrecomputed embeddings for {len(embeddings)} f-grams")
    
    print("\nStep 7: Create embedding cache")
    cache_dir = "./cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    embedding_cache = EmbeddingCache(
        n_gram_extractor=n_gram_extractor,
        embedding_dim=f_gram_model.config.hidden_size,
        cache_dir=cache_dir,
    )
    
    embedding_cache.cache_embeddings(embeddings)
    
    print("\nStep 8: Create inference engine")
    inference_engine = SconeInferenceEngine(
        model=model,
        tokenizer=tokenizer,
        f_gram_tokenizer=f_gram_tokenizer,
        embedding_cache=embedding_cache,
        device=device,
    )
    
    print("\nStep 9: Generate text")
    prompt = "The quick brown fox"
    print(f"Prompt: {prompt}")
    
    generated_texts = inference_engine.generate(
        text=prompt,
        max_length=30,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        num_return_sequences=3,
    )
    
    print("\nGenerated texts:")
    for i, text in enumerate(generated_texts):
        print(f"\nSequence {i+1}:")
        print(text)
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main() 