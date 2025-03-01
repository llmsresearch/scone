#!/usr/bin/env python
"""Script to precompute f-gram embeddings for SCONE."""

import argparse
import os
from typing import Dict, List, Optional, Tuple, Union

import torch
from transformers import AutoTokenizer

from scone.data.preprocessing import precompute_f_gram_embeddings
from scone.inference.embedding_cache import EmbeddingCache
from scone.models.f_gram_model import FGramModel
from scone.models.language_model import SconeLanguageModel
from scone.tokenization.f_gram_tokenizer import FGramTokenizer


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Precompute f-gram embeddings for SCONE")
    
    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model directory",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to tokenizer directory (defaults to model_path/tokenizer)",
    )
    parser.add_argument(
        "--f_gram_tokenizer_path",
        type=str,
        default=None,
        help="Path to f-gram tokenizer directory (defaults to model_path/f_gram_tokenizer)",
    )
    
    # Output arguments
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save embedding cache",
    )
    parser.add_argument(
        "--use_memory_map",
        action="store_true",
        help="Whether to use memory mapping for embedding cache",
    )
    
    # Computation arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for embedding computation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for computation (defaults to cuda if available, otherwise cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    return parser.parse_args()


def main() -> None:
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Set default paths
    if args.tokenizer_path is None:
        args.tokenizer_path = os.path.join(args.model_path, "tokenizer")
    
    if args.f_gram_tokenizer_path is None:
        args.f_gram_tokenizer_path = os.path.join(args.model_path, "f_gram_tokenizer")
    
    # Set device
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    
    # Load tokenizers
    print("Loading tokenizers...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    f_gram_tokenizer = FGramTokenizer.from_pretrained(args.f_gram_tokenizer_path)
    
    # Load model
    print("Loading model...")
    model = SconeLanguageModel.from_pretrained(args.model_path)
    model.to(device)
    model.eval()
    
    # Extract f-gram model
    f_gram_model = model.f_gram_model
    
    # Precompute embeddings
    print("Precomputing f-gram embeddings...")
    embeddings = precompute_f_gram_embeddings(
        n_gram_extractor=f_gram_tokenizer.n_gram_extractor,
        f_gram_model=f_gram_model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        device=device,
        verbose=True,
    )
    
    # Create embedding cache
    print("Creating embedding cache...")
    embedding_cache = EmbeddingCache(
        n_gram_extractor=f_gram_tokenizer.n_gram_extractor,
        embedding_dim=f_gram_model.config.hidden_size,
        cache_dir=os.path.dirname(args.output_path),
        use_memory_map=args.use_memory_map,
    )
    
    # Cache embeddings
    print("Caching embeddings...")
    embedding_cache.cache_embeddings(embeddings)
    
    # Save embedding cache
    print(f"Saving embedding cache to {args.output_path}...")
    embedding_cache.save(args.output_path)
    
    print("Done!")


if __name__ == "__main__":
    main() 