#!/usr/bin/env python
"""Text generation script for SCONE models."""

import argparse
import os
from typing import Dict, List, Optional, Tuple, Union

import torch
from transformers import AutoTokenizer

from scone.inference.engine import SconeInferenceEngine
from scone.tokenization.f_gram_tokenizer import FGramTokenizer


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Generate text with SCONE model")
    
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
    parser.add_argument(
        "--embedding_cache_path",
        type=str,
        default=None,
        help="Path to embedding cache file (optional)",
    )
    
    # Generation arguments
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for generation",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=100,
        help="Maximum length of generated text",
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=10,
        help="Minimum length of generated text",
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Whether to use sampling for generation",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help="Number of beams for beam search",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for sampling",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k value for sampling",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top-p value for sampling",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="Repetition penalty",
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help="Number of sequences to return",
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
    
    # Load inference engine
    print("Loading inference engine...")
    inference_engine = SconeInferenceEngine.from_pretrained(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        f_gram_tokenizer_path=args.f_gram_tokenizer_path,
        embedding_cache_path=args.embedding_cache_path,
    )
    
    # Generate text
    print(f"Generating text from prompt: {args.prompt}")
    generated_texts = inference_engine.generate(
        text=args.prompt,
        max_length=args.max_length,
        min_length=args.min_length,
        do_sample=args.do_sample,
        num_beams=args.num_beams,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        num_return_sequences=args.num_return_sequences,
    )
    
    # Print generated text
    print("\nGenerated text:")
    for i, text in enumerate(generated_texts):
        print(f"\nSequence {i+1}:")
        print(text)


if __name__ == "__main__":
    main() 