#!/usr/bin/env python
"""Evaluation script for SCONE models."""

import argparse
import os
from typing import Dict, List, Optional, Tuple, Union

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from scone.data.dataset import SconeDataset
from scone.inference.engine import SconeInferenceEngine
from scone.tokenization.f_gram_tokenizer import FGramTokenizer


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate SCONE model")
    
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
    
    # Data arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="wikitext",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default="wikitext-103-v1",
        help="HuggingFace dataset config name",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Column name for text data",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate on",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size",
    )
    
    # Evaluation arguments
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for evaluation (defaults to cuda if available, otherwise cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    return parser.parse_args()


def evaluate_perplexity(
    inference_engine: SconeInferenceEngine,
    dataset: SconeDataset,
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate perplexity on a dataset.
    
    Args:
        inference_engine: SCONE inference engine.
        dataset: Dataset to evaluate on.
        batch_size: Batch size.
        device: Device to use for evaluation.
    
    Returns:
        Dictionary of evaluation metrics.
    """
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    
    # Set model to evaluation mode
    inference_engine.model.eval()
    
    # Evaluate
    total_loss = 0.0
    total_tokens = 0
    
    for batch in dataloader:
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = inference_engine.model(**batch)
            loss = outputs["loss"]
        
        # Update metrics
        total_loss += loss.item() * batch["input_ids"].size(0)
        total_tokens += batch["input_ids"].size(0)
    
    # Calculate metrics
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return {
        "loss": avg_loss,
        "perplexity": perplexity,
    }


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
    
    # Load inference engine
    print("Loading inference engine...")
    inference_engine = SconeInferenceEngine.from_pretrained(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        f_gram_tokenizer_path=args.f_gram_tokenizer_path,
        embedding_cache_path=args.embedding_cache_path,
        device=device,
    )
    
    # Load dataset
    print(f"Loading dataset {args.dataset_name}/{args.dataset_config_name}...")
    dataset = load_dataset(args.dataset_name, args.dataset_config_name)
    
    # Create evaluation dataset
    print(f"Creating evaluation dataset from {args.split} split...")
    eval_dataset = SconeDataset.from_huggingface_dataset(
        dataset=dataset[args.split],
        text_column=args.text_column,
        tokenizer=inference_engine.tokenizer,
        f_gram_tokenizer=inference_engine.f_gram_tokenizer,
        max_length=args.max_length,
        task="causal_lm",
    )
    
    # Evaluate perplexity
    print("Evaluating perplexity...")
    metrics = evaluate_perplexity(
        inference_engine=inference_engine,
        dataset=eval_dataset,
        batch_size=args.batch_size,
        device=device,
    )
    
    # Print metrics
    print("\nEvaluation metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    main() 