#!/usr/bin/env python
"""Benchmarking script for SCONE models.

This script benchmarks SCONE models with different configurations against baseline models,
measuring perplexity, inference speed, and memory usage as described in the paper
"Scaling Embedding Layers in Language Models" (arXiv:2502.01637).
"""

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import psutil
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from scone.data.dataset import SconeDataset
from scone.inference.engine import SconeInferenceEngine
from scone.tokenization.f_gram_tokenizer import FGramTokenizer
from scone.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking."""
    # Model configurations to benchmark
    model_configs: List[Dict[str, Any]]
    
    # Baseline model configurations
    baseline_configs: List[Dict[str, Any]]
    
    # Dataset configurations
    dataset_configs: List[Dict[str, Any]]
    
    # Benchmark settings
    batch_sizes: List[int]
    sequence_lengths: List[int]
    num_runs: int = 5
    warmup_runs: int = 2
    
    # Output settings
    output_dir: str = "./benchmark_results"
    save_results: bool = True


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    # Model information
    model_name: str
    model_type: str  # "scone" or "baseline"
    model_config: Dict[str, Any]
    
    # Dataset information
    dataset_name: str
    dataset_config: Dict[str, Any]
    
    # Performance metrics
    perplexity: float
    inference_time_ms: float
    memory_usage_mb: float
    parameters_count: int
    flops: Optional[int] = None
    
    # Additional information
    batch_size: int
    sequence_length: int
    device: str


def count_parameters(model: PreTrainedModel) -> int:
    """Count the number of parameters in a model.
    
    Args:
        model: The model to count parameters for.
        
    Returns:
        Number of parameters.
    """
    return sum(p.numel() for p in model.parameters())


def estimate_flops(
    model: PreTrainedModel,
    sequence_length: int,
    is_scone: bool = False,
    f_gram_model_size: Optional[int] = None,
) -> int:
    """Estimate the number of FLOPs for a forward pass.
    
    This is a rough estimate based on the model architecture.
    For SCONE models, we account for the reduced FLOPs during inference
    as described in the paper.
    
    Args:
        model: The model to estimate FLOPs for.
        sequence_length: Sequence length for the forward pass.
        is_scone: Whether the model is a SCONE model.
        f_gram_model_size: Size of the f-gram model (if applicable).
        
    Returns:
        Estimated number of FLOPs.
    """
    # This is a simplified estimate - in a real implementation,
    # you would need to account for the specific model architecture
    if hasattr(model.config, "n_embd"):
        # GPT-2 style config
        hidden_size = model.config.n_embd
        num_layers = model.config.n_layer
        vocab_size = model.config.vocab_size
    else:
        # Generic config
        hidden_size = getattr(model.config, "hidden_size", 768)
        num_layers = getattr(model.config, "num_hidden_layers", 12)
        vocab_size = getattr(model.config, "vocab_size", 50257)
    
    # Estimate FLOPs for attention and feed-forward layers
    # This is a simplified calculation
    flops_per_token = 2 * num_layers * (
        # Self-attention
        4 * hidden_size * hidden_size +
        # Feed-forward
        8 * hidden_size * hidden_size
    )
    
    # Add embedding and output layer FLOPs
    if is_scone:
        # For SCONE, we don't include the embedding layer FLOPs
        # since they're precomputed
        flops_per_token += hidden_size * vocab_size
    else:
        # For baseline, include both embedding and output layer
        flops_per_token += 2 * hidden_size * vocab_size
    
    # Total FLOPs for the sequence
    total_flops = flops_per_token * sequence_length
    
    return total_flops


def measure_inference_speed(
    model: Union[PreTrainedModel, SconeInferenceEngine],
    input_ids: torch.Tensor,
    f_gram_embeddings: Optional[torch.Tensor] = None,
    num_runs: int = 5,
    warmup_runs: int = 2,
) -> float:
    """Measure inference speed.
    
    Args:
        model: Model or inference engine to benchmark.
        input_ids: Input token IDs.
        f_gram_embeddings: F-gram embeddings (for SCONE models).
        num_runs: Number of runs to average over.
        warmup_runs: Number of warmup runs.
        
    Returns:
        Average inference time in milliseconds.
    """
    # Warmup runs
    for _ in range(warmup_runs):
        with torch.no_grad():
            if isinstance(model, SconeInferenceEngine):
                model.model(
                    input_ids=input_ids,
                    f_gram_embeddings=f_gram_embeddings,
                )
            else:
                model(input_ids=input_ids)
    
    # Timed runs
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(num_runs):
        with torch.no_grad():
            if isinstance(model, SconeInferenceEngine):
                model.model(
                    input_ids=input_ids,
                    f_gram_embeddings=f_gram_embeddings,
                )
            else:
                model(input_ids=input_ids)
        
        torch.cuda.synchronize()
    
    end_time = time.time()
    
    # Calculate average time in milliseconds
    avg_time_ms = (end_time - start_time) * 1000 / num_runs
    
    return avg_time_ms


def measure_memory_usage(
    model: Union[PreTrainedModel, SconeInferenceEngine],
    input_ids: torch.Tensor,
    f_gram_embeddings: Optional[torch.Tensor] = None,
) -> float:
    """Measure memory usage.
    
    Args:
        model: Model or inference engine to benchmark.
        input_ids: Input token IDs.
        f_gram_embeddings: F-gram embeddings (for SCONE models).
        
    Returns:
        Memory usage in megabytes.
    """
    # Clear cache
    torch.cuda.empty_cache()
    
    # Record memory before
    torch.cuda.synchronize()
    memory_before = torch.cuda.memory_allocated()
    
    # Forward pass
    with torch.no_grad():
        if isinstance(model, SconeInferenceEngine):
            model.model(
                input_ids=input_ids,
                f_gram_embeddings=f_gram_embeddings,
            )
        else:
            model(input_ids=input_ids)
    
    # Record memory after
    torch.cuda.synchronize()
    memory_after = torch.cuda.memory_allocated()
    
    # Calculate memory usage in MB
    memory_usage_mb = (memory_after - memory_before) / (1024 * 1024)
    
    return memory_usage_mb


def evaluate_perplexity(
    model: Union[PreTrainedModel, SconeInferenceEngine],
    dataset: Union[SconeDataset, DataLoader],
    batch_size: int,
    device: torch.device,
    is_scone: bool = False,
) -> float:
    """Evaluate perplexity on a dataset.
    
    Args:
        model: Model or inference engine to evaluate.
        dataset: Dataset to evaluate on.
        batch_size: Batch size.
        device: Device to use for evaluation.
        is_scone: Whether the model is a SCONE model.
    
    Returns:
        Perplexity.
    """
    # Create dataloader if needed
    if isinstance(dataset, SconeDataset):
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
        )
    else:
        dataloader = dataset
    
    # Set model to evaluation mode
    if isinstance(model, SconeInferenceEngine):
        model.model.eval()
    else:
        model.eval()
    
    # Evaluate
    total_loss = 0.0
    total_tokens = 0
    
    for batch in dataloader:
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Forward pass
        with torch.no_grad():
            if isinstance(model, SconeInferenceEngine):
                outputs = model.model(**batch)
            else:
                outputs = model(**{k: v for k, v in batch.items() if k in ["input_ids", "attention_mask", "labels"]})
            
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        
        # Update metrics
        total_loss += loss.item() * batch["input_ids"].size(0) * batch["input_ids"].size(1)
        total_tokens += batch["input_ids"].size(0) * batch["input_ids"].size(1)
    
    # Calculate perplexity
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return perplexity


def benchmark_model(
    model_config: Dict[str, Any],
    dataset_config: Dict[str, Any],
    batch_size: int,
    sequence_length: int,
    device: torch.device,
    num_runs: int = 5,
    warmup_runs: int = 2,
    is_baseline: bool = False,
) -> BenchmarkResult:
    """Benchmark a model configuration.
    
    Args:
        model_config: Model configuration.
        dataset_config: Dataset configuration.
        batch_size: Batch size.
        sequence_length: Sequence length.
        device: Device to use for benchmarking.
        num_runs: Number of runs to average over.
        warmup_runs: Number of warmup runs.
        is_baseline: Whether this is a baseline model.
        
    Returns:
        Benchmark results.
    """
    logger.info(f"Benchmarking {'baseline' if is_baseline else 'SCONE'} model: {model_config['name']}")
    logger.info(f"Dataset: {dataset_config['name']}")
    logger.info(f"Batch size: {batch_size}, Sequence length: {sequence_length}")
    
    # Load dataset
    dataset = load_dataset(
        dataset_config["dataset_name"],
        dataset_config["dataset_config_name"],
    )
    
    if is_baseline:
        # Load baseline model
        model = AutoModelForCausalLM.from_pretrained(model_config["model_name"])
        tokenizer = AutoTokenizer.from_pretrained(model_config["model_name"])
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create evaluation dataset
        def tokenize_function(examples):
            return tokenizer(
                examples[dataset_config["text_column"]],
                padding="max_length",
                truncation=True,
                max_length=sequence_length,
                return_tensors="pt",
            )
        
        tokenized_dataset = dataset[dataset_config["split"]].map(
            tokenize_function,
            batched=True,
            remove_columns=dataset[dataset_config["split"]].column_names,
        )
        
        tokenized_dataset = tokenized_dataset.map(
            lambda examples: {"labels": examples["input_ids"]},
            batched=True,
        )
        
        eval_dataloader = DataLoader(
            tokenized_dataset,
            batch_size=batch_size,
            shuffle=False,
        )
        
        # Move model to device
        model.to(device)
        
        # Prepare input for speed and memory measurements
        sample_input_ids = torch.randint(
            0, tokenizer.vocab_size, (batch_size, sequence_length), device=device
        )
        
        # Measure perplexity
        perplexity = evaluate_perplexity(
            model=model,
            dataset=eval_dataloader,
            batch_size=batch_size,
            device=device,
            is_scone=False,
        )
        
        # Measure inference speed
        inference_time_ms = measure_inference_speed(
            model=model,
            input_ids=sample_input_ids,
            num_runs=num_runs,
            warmup_runs=warmup_runs,
        )
        
        # Measure memory usage
        memory_usage_mb = measure_memory_usage(
            model=model,
            input_ids=sample_input_ids,
        )
        
        # Count parameters
        parameters_count = count_parameters(model)
        
        # Estimate FLOPs
        flops = estimate_flops(
            model=model,
            sequence_length=sequence_length,
            is_scone=False,
        )
    else:
        # Load SCONE model
        inference_engine = SconeInferenceEngine.from_pretrained(
            model_path=model_config["model_path"],
            tokenizer_path=model_config.get("tokenizer_path"),
            f_gram_tokenizer_path=model_config.get("f_gram_tokenizer_path"),
            embedding_cache_path=model_config.get("embedding_cache_path"),
            device=device,
        )
        
        # Create evaluation dataset
        eval_dataset = SconeDataset.from_huggingface_dataset(
            dataset=dataset[dataset_config["split"]],
            text_column=dataset_config["text_column"],
            tokenizer=inference_engine.tokenizer,
            f_gram_tokenizer=inference_engine.f_gram_tokenizer,
            max_length=sequence_length,
            task="causal_lm",
        )
        
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
        )
        
        # Prepare input for speed and memory measurements
        sample_batch = next(iter(eval_dataloader))
        sample_input_ids = sample_batch["input_ids"].to(device)
        sample_f_gram_embeddings = sample_batch["f_gram_embeddings"].to(device) if "f_gram_embeddings" in sample_batch else None
        
        # Measure perplexity
        perplexity = evaluate_perplexity(
            model=inference_engine,
            dataset=eval_dataloader,
            batch_size=batch_size,
            device=device,
            is_scone=True,
        )
        
        # Measure inference speed
        inference_time_ms = measure_inference_speed(
            model=inference_engine,
            input_ids=sample_input_ids,
            f_gram_embeddings=sample_f_gram_embeddings,
            num_runs=num_runs,
            warmup_runs=warmup_runs,
        )
        
        # Measure memory usage
        memory_usage_mb = measure_memory_usage(
            model=inference_engine,
            input_ids=sample_input_ids,
            f_gram_embeddings=sample_f_gram_embeddings,
        )
        
        # Count parameters
        parameters_count = count_parameters(inference_engine.model)
        
        # Estimate FLOPs
        flops = estimate_flops(
            model=inference_engine.model,
            sequence_length=sequence_length,
            is_scone=True,
            f_gram_model_size=model_config.get("f_gram_model_size"),
        )
    
    # Create benchmark result
    result = BenchmarkResult(
        model_name=model_config["name"],
        model_type="baseline" if is_baseline else "scone",
        model_config=model_config,
        dataset_name=dataset_config["name"],
        dataset_config=dataset_config,
        perplexity=perplexity,
        inference_time_ms=inference_time_ms,
        memory_usage_mb=memory_usage_mb,
        parameters_count=parameters_count,
        flops=flops,
        batch_size=batch_size,
        sequence_length=sequence_length,
        device=str(device),
    )
    
    logger.info(f"Results: perplexity={perplexity:.4f}, inference_time={inference_time_ms:.2f}ms, memory={memory_usage_mb:.2f}MB")
    
    return result


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Benchmark SCONE models")
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to benchmark configuration file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./benchmark_results",
        help="Directory to save benchmark results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for benchmarking (defaults to cuda if available, otherwise cpu)",
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
    np.random.seed(args.seed)
    
    # Set device
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    
    # Load benchmark configuration
    with open(args.config, "r") as f:
        config_dict = json.load(f)
    
    benchmark_config = BenchmarkConfig(**config_dict)
    benchmark_config.output_dir = args.output_dir
    
    # Create output directory
    os.makedirs(benchmark_config.output_dir, exist_ok=True)
    
    # Run benchmarks
    results = []
    
    # Benchmark SCONE models
    for model_config in benchmark_config.model_configs:
        for dataset_config in benchmark_config.dataset_configs:
            for batch_size in benchmark_config.batch_sizes:
                for sequence_length in benchmark_config.sequence_lengths:
                    result = benchmark_model(
                        model_config=model_config,
                        dataset_config=dataset_config,
                        batch_size=batch_size,
                        sequence_length=sequence_length,
                        device=device,
                        num_runs=benchmark_config.num_runs,
                        warmup_runs=benchmark_config.warmup_runs,
                        is_baseline=False,
                    )
                    results.append(result)
    
    # Benchmark baseline models
    for model_config in benchmark_config.baseline_configs:
        for dataset_config in benchmark_config.dataset_configs:
            for batch_size in benchmark_config.batch_sizes:
                for sequence_length in benchmark_config.sequence_lengths:
                    result = benchmark_model(
                        model_config=model_config,
                        dataset_config=dataset_config,
                        batch_size=batch_size,
                        sequence_length=sequence_length,
                        device=device,
                        num_runs=benchmark_config.num_runs,
                        warmup_runs=benchmark_config.warmup_runs,
                        is_baseline=True,
                    )
                    results.append(result)
    
    # Save results
    if benchmark_config.save_results:
        results_file = os.path.join(benchmark_config.output_dir, "benchmark_results.json")
        with open(results_file, "w") as f:
            json.dump([asdict(result) for result in results], f, indent=2)
        
        logger.info(f"Saved benchmark results to {results_file}")
    
    # Print summary
    logger.info("\nBenchmark Summary:")
    logger.info("=================")
    
    # Group results by model and dataset
    grouped_results = {}
    for result in results:
        key = (result.model_name, result.dataset_name)
        if key not in grouped_results:
            grouped_results[key] = []
        grouped_results[key].append(result)
    
    # Print summary for each model and dataset
    for (model_name, dataset_name), model_results in grouped_results.items():
        logger.info(f"\nModel: {model_name}, Dataset: {dataset_name}")
        
        # Calculate average metrics
        avg_perplexity = sum(r.perplexity for r in model_results) / len(model_results)
        avg_inference_time = sum(r.inference_time_ms for r in model_results) / len(model_results)
        avg_memory_usage = sum(r.memory_usage_mb for r in model_results) / len(model_results)
        
        logger.info(f"  Average Perplexity: {avg_perplexity:.4f}")
        logger.info(f"  Average Inference Time: {avg_inference_time:.2f} ms")
        logger.info(f"  Average Memory Usage: {avg_memory_usage:.2f} MB")
        logger.info(f"  Parameters: {model_results[0].parameters_count:,}")
        if model_results[0].flops:
            logger.info(f"  Estimated FLOPs: {model_results[0].flops:,}")


if __name__ == "__main__":
    main() 