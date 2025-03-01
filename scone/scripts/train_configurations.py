#!/usr/bin/env python
"""Script to train SCONE models with different f-gram configurations.

This script trains multiple SCONE models with different f-gram configurations
as described in the paper "Scaling Embedding Layers in Language Models" (arXiv:2502.01637).
It supports training models with different numbers of f-grams and different f-gram model sizes.
"""

import argparse
import json
import os
import subprocess
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Union, Any

from scone.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FGramConfig:
    """Configuration for f-gram training."""
    # F-gram extraction settings
    max_n: int
    min_freq: int
    max_f_grams: int
    
    # F-gram model settings
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Base model settings
    base_model_name: str
    
    # Dataset settings
    dataset_name: str
    dataset_config_name: str
    text_column: str
    max_length: int
    
    # Training settings
    output_dir: str
    num_epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    gradient_accumulation_steps: int
    fp16: bool
    gradient_checkpointing: bool
    
    # Distributed training settings
    distributed: bool
    num_gpus: int


def get_f_gram_configs() -> Dict[str, FGramConfig]:
    """Get f-gram configurations for different model sizes.
    
    Returns:
        Dictionary of f-gram configurations.
    """
    return {
        # Small f-gram model configurations
        "small-100k": FGramConfig(
            max_n=3,
            min_freq=100,
            max_f_grams=100000,
            hidden_size=384,
            num_hidden_layers=6,
            num_attention_heads=6,
            intermediate_size=1536,
        ),
        "small-500k": FGramConfig(
            max_n=3,
            min_freq=50,
            max_f_grams=500000,
            hidden_size=384,
            num_hidden_layers=6,
            num_attention_heads=6,
            intermediate_size=1536,
        ),
        "small-1m": FGramConfig(
            max_n=3,
            min_freq=20,
            max_f_grams=1000000,
            hidden_size=384,
            num_hidden_layers=6,
            num_attention_heads=6,
            intermediate_size=1536,
        ),
        
        # Medium f-gram model configurations
        "medium-100k": FGramConfig(
            max_n=3,
            min_freq=100,
            max_f_grams=100000,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
        ),
        "medium-500k": FGramConfig(
            max_n=3,
            min_freq=50,
            max_f_grams=500000,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
        ),
        "medium-1m": FGramConfig(
            max_n=3,
            min_freq=20,
            max_f_grams=1000000,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
        ),
        
        # Large f-gram model configurations
        "large-100k": FGramConfig(
            max_n=3,
            min_freq=100,
            max_f_grams=100000,
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            intermediate_size=4096,
        ),
        "large-500k": FGramConfig(
            max_n=3,
            min_freq=50,
            max_f_grams=500000,
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            intermediate_size=4096,
        ),
        "large-1m": FGramConfig(
            max_n=3,
            min_freq=20,
            max_f_grams=1000000,
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            intermediate_size=4096,
        ),
    }


def create_hydra_config(
    f_gram_config: FGramConfig,
    training_config: TrainingConfig,
    config_name: str,
) -> str:
    """Create a Hydra configuration file for a specific f-gram configuration.
    
    Args:
        f_gram_config: F-gram configuration.
        training_config: Training configuration.
        config_name: Name of the configuration.
        
    Returns:
        Path to the created configuration file.
    """
    # Create output directory
    output_dir = os.path.join(training_config.output_dir, f"scone-{config_name}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create configuration file
    config_path = os.path.join(output_dir, "config.yaml")
    
    # Create configuration content
    config = {
        "defaults": [
            "_self_",
            {"override hydra/job_logging": "colorlog"},
            {"override hydra/hydra_logging": "colorlog"},
        ],
        "hydra": {
            "run": {
                "dir": "${training.output_dir}/runs/${now:%Y-%m-%d_%H-%M-%S}",
            },
            "sweep": {
                "dir": "${training.output_dir}/multirun/${now:%Y-%m-%d_%H-%M-%S}",
                "subdir": "${hydra.job.num}",
            },
            "job": {
                "config": {
                    "override_dirname": {
                        "exclude_keys": [
                            "training.output_dir",
                            "training.seed",
                            "hydra.run.dir",
                            "hydra.sweep.dir",
                        ],
                    },
                },
            },
        },
        "data": {
            "dataset_name": training_config.dataset_name,
            "dataset_config_name": training_config.dataset_config_name,
            "text_column": training_config.text_column,
            "max_length": training_config.max_length,
            "max_n": f_gram_config.max_n,
            "min_freq": f_gram_config.min_freq,
            "max_f_grams": f_gram_config.max_f_grams,
        },
        "model": {
            "base_model_name": training_config.base_model_name,
            "hidden_size": f_gram_config.hidden_size,
            "num_hidden_layers": f_gram_config.num_hidden_layers,
            "num_attention_heads": f_gram_config.num_attention_heads,
            "intermediate_size": f_gram_config.intermediate_size,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 1024,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "layer_norm_eps": 1e-12,
            "use_f_gram_embeddings": True,
        },
        "training": {
            "output_dir": output_dir,
            "num_epochs": training_config.num_epochs,
            "batch_size": training_config.batch_size,
            "learning_rate": training_config.learning_rate,
            "weight_decay": training_config.weight_decay,
            "warmup_steps": training_config.warmup_steps,
            "max_grad_norm": 1.0,
            "logging_steps": 100,
            "save_steps": 1000,
            "eval_steps": 1000,
            "seed": 42,
            "gradient_accumulation_steps": training_config.gradient_accumulation_steps,
            "fp16": training_config.fp16,
            "gradient_checkpointing": training_config.gradient_checkpointing,
            "num_workers": 4,
        },
        "inference": {
            "use_memory_map": True,
            "batch_size": 32,
        },
    }
    
    # Write configuration to file
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return config_path


def train_model(
    config_path: str,
    training_config: TrainingConfig,
) -> None:
    """Train a model with a specific configuration.
    
    Args:
        config_path: Path to the configuration file.
        training_config: Training configuration.
    """
    # Create command
    if training_config.distributed and training_config.num_gpus > 1:
        cmd = [
            "python", "-m", "torch.distributed.launch",
            f"--nproc_per_node={training_config.num_gpus}",
            "-m", "scone.scripts.hydra_train",
            f"--config-path={os.path.dirname(config_path)}",
            f"--config-name={os.path.basename(config_path)}",
        ]
    else:
        cmd = [
            "python", "-m", "scone.scripts.hydra_train",
            f"--config-path={os.path.dirname(config_path)}",
            f"--config-name={os.path.basename(config_path)}",
        ]
    
    # Run command
    logger.info(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def precompute_embeddings(
    model_path: str,
    output_path: str,
    batch_size: int = 32,
    use_memory_map: bool = True,
) -> None:
    """Precompute embeddings for a trained model.
    
    Args:
        model_path: Path to the trained model.
        output_path: Path to save the embeddings.
        batch_size: Batch size for precomputation.
        use_memory_map: Whether to use memory mapping.
    """
    # Create command
    cmd = [
        "python", "-m", "scone.scripts.precompute_embeddings",
        f"--model_path={model_path}",
        f"--output_path={output_path}",
        f"--batch_size={batch_size}",
    ]
    
    if use_memory_map:
        cmd.append("--use_memory_map")
    
    # Run command
    logger.info(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train SCONE models with different f-gram configurations")
    
    # Base model settings
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="gpt2",
        help="Base model name",
    )
    
    # Dataset settings
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="wikitext",
        help="Dataset name",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default="wikitext-103-v1",
        help="Dataset config name",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Text column name",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length",
    )
    
    # Training settings
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Output directory",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="Warmup steps",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use mixed precision training",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Use gradient checkpointing",
    )
    
    # Distributed training settings
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Use distributed training",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use",
    )
    
    # Configuration settings
    parser.add_argument(
        "--configs",
        type=str,
        nargs="+",
        default=["small-100k", "small-500k", "small-1m", "medium-100k", "medium-500k", "medium-1m"],
        help="F-gram configurations to train",
    )
    
    return parser.parse_args()


def main() -> None:
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Create training configuration
    training_config = TrainingConfig(
        base_model_name=args.base_model_name,
        dataset_name=args.dataset_name,
        dataset_config_name=args.dataset_config_name,
        text_column=args.text_column,
        max_length=args.max_length,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        distributed=args.distributed,
        num_gpus=args.num_gpus,
    )
    
    # Get f-gram configurations
    f_gram_configs = get_f_gram_configs()
    
    # Train models with different configurations
    for config_name in args.configs:
        if config_name not in f_gram_configs:
            logger.warning(f"Unknown configuration: {config_name}. Skipping.")
            continue
        
        logger.info(f"Training model with configuration: {config_name}")
        
        # Get f-gram configuration
        f_gram_config = f_gram_configs[config_name]
        
        # Create Hydra configuration
        config_path = create_hydra_config(
            f_gram_config=f_gram_config,
            training_config=training_config,
            config_name=config_name,
        )
        
        # Train model
        train_model(
            config_path=config_path,
            training_config=training_config,
        )
        
        # Precompute embeddings
        model_path = os.path.join(training_config.output_dir, f"scone-{config_name}", "final_model")
        embedding_cache_path = os.path.join(training_config.output_dir, f"scone-{config_name}", "embeddings.cache")
        
        precompute_embeddings(
            model_path=model_path,
            output_path=embedding_cache_path,
            batch_size=32,
            use_memory_map=True,
        )
        
        logger.info(f"Finished training model with configuration: {config_name}")


if __name__ == "__main__":
    import yaml
    main() 