#!/usr/bin/env python
"""Training script for SCONE models."""

import argparse
import os
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from scone.data.dataset import SconeDataset
from scone.data.preprocessing import extract_f_grams
from scone.models.f_gram_model import FGramConfig, FGramModel
from scone.models.language_model import SconeConfig, SconeLanguageModel
from scone.tokenization.f_gram_tokenizer import FGramTokenizer
from scone.training.optimizer import get_optimizer, get_lr_scheduler
from scone.training.trainer import SconeTrainer
from scone.utils.config import get_config


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train SCONE model")
    
    # Configuration arguments
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file",
    )
    
    # Override arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="HuggingFace dataset config name",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default=None,
        help="Column name for text data",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--max_n",
        type=int,
        default=None,
        help="Maximum n-gram length",
    )
    parser.add_argument(
        "--min_freq",
        type=int,
        default=None,
        help="Minimum frequency for f-grams",
    )
    parser.add_argument(
        "--max_f_grams",
        type=int,
        default=None,
        help="Maximum number of f-grams",
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        default=None,
        help="Base model name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=None,
        help="Weight decay",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=None,
        help="Number of warmup steps",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=None,
        help="Maximum gradient norm",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=None,
        help="Number of steps between logging",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=None,
        help="Number of steps between saving",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=None,
        help="Number of steps between evaluation",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=None,
        help="Number of steps to accumulate gradients",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use mixed precision training",
    )
    
    return parser.parse_args()


def get_override_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Get override configuration from command line arguments.
    
    Args:
        args: Command line arguments.
    
    Returns:
        Dictionary containing the override configuration.
    """
    override_config = {}
    
    # Data configuration
    data_config = {}
    if args.dataset_name is not None:
        data_config["dataset_name"] = args.dataset_name
    if args.dataset_config_name is not None:
        data_config["dataset_config_name"] = args.dataset_config_name
    if args.text_column is not None:
        data_config["text_column"] = args.text_column
    if args.max_length is not None:
        data_config["max_length"] = args.max_length
    if args.max_n is not None:
        data_config["max_n"] = args.max_n
    if args.min_freq is not None:
        data_config["min_freq"] = args.min_freq
    if args.max_f_grams is not None:
        data_config["max_f_grams"] = args.max_f_grams
    
    if data_config:
        override_config["data"] = data_config
    
    # Model configuration
    model_config = {}
    if args.base_model_name is not None:
        model_config["base_model_name"] = args.base_model_name
    
    if model_config:
        override_config["model"] = model_config
    
    # Training configuration
    training_config = {}
    if args.output_dir is not None:
        training_config["output_dir"] = args.output_dir
    if args.num_epochs is not None:
        training_config["num_epochs"] = args.num_epochs
    if args.batch_size is not None:
        training_config["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        training_config["learning_rate"] = args.learning_rate
    if args.weight_decay is not None:
        training_config["weight_decay"] = args.weight_decay
    if args.warmup_steps is not None:
        training_config["warmup_steps"] = args.warmup_steps
    if args.max_grad_norm is not None:
        training_config["max_grad_norm"] = args.max_grad_norm
    if args.logging_steps is not None:
        training_config["logging_steps"] = args.logging_steps
    if args.save_steps is not None:
        training_config["save_steps"] = args.save_steps
    if args.eval_steps is not None:
        training_config["eval_steps"] = args.eval_steps
    if args.seed is not None:
        training_config["seed"] = args.seed
    if args.gradient_accumulation_steps is not None:
        training_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    if args.fp16:
        training_config["fp16"] = True
    
    if training_config:
        override_config["training"] = training_config
    
    return override_config


def main() -> None:
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Get configuration
    override_config = get_override_config(args)
    config = get_config(args.config, override_config)
    
    # Set random seed
    seed = config["training"].get("seed", 42)
    torch.manual_seed(seed)
    
    # Create output directory
    output_dir = config["training"]["output_dir"]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load dataset
    dataset = load_dataset(
        config["data"]["dataset_name"],
        config["data"]["dataset_config_name"],
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["base_model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Extract f-grams
    print("Extracting f-grams...")
    n_gram_extractor = extract_f_grams(
        texts=dataset["train"][config["data"]["text_column"]],
        tokenizer=tokenizer,
        max_n=config["data"]["max_n"],
        min_freq=config["data"]["min_freq"],
        max_f_grams=config["data"]["max_f_grams"],
        verbose=True,
    )
    
    # Create f-gram tokenizer
    f_gram_tokenizer = FGramTokenizer(
        n_gram_extractor=n_gram_extractor,
        tokenizer=tokenizer,
    )
    
    # Save tokenizers
    tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))
    f_gram_tokenizer.save_pretrained(os.path.join(output_dir, "f_gram_tokenizer"))
    
    # Create datasets
    train_dataset = SconeDataset.from_huggingface_dataset(
        dataset=dataset["train"],
        text_column=config["data"]["text_column"],
        tokenizer=tokenizer,
        f_gram_tokenizer=f_gram_tokenizer,
        max_length=config["data"]["max_length"],
        task="causal_lm",
    )
    
    eval_dataset = SconeDataset.from_huggingface_dataset(
        dataset=dataset["validation"],
        text_column=config["data"]["text_column"],
        tokenizer=tokenizer,
        f_gram_tokenizer=f_gram_tokenizer,
        max_length=config["data"]["max_length"],
        task="causal_lm",
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
    )
    
    # Create f-gram model
    f_gram_config = FGramConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=config["model"]["hidden_size"],
        num_hidden_layers=config["model"]["num_hidden_layers"],
        num_attention_heads=config["model"]["num_attention_heads"],
        intermediate_size=config["model"]["intermediate_size"],
        hidden_act=config["model"]["hidden_act"],
        hidden_dropout_prob=config["model"]["hidden_dropout_prob"],
        attention_probs_dropout_prob=config["model"]["attention_probs_dropout_prob"],
        max_position_embeddings=config["model"]["max_position_embeddings"],
        type_vocab_size=config["model"]["type_vocab_size"],
        initializer_range=config["model"]["initializer_range"],
        layer_norm_eps=config["model"]["layer_norm_eps"],
    )
    
    f_gram_model = FGramModel(f_gram_config)
    
    # Create SCONE model
    scone_config = SconeConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=config["model"]["hidden_size"],
        num_hidden_layers=config["model"]["num_hidden_layers"],
        num_attention_heads=config["model"]["num_attention_heads"],
        intermediate_size=config["model"]["intermediate_size"],
        hidden_act=config["model"]["hidden_act"],
        hidden_dropout_prob=config["model"]["hidden_dropout_prob"],
        attention_probs_dropout_prob=config["model"]["attention_probs_dropout_prob"],
        max_position_embeddings=config["model"]["max_position_embeddings"],
        type_vocab_size=config["model"]["type_vocab_size"],
        initializer_range=config["model"]["initializer_range"],
        layer_norm_eps=config["model"]["layer_norm_eps"],
        f_gram_model_config=f_gram_config,
        use_f_gram_embeddings=config["model"]["use_f_gram_embeddings"],
    )
    
    model = SconeLanguageModel(scone_config)
    
    # Create optimizer and scheduler
    optimizer = get_optimizer(
        model=model,
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    
    num_training_steps = len(train_dataloader) * config["training"]["num_epochs"]
    lr_scheduler = get_lr_scheduler(
        optimizer=optimizer,
        num_training_steps=num_training_steps,
        warmup_steps=config["training"]["warmup_steps"],
    )
    
    # Create trainer
    trainer = SconeTrainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        output_dir=output_dir,
        max_grad_norm=config["training"]["max_grad_norm"],
        logging_steps=config["training"]["logging_steps"],
        save_steps=config["training"]["save_steps"],
        eval_steps=config["training"]["eval_steps"],
    )
    
    # Train model
    trainer.train(
        num_epochs=config["training"]["num_epochs"],
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    
    # Save final model
    model.save_pretrained(os.path.join(output_dir, "final_model"))


if __name__ == "__main__":
    main() 