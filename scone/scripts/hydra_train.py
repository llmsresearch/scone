#!/usr/bin/env python
"""Hydra-based training script for SCONE models."""

import os
import random
from typing import Dict, List, Optional, Tuple, Union, Any

import hydra
import numpy as np
import torch
import torch.distributed as dist
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, set_seed

from scone.configs.hydra_config import SconeConfig
from scone.data.dataset import SconeDataset
from scone.data.preprocessing import extract_f_grams
from scone.models.f_gram_model import FGramConfig, FGramModel
from scone.models.language_model import SconeConfig as ModelSconeConfig
from scone.models.language_model import SconeLanguageModel
from scone.tokenization.f_gram_tokenizer import FGramTokenizer
from scone.training.optimizer import get_optimizer, get_lr_scheduler
from scone.training.trainer import SconeTrainer
from scone.utils.logging import get_logger

logger = get_logger(__name__)


def setup_distributed(rank: int, world_size: int) -> None:
    """Set up distributed training.
    
    Args:
        rank: Rank of the current process.
        world_size: Number of processes.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_distributed() -> None:
    """Clean up distributed training."""
    dist.destroy_process_group()


@hydra.main(config_path="../configs", config_name="hydra_config")
def main(cfg: DictConfig) -> None:
    """Main function.
    
    Args:
        cfg: Hydra configuration.
    """
    # Convert config to Python dataclass
    config = OmegaConf.to_object(cfg)
    
    # Set random seed
    seed = config.training.seed
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Distributed training setup
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    distributed = world_size > 1
    
    if distributed:
        setup_distributed(local_rank, world_size)
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create output directory
    output_dir = config.training.output_dir
    if local_rank <= 0:  # Only create directory on main process
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    # Load dataset
    dataset = load_dataset(
        config.data.dataset_name,
        config.data.dataset_config_name,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Extract f-grams
    if local_rank <= 0:  # Only extract f-grams on main process
        logger.info("Extracting f-grams...")
        n_gram_extractor = extract_f_grams(
            texts=dataset["train"][config.data.text_column],
            tokenizer=tokenizer,
            max_n=config.data.max_n,
            min_freq=config.data.min_freq,
            max_f_grams=config.data.max_f_grams,
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
    
    # Synchronize processes
    if distributed:
        dist.barrier()
    
    # Load f-gram tokenizer on other processes
    if distributed and local_rank > 0:
        f_gram_tokenizer = FGramTokenizer.from_pretrained(os.path.join(output_dir, "f_gram_tokenizer"))
    
    # Create datasets
    train_dataset = SconeDataset.from_huggingface_dataset(
        dataset=dataset["train"],
        text_column=config.data.text_column,
        tokenizer=tokenizer,
        f_gram_tokenizer=f_gram_tokenizer,
        max_length=config.data.max_length,
        task="causal_lm",
    )
    
    eval_dataset = SconeDataset.from_huggingface_dataset(
        dataset=dataset["validation"],
        text_column=config.data.text_column,
        tokenizer=tokenizer,
        f_gram_tokenizer=f_gram_tokenizer,
        max_length=config.data.max_length,
        task="causal_lm",
    )
    
    # Create samplers for distributed training
    train_sampler = DistributedSampler(train_dataset) if distributed else None
    eval_sampler = DistributedSampler(eval_dataset, shuffle=False) if distributed else None
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config.training.get("num_workers", 0),
        pin_memory=True,
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        sampler=eval_sampler,
        num_workers=config.training.get("num_workers", 0),
        pin_memory=True,
    )
    
    # Create f-gram model
    f_gram_config = FGramConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=config.model.hidden_size,
        num_hidden_layers=config.model.num_hidden_layers,
        num_attention_heads=config.model.num_attention_heads,
        intermediate_size=config.model.intermediate_size,
        hidden_act=config.model.hidden_act,
        hidden_dropout_prob=config.model.hidden_dropout_prob,
        attention_probs_dropout_prob=config.model.attention_probs_dropout_prob,
        max_position_embeddings=config.model.max_position_embeddings,
        type_vocab_size=config.model.type_vocab_size,
        initializer_range=config.model.initializer_range,
        layer_norm_eps=config.model.layer_norm_eps,
    )
    
    f_gram_model = FGramModel(f_gram_config)
    
    # Create SCONE model
    scone_config = ModelSconeConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=config.model.hidden_size,
        num_hidden_layers=config.model.num_hidden_layers,
        num_attention_heads=config.model.num_attention_heads,
        intermediate_size=config.model.intermediate_size,
        hidden_act=config.model.hidden_act,
        hidden_dropout_prob=config.model.hidden_dropout_prob,
        attention_probs_dropout_prob=config.model.attention_probs_dropout_prob,
        max_position_embeddings=config.model.max_position_embeddings,
        type_vocab_size=config.model.type_vocab_size,
        initializer_range=config.model.initializer_range,
        layer_norm_eps=config.model.layer_norm_eps,
        f_gram_model_config=f_gram_config,
        use_f_gram_embeddings=config.model.use_f_gram_embeddings,
    )
    
    model = SconeLanguageModel(scone_config)
    
    # Enable gradient checkpointing if specified
    if config.training.get("gradient_checkpointing", False):
        logger.info("Enabling gradient checkpointing")
        model.gradient_checkpointing_enable()
    
    # Move model to device
    model.to(device)
    
    # Wrap model with DDP for distributed training
    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
    
    # Create optimizer and scheduler
    optimizer = get_optimizer(
        model=model,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    
    num_training_steps = len(train_dataloader) * config.training.num_epochs
    lr_scheduler = get_lr_scheduler(
        optimizer=optimizer,
        num_training_steps=num_training_steps,
        warmup_steps=config.training.warmup_steps,
    )
    
    # Create trainer
    trainer = SconeTrainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        output_dir=output_dir,
        max_grad_norm=config.training.max_grad_norm,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        eval_steps=config.training.eval_steps,
        device=device,
        fp16=config.training.fp16,
        local_rank=local_rank,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
    )
    
    # Train model
    trainer.train(
        num_epochs=config.training.num_epochs,
        resume_from_checkpoint=config.training.get("resume_from_checkpoint"),
    )
    
    # Save final model (only on main process)
    if local_rank <= 0:
        # Get the model without DDP wrapper
        if distributed:
            model_to_save = model.module
        else:
            model_to_save = model
        
        model_to_save.save_pretrained(os.path.join(output_dir, "final_model"))
    
    # Clean up distributed training
    if distributed:
        cleanup_distributed()


if __name__ == "__main__":
    main() 