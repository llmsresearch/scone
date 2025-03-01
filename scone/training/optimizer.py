"""Optimizer utilities for SCONE."""

from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.optim import AdamW, Optimizer

from scone.models.language_model import SconeLanguageModel


def get_optimizer(
    model: SconeLanguageModel,
    lr: float = 5e-5,
    weight_decay: float = 0.01,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> Optimizer:
    """Get optimizer for SCONE model.
    
    This function creates an AdamW optimizer for the SCONE model with
    appropriate parameter groups and learning rates.
    
    Args:
        model: The SCONE language model.
        lr: Learning rate.
        weight_decay: Weight decay.
        betas: AdamW betas.
        eps: AdamW epsilon.
    
    Returns:
        AdamW optimizer.
    """
    # Create parameter groups with different learning rates
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    
    # Create optimizer
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=lr,
        betas=betas,
        eps=eps,
    )
    
    return optimizer


def get_lr_scheduler(
    optimizer: Optimizer,
    num_training_steps: int,
    warmup_steps: int = 0,
    last_epoch: int = -1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Get learning rate scheduler.
    
    This function creates a learning rate scheduler with linear warmup and decay.
    
    Args:
        optimizer: Optimizer.
        num_training_steps: Total number of training steps.
        warmup_steps: Number of warmup steps.
        last_epoch: Last epoch.
    
    Returns:
        Learning rate scheduler.
    """
    # Define learning rate schedule function
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step) / float(max(1, num_training_steps - warmup_steps)),
        )
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda,
        last_epoch=last_epoch,
    )
    
    return scheduler 