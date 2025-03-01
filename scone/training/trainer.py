"""Trainer for SCONE models."""

import os
import time
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from scone.models.language_model import SconeLanguageModel
from scone.training.optimizer import get_optimizer
from scone.utils.logging import get_logger

logger = get_logger(__name__)


class SconeTrainer:
    """Trainer for SCONE models.
    
    This class provides functionality for training SCONE models. It handles
    training loops, optimization, and evaluation.
    
    Attributes:
        model: The SCONE language model.
        train_dataloader: DataLoader for training data.
        eval_dataloader: DataLoader for evaluation data.
        optimizer: Optimizer for training.
        lr_scheduler: Learning rate scheduler.
        device: Device to train on.
        output_dir: Directory to save outputs.
        max_grad_norm: Maximum gradient norm for gradient clipping.
        logging_steps: Number of steps between logging.
        save_steps: Number of steps between saving.
        eval_steps: Number of steps between evaluation.
        fp16: Whether to use mixed precision training.
        local_rank: Rank of the current process in distributed training.
        gradient_accumulation_steps: Number of steps to accumulate gradients.
    """
    
    def __init__(
        self,
        model: SconeLanguageModel,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[torch.device] = None,
        output_dir: Optional[str] = None,
        max_grad_norm: float = 1.0,
        logging_steps: int = 100,
        save_steps: int = 1000,
        eval_steps: int = 1000,
        fp16: bool = False,
        local_rank: int = -1,
        gradient_accumulation_steps: int = 1,
    ) -> None:
        """Initialize the SconeTrainer.
        
        Args:
            model: The SCONE language model.
            train_dataloader: DataLoader for training data.
            eval_dataloader: DataLoader for evaluation data.
            optimizer: Optimizer for training.
            lr_scheduler: Learning rate scheduler.
            device: Device to train on.
            output_dir: Directory to save outputs.
            max_grad_norm: Maximum gradient norm for gradient clipping.
            logging_steps: Number of steps between logging.
            save_steps: Number of steps between saving.
            eval_steps: Number of steps between evaluation.
            fp16: Whether to use mixed precision training.
            local_rank: Rank of the current process in distributed training.
            gradient_accumulation_steps: Number of steps to accumulate gradients.
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        
        # Set device
        self.device = device
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set optimizer
        self.optimizer = optimizer
        if self.optimizer is None:
            self.optimizer = get_optimizer(self.model)
        
        # Set learning rate scheduler
        self.lr_scheduler = lr_scheduler
        
        # Set output directory
        self.output_dir = output_dir
        if self.output_dir is not None and not os.path.exists(self.output_dir) and (local_rank == -1 or local_rank == 0):
            os.makedirs(self.output_dir)
        
        # Set training parameters
        self.max_grad_norm = max_grad_norm
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.fp16 = fp16
        self.local_rank = local_rank
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Initialize mixed precision training
        self.scaler = GradScaler() if self.fp16 else None
        
        # Initialize distributed training
        self.is_distributed = local_rank != -1
        
        # Initialize training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float("inf")
    
    def train(
        self,
        num_epochs: int,
        resume_from_checkpoint: Optional[str] = None,
    ) -> None:
        """Train the model.
        
        Args:
            num_epochs: Number of epochs to train for.
            resume_from_checkpoint: Path to checkpoint to resume from.
        """
        # Resume from checkpoint if provided
        if resume_from_checkpoint is not None:
            self._load_checkpoint(resume_from_checkpoint)
        
        # Training loop
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            
            # Set epoch for distributed sampler
            if self.is_distributed and hasattr(self.train_dataloader.sampler, "set_epoch"):
                self.train_dataloader.sampler.set_epoch(epoch)
            
            self._train_epoch()
            
            # Save checkpoint at the end of each epoch (only on main process)
            if self.output_dir is not None and (self.local_rank == -1 or self.local_rank == 0):
                self._save_checkpoint(os.path.join(self.output_dir, f"checkpoint-{self.global_step}"))
    
    def _train_epoch(self) -> None:
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        epoch_start_time = time.time()
        
        # Progress bar (only on main process)
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {self.epoch}",
            disable=not (self.local_rank == -1 or self.local_rank == 0),
        )
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass with mixed precision if enabled
            with autocast(enabled=self.fp16):
                outputs = self.model(**batch)
                loss = outputs["loss"]
                
                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps
            
            # Backward pass with mixed precision if enabled
            if self.fp16:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights if gradient accumulation is complete
            if (step + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.fp16:
                    self.scaler.unscale_(self.optimizer)
                
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                # Optimizer step with mixed precision if enabled
                if self.fp16:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # Learning rate scheduler step
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                
                # Update global step
                self.global_step += 1
                
                # Logging
                if self.global_step % self.logging_steps == 0 and (self.local_rank == -1 or self.local_rank == 0):
                    # Calculate training speed
                    elapsed = time.time() - epoch_start_time
                    steps_per_second = (step + 1) / elapsed
                    samples_per_second = (step + 1) * self.train_dataloader.batch_size / elapsed
                    
                    self._log_metrics({
                        "loss": loss.item() * self.gradient_accumulation_steps,
                        "learning_rate": self.lr_scheduler.get_last_lr()[0] if self.lr_scheduler else 0.0,
                        "steps_per_second": steps_per_second,
                        "samples_per_second": samples_per_second,
                    })
                
                # Save checkpoint
                if self.output_dir is not None and self.global_step % self.save_steps == 0 and (self.local_rank == -1 or self.local_rank == 0):
                    self._save_checkpoint(os.path.join(self.output_dir, f"checkpoint-{self.global_step}"))
                
                # Evaluation
                if self.eval_dataloader is not None and self.global_step % self.eval_steps == 0:
                    eval_metrics = self.evaluate()
                    
                    # Log metrics on main process
                    if self.local_rank == -1 or self.local_rank == 0:
                        self._log_metrics(eval_metrics, prefix="eval")
                        
                        # Save best model
                        if eval_metrics["loss"] < self.best_eval_loss:
                            self.best_eval_loss = eval_metrics["loss"]
                            if self.output_dir is not None:
                                self._save_checkpoint(os.path.join(self.output_dir, "best_model"))
            
            # Update progress bar
            if self.local_rank == -1 or self.local_rank == 0:
                epoch_loss += loss.item() * self.gradient_accumulation_steps
                progress_bar.set_postfix({
                    "loss": epoch_loss / (step + 1),
                    "lr": self.lr_scheduler.get_last_lr()[0] if self.lr_scheduler else 0.0,
                })
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model.
        
        Returns:
            Dictionary of evaluation metrics.
        """
        if self.eval_dataloader is None:
            raise ValueError("Evaluation dataloader not provided")
        
        self.model.eval()
        eval_loss = 0.0
        
        # Progress bar (only on main process)
        progress_bar = tqdm(
            self.eval_dataloader,
            desc="Evaluating",
            disable=not (self.local_rank == -1 or self.local_rank == 0),
        )
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            with torch.no_grad():
                with autocast(enabled=self.fp16):
                    outputs = self.model(**batch)
                    loss = outputs["loss"]
            
            # Update metrics
            eval_loss += loss.item()
            
            # Update progress bar
            if self.local_rank == -1 or self.local_rank == 0:
                progress_bar.set_postfix({"loss": eval_loss / (step + 1)})
        
        # Calculate metrics
        eval_loss /= len(self.eval_dataloader)
        perplexity = torch.exp(torch.tensor(eval_loss)).item()
        
        # Synchronize metrics across processes in distributed training
        if self.is_distributed:
            # Create tensor for each metric
            eval_loss_tensor = torch.tensor(eval_loss, device=self.device)
            perplexity_tensor = torch.tensor(perplexity, device=self.device)
            
            # All-reduce
            dist.all_reduce(eval_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(perplexity_tensor, op=dist.ReduceOp.SUM)
            
            # Average
            world_size = dist.get_world_size()
            eval_loss = eval_loss_tensor.item() / world_size
            perplexity = perplexity_tensor.item() / world_size
        
        return {
            "loss": eval_loss,
            "perplexity": perplexity,
        }
    
    def _log_metrics(self, metrics: Dict[str, float], prefix: str = "") -> None:
        """Log metrics.
        
        Args:
            metrics: Dictionary of metrics to log.
            prefix: Prefix for metric names.
        """
        # Add prefix to metric names
        if prefix:
            metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}
        
        # Log metrics
        log_str = f"Step {self.global_step}:"
        for name, value in metrics.items():
            log_str += f" {name}: {value:.4f}"
        
        logger.info(log_str)
    
    def _save_checkpoint(self, path: str) -> None:
        """Save a checkpoint.
        
        Args:
            path: Path to save the checkpoint.
        """
        # Create directory if it doesn't exist
        if not os.path.exists(path):
            os.makedirs(path)
        
        # Get model state dict (unwrap DDP if needed)
        if hasattr(self.model, "module"):
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()
        
        # Create checkpoint
        checkpoint = {
            "model": model_state_dict,
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_eval_loss": self.best_eval_loss,
        }
        
        # Save checkpoint
        torch.save(checkpoint, os.path.join(path, "trainer_state.pt"))
        
        # Save model
        if hasattr(self.model, "module"):
            self.model.module.save_pretrained(path)
        else:
            self.model.save_pretrained(path)
        
        logger.info(f"Saved checkpoint to {path}")
    
    def _load_checkpoint(self, path: str) -> None:
        """Load a checkpoint.
        
        Args:
            path: Path to load the checkpoint from.
        """
        # Load checkpoint
        checkpoint = torch.load(os.path.join(path, "trainer_state.pt"), map_location=self.device)
        
        # Load model
        if hasattr(self.model, "module"):
            self.model.module.load_state_dict(checkpoint["model"])
        else:
            self.model.load_state_dict(checkpoint["model"])
        
        # Load optimizer
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        
        # Load learning rate scheduler
        if self.lr_scheduler is not None and checkpoint["lr_scheduler"] is not None:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        
        # Load scaler
        if self.scaler is not None and checkpoint["scaler"] is not None:
            self.scaler.load_state_dict(checkpoint["scaler"])
        
        # Load training state
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_eval_loss = checkpoint["best_eval_loss"]
        
        logger.info(f"Loaded checkpoint from {path} (epoch {self.epoch}, step {self.global_step})") 