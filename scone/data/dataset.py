"""Dataset class for SCONE."""

from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from scone.tokenization.f_gram_tokenizer import FGramTokenizer


class SconeDataset(Dataset):
    """Dataset for training SCONE models.
    
    This dataset handles tokenization and f-gram extraction for training
    SCONE models. It supports both causal language modeling and masked
    language modeling.
    
    Attributes:
        texts: List of texts.
        tokenizer: The tokenizer for the model.
        f_gram_tokenizer: The f-gram tokenizer.
        max_length: Maximum sequence length.
        task: Task type (causal_lm or masked_lm).
    """
    
    def __init__(
        self,
        texts: List[str],
        tokenizer: PreTrainedTokenizer,
        f_gram_tokenizer: FGramTokenizer,
        max_length: int = 512,
        task: str = "causal_lm",
    ) -> None:
        """Initialize the SconeDataset.
        
        Args:
            texts: List of texts.
            tokenizer: The tokenizer for the model.
            f_gram_tokenizer: The f-gram tokenizer.
            max_length: Maximum sequence length.
            task: Task type (causal_lm or masked_lm).
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.f_gram_tokenizer = f_gram_tokenizer
        self.max_length = max_length
        self.task = task
        
        # Validate task
        if self.task not in ["causal_lm", "masked_lm"]:
            raise ValueError(f"Invalid task: {self.task}")
    
    def __len__(self) -> int:
        """Get the length of the dataset.
        
        Returns:
            Length of the dataset.
        """
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get an item from the dataset.
        
        Args:
            idx: Index of the item.
            
        Returns:
            Dictionary containing:
                - input_ids: Token IDs.
                - attention_mask: Attention mask.
                - labels: Labels for language modeling.
                - f_gram_ids: F-gram IDs.
                - f_gram_attention_mask: Attention mask for f-grams.
        """
        text = self.texts[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        # Get f-grams
        f_gram_encoding = self.f_gram_tokenizer.tokenize(
            text,
            return_f_grams=True,
            max_length=self.max_length,
            padding=False,
            truncation=True,
        )
        
        # Create labels
        if self.task == "causal_lm":
            # For causal language modeling, labels are the same as input_ids
            labels = encoding["input_ids"].clone()
        else:
            # For masked language modeling, labels are -100 for non-masked tokens
            labels = encoding["input_ids"].clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            
            # Mask 15% of tokens
            mask_indices = torch.bernoulli(torch.full(labels.shape, 0.15)).bool()
            mask_indices &= (labels != self.tokenizer.pad_token_id)
            
            # Set masked tokens to mask token ID
            encoding["input_ids"][mask_indices] = self.tokenizer.mask_token_id
            
            # Set non-masked tokens to -100 in labels
            labels[~mask_indices] = -100
        
        # Create f-gram IDs and attention mask
        f_gram_ids = []
        f_gram_attention_mask = []
        
        for pos, f_grams in f_gram_encoding["token_f_grams"].items():
            if pos >= self.max_length:
                continue
                
            if not f_grams:
                continue
                
            # Get f-gram IDs
            pos_f_gram_ids = [
                self.f_gram_tokenizer.n_gram_extractor.f_gram_to_id[f_gram]
                for f_gram in f_grams
            ]
            
            # Add f-gram IDs and attention mask
            f_gram_ids.extend(pos_f_gram_ids)
            f_gram_attention_mask.extend([1] * len(pos_f_gram_ids))
        
        # Pad f-gram IDs and attention mask
        max_f_grams = 10  # Maximum number of f-grams per sequence
        if len(f_gram_ids) > max_f_grams:
            f_gram_ids = f_gram_ids[:max_f_grams]
            f_gram_attention_mask = f_gram_attention_mask[:max_f_grams]
        else:
            f_gram_ids.extend([0] * (max_f_grams - len(f_gram_ids)))
            f_gram_attention_mask.extend([0] * (max_f_grams - len(f_gram_attention_mask)))
        
        # Convert to tensors
        f_gram_ids = torch.tensor(f_gram_ids, dtype=torch.long)
        f_gram_attention_mask = torch.tensor(f_gram_attention_mask, dtype=torch.float)
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
            "f_gram_ids": f_gram_ids,
            "f_gram_attention_mask": f_gram_attention_mask,
        }
    
    @classmethod
    def from_huggingface_dataset(
        cls,
        dataset,
        tokenizer: PreTrainedTokenizer,
        f_gram_tokenizer: FGramTokenizer,
        text_column: str = "text",
        max_length: int = 512,
        task: str = "causal_lm",
    ) -> "SconeDataset":
        """Create a SconeDataset from a Hugging Face dataset.
        
        Args:
            dataset: Hugging Face dataset.
            tokenizer: The tokenizer for the model.
            f_gram_tokenizer: The f-gram tokenizer.
            text_column: Name of the text column.
            max_length: Maximum sequence length.
            task: Task type (causal_lm or masked_lm).
            
        Returns:
            SconeDataset.
        """
        texts = dataset[text_column]
        return cls(
            texts=texts,
            tokenizer=tokenizer,
            f_gram_tokenizer=f_gram_tokenizer,
            max_length=max_length,
            task=task,
        ) 