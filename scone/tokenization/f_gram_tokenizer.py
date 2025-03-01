"""F-gram tokenizer for SCONE."""

from typing import Dict, List, Optional, Tuple, Union

import torch
from transformers import PreTrainedTokenizer

from scone.tokenization.n_gram_extractor import NGramExtractor


class FGramTokenizer:
    """Tokenizer for handling f-grams in SCONE.
    
    This tokenizer wraps a base tokenizer and adds functionality for
    identifying f-grams in the input text. It uses an NGramExtractor
    to identify f-grams and provides methods for tokenizing text with
    f-gram information.
    
    Attributes:
        base_tokenizer: The base tokenizer to use for tokenizing text.
        n_gram_extractor: The NGramExtractor for identifying f-grams.
    """
    
    def __init__(
        self,
        base_tokenizer: PreTrainedTokenizer,
        n_gram_extractor: NGramExtractor,
    ) -> None:
        """Initialize the FGramTokenizer.
        
        Args:
            base_tokenizer: The base tokenizer to use for tokenizing text.
            n_gram_extractor: The NGramExtractor for identifying f-grams.
        """
        self.base_tokenizer = base_tokenizer
        self.n_gram_extractor = n_gram_extractor
    
    def tokenize(
        self,
        text: str,
        return_f_grams: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False,
    ) -> Dict[str, Union[List[int], Dict[int, List[Tuple[int, ...]]]]]:
        """Tokenize text and identify f-grams.
        
        Args:
            text: The text to tokenize.
            return_f_grams: Whether to return f-gram information.
            max_length: Maximum sequence length.
            padding: Whether to pad sequences.
            truncation: Whether to truncate sequences.
            
        Returns:
            Dictionary containing:
                - input_ids: List of token IDs.
                - attention_mask: Attention mask.
                - token_f_grams: Dictionary mapping token positions to lists of f-grams.
        """
        # Tokenize text using base tokenizer
        encoding = self.base_tokenizer(
            text,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors="pt",
        )
        
        result = {
            "input_ids": encoding["input_ids"].squeeze(0).tolist(),
            "attention_mask": encoding["attention_mask"].squeeze(0).tolist(),
        }
        
        # Get f-grams for each token if requested
        if return_f_grams:
            token_f_grams = self.n_gram_extractor.get_token_f_grams(result["input_ids"])
            result["token_f_grams"] = token_f_grams
        
        return result
    
    def batch_tokenize(
        self,
        texts: List[str],
        return_f_grams: bool = True,
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
    ) -> Dict[str, Union[torch.Tensor, List[Dict[int, List[Tuple[int, ...]]]]]]:
        """Tokenize a batch of texts and identify f-grams.
        
        Args:
            texts: List of texts to tokenize.
            return_f_grams: Whether to return f-gram information.
            max_length: Maximum sequence length.
            padding: Whether to pad sequences.
            truncation: Whether to truncate sequences.
            
        Returns:
            Dictionary containing:
                - input_ids: Tensor of token IDs.
                - attention_mask: Attention mask.
                - token_f_grams: List of dictionaries mapping token positions to lists of f-grams.
        """
        # Tokenize texts using base tokenizer
        encodings = self.base_tokenizer(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors="pt",
        )
        
        result = {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
        }
        
        # Get f-grams for each token in each sequence if requested
        if return_f_grams:
            token_f_grams = []
            for input_ids in encodings["input_ids"].tolist():
                token_f_grams.append(self.n_gram_extractor.get_token_f_grams(input_ids))
            result["token_f_grams"] = token_f_grams
        
        return result
    
    def save_pretrained(self, save_directory: str) -> None:
        """Save the tokenizer and n-gram extractor.
        
        Args:
            save_directory: Directory to save to.
        """
        self.base_tokenizer.save_pretrained(save_directory)
        self.n_gram_extractor.save(f"{save_directory}/n_gram_extractor.npy")
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        n_gram_extractor_path: Optional[str] = None,
    ) -> "FGramTokenizer":
        """Load a tokenizer and n-gram extractor from pretrained files.
        
        Args:
            pretrained_model_name_or_path: Path to pretrained tokenizer.
            n_gram_extractor_path: Path to n-gram extractor. If None, will look in
                the same directory as the tokenizer.
                
        Returns:
            Loaded FGramTokenizer.
        """
        from transformers import AutoTokenizer
        
        base_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        
        if n_gram_extractor_path is None:
            n_gram_extractor_path = f"{pretrained_model_name_or_path}/n_gram_extractor.npy"
        
        n_gram_extractor = NGramExtractor.load(n_gram_extractor_path)
        
        return cls(base_tokenizer, n_gram_extractor) 