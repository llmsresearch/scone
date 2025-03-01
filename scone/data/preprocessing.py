"""Preprocessing utilities for SCONE."""

from typing import Dict, List, Optional, Tuple, Union

import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from scone.tokenization.n_gram_extractor import NGramExtractor


def extract_f_grams(
    texts: List[str],
    tokenizer: PreTrainedTokenizer,
    max_n: int = 3,
    min_freq: int = 100,
    max_f_grams: int = 10_000_000,
    verbose: bool = True,
) -> NGramExtractor:
    """Extract frequent n-grams (f-grams) from a corpus.
    
    Args:
        texts: List of texts.
        tokenizer: Tokenizer for tokenizing texts.
        max_n: Maximum n-gram length.
        min_freq: Minimum frequency for an n-gram to be considered an f-gram.
        max_f_grams: Maximum number of f-grams to keep.
        verbose: Whether to show progress bar.
        
    Returns:
        NGramExtractor with extracted f-grams.
    """
    # Initialize n-gram extractor
    n_gram_extractor = NGramExtractor(
        max_n=max_n,
        min_freq=min_freq,
        max_f_grams=max_f_grams,
    )
    
    # Tokenize texts
    tokenized_texts = []
    iterator = tqdm(texts, desc="Tokenizing texts") if verbose else texts
    for text in iterator:
        encoding = tokenizer(text, add_special_tokens=False)
        tokenized_texts.append(encoding["input_ids"])
    
    # Extract f-grams
    n_gram_extractor.fit(tokenized_texts, verbose=verbose)
    
    return n_gram_extractor


def precompute_f_gram_embeddings(
    n_gram_extractor: NGramExtractor,
    f_gram_model,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict[int, torch.Tensor]:
    """Precompute embeddings for f-grams.
    
    Args:
        n_gram_extractor: NGramExtractor with extracted f-grams.
        f_gram_model: Model for generating f-gram embeddings.
        tokenizer: Tokenizer for tokenizing f-grams.
        batch_size: Batch size for processing f-grams.
        device: Device to run inference on.
        verbose: Whether to show progress bar.
        
    Returns:
        Dictionary mapping f-gram IDs to embeddings.
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move model to device
    f_gram_model.to(device)
    
    # Precompute embeddings for f-grams
    f_gram_embeddings = {}
    
    # Process f-grams in batches
    f_grams = list(n_gram_extractor.f_grams)
    num_batches = (len(f_grams) + batch_size - 1) // batch_size
    
    iterator = range(num_batches)
    if verbose:
        iterator = tqdm(iterator, desc="Precomputing f-gram embeddings")
    
    for batch_idx in iterator:
        # Get batch of f-grams
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, len(f_grams))
        batch_f_grams = f_grams[batch_start:batch_end]
        
        # Convert f-grams to token IDs
        batch_token_ids = []
        for f_gram in batch_f_grams:
            batch_token_ids.append(list(f_gram))
        
        # Pad token IDs
        max_length = max(len(token_ids) for token_ids in batch_token_ids)
        padded_token_ids = []
        attention_mask = []
        for token_ids in batch_token_ids:
            padding = [0] * (max_length - len(token_ids))
            padded_token_ids.append(token_ids + padding)
            attention_mask.append([1] * len(token_ids) + [0] * len(padding))
        
        # Convert to tensors
        input_ids = torch.tensor(padded_token_ids, device=device)
        attention_mask = torch.tensor(attention_mask, device=device)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = f_gram_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
        
        # Get last hidden state
        last_hidden_state = outputs["last_hidden_state"]
        
        # Average hidden states for each f-gram
        for i, (token_ids, mask) in enumerate(zip(batch_token_ids, attention_mask)):
            # Get f-gram ID
            f_gram = tuple(token_ids)
            f_gram_id = n_gram_extractor.f_gram_to_id[f_gram]
            
            # Average hidden states
            hidden_states = last_hidden_state[i, :len(token_ids)]
            embedding = hidden_states.mean(dim=0)
            
            # Store embedding
            f_gram_embeddings[f_gram_id] = embedding
    
    return f_gram_embeddings 