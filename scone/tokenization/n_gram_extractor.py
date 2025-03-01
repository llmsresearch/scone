"""N-gram extraction utilities for SCONE."""

from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from tqdm import tqdm


class NGramExtractor:
    """Extracts n-grams from text and identifies frequent n-grams (f-grams).
    
    This class is responsible for extracting n-grams from tokenized text and
    identifying the most frequent n-grams (f-grams) that will be used for
    contextualized embeddings in SCONE.
    
    Attributes:
        max_n: Maximum n-gram length to extract.
        min_freq: Minimum frequency for an n-gram to be considered an f-gram.
        max_f_grams: Maximum number of f-grams to keep.
        f_grams: Set of frequent n-grams (f-grams).
        f_gram_to_id: Mapping from f-gram to its unique ID.
        id_to_f_gram: Mapping from ID to f-gram.
    """
    
    def __init__(
        self,
        max_n: int = 3,
        min_freq: int = 100,
        max_f_grams: int = 10_000_000,
    ) -> None:
        """Initialize the NGramExtractor.
        
        Args:
            max_n: Maximum n-gram length to extract.
            min_freq: Minimum frequency for an n-gram to be considered an f-gram.
            max_f_grams: Maximum number of f-grams to keep.
        """
        self.max_n = max_n
        self.min_freq = min_freq
        self.max_f_grams = max_f_grams
        self.f_grams: Set[Tuple[int, ...]] = set()
        self.f_gram_to_id: Dict[Tuple[int, ...], int] = {}
        self.id_to_f_gram: Dict[int, Tuple[int, ...]] = {}
    
    def extract_n_grams(self, token_ids: List[int], n: int) -> List[Tuple[int, ...]]:
        """Extract n-grams from a list of token IDs.
        
        Args:
            token_ids: List of token IDs.
            n: Length of n-grams to extract.
            
        Returns:
            List of n-grams, where each n-gram is a tuple of token IDs.
        """
        return [tuple(token_ids[i:i+n]) for i in range(len(token_ids) - n + 1)]
    
    def extract_all_n_grams(self, token_ids: List[int]) -> List[Tuple[int, ...]]:
        """Extract all n-grams up to max_n from a list of token IDs.
        
        Args:
            token_ids: List of token IDs.
            
        Returns:
            List of all n-grams, where each n-gram is a tuple of token IDs.
        """
        all_n_grams = []
        for n in range(1, min(self.max_n + 1, len(token_ids) + 1)):
            all_n_grams.extend(self.extract_n_grams(token_ids, n))
        return all_n_grams
    
    def fit(self, tokenized_texts: List[List[int]], verbose: bool = True) -> "NGramExtractor":
        """Identify frequent n-grams (f-grams) from a corpus.
        
        Args:
            tokenized_texts: List of tokenized texts, where each text is a list of token IDs.
            verbose: Whether to show progress bar.
            
        Returns:
            Self for method chaining.
        """
        # Count n-grams
        n_gram_counter = Counter()
        
        iterator = tqdm(tokenized_texts, desc="Extracting n-grams") if verbose else tokenized_texts
        for token_ids in iterator:
            n_grams = self.extract_all_n_grams(token_ids)
            n_gram_counter.update(n_grams)
        
        # Filter by frequency and limit to max_f_grams
        frequent_n_grams = [
            n_gram for n_gram, count in n_gram_counter.most_common(self.max_f_grams)
            if count >= self.min_freq
        ]
        
        # Store f-grams and create mappings
        self.f_grams = set(frequent_n_grams)
        self.f_gram_to_id = {f_gram: i for i, f_gram in enumerate(frequent_n_grams)}
        self.id_to_f_gram = {i: f_gram for i, f_gram in enumerate(frequent_n_grams)}
        
        if verbose:
            print(f"Extracted {len(self.f_grams)} f-grams")
            
        return self
    
    def get_token_f_grams(self, token_ids: List[int]) -> Dict[int, List[Tuple[int, ...]]]:
        """Get all f-grams that contain each token in the input.
        
        For each token position, find all f-grams that include that token.
        
        Args:
            token_ids: List of token IDs.
            
        Returns:
            Dictionary mapping token positions to lists of f-grams.
        """
        token_f_grams = {i: [] for i in range(len(token_ids))}
        
        for n in range(1, min(self.max_n + 1, len(token_ids) + 1)):
            for i in range(len(token_ids) - n + 1):
                n_gram = tuple(token_ids[i:i+n])
                if n_gram in self.f_grams:
                    for j in range(i, i+n):
                        token_f_grams[j].append(n_gram)
        
        return token_f_grams
    
    def save(self, path: str) -> None:
        """Save the extractor to a file.
        
        Args:
            path: Path to save the extractor.
        """
        data = {
            "max_n": self.max_n,
            "min_freq": self.min_freq,
            "max_f_grams": self.max_f_grams,
            "f_gram_to_id": {",".join(map(str, k)): v for k, v in self.f_gram_to_id.items()},
        }
        np.save(path, data, allow_pickle=True)
    
    @classmethod
    def load(cls, path: str) -> "NGramExtractor":
        """Load an extractor from a file.
        
        Args:
            path: Path to load the extractor from.
            
        Returns:
            Loaded NGramExtractor.
        """
        data = np.load(path, allow_pickle=True).item()
        
        extractor = cls(
            max_n=data["max_n"],
            min_freq=data["min_freq"],
            max_f_grams=data["max_f_grams"],
        )
        
        extractor.f_gram_to_id = {
            tuple(map(int, k.split(","))): v for k, v in data["f_gram_to_id"].items()
        }
        extractor.id_to_f_gram = {v: k for k, v in extractor.f_gram_to_id.items()}
        extractor.f_grams = set(extractor.f_gram_to_id.keys())
        
        return extractor 