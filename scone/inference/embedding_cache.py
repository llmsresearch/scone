"""Embedding cache for SCONE inference."""

import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm

from scone.tokenization.n_gram_extractor import NGramExtractor


class EmbeddingCache:
    """Cache for f-gram embeddings.
    
    This class provides functionality for caching f-gram embeddings in memory
    or on disk. It supports both in-memory caching and memory-mapped caching
    for large embedding tables.
    
    Attributes:
        n_gram_extractor: The NGramExtractor for identifying f-grams.
        embedding_dim: Dimension of the embeddings.
        cache_dir: Directory for storing cached embeddings.
        use_memory_map: Whether to use memory mapping for large caches.
        embeddings: Dictionary mapping f-gram IDs to embeddings.
        memory_mapped_embeddings: Memory-mapped array of embeddings.
    """
    
    def __init__(
        self,
        n_gram_extractor: NGramExtractor,
        embedding_dim: int,
        cache_dir: Optional[str] = None,
        use_memory_map: bool = False,
    ) -> None:
        """Initialize the EmbeddingCache.
        
        Args:
            n_gram_extractor: The NGramExtractor for identifying f-grams.
            embedding_dim: Dimension of the embeddings.
            cache_dir: Directory for storing cached embeddings.
            use_memory_map: Whether to use memory mapping for large caches.
        """
        self.n_gram_extractor = n_gram_extractor
        self.embedding_dim = embedding_dim
        self.cache_dir = cache_dir
        self.use_memory_map = use_memory_map
        
        self.embeddings: Dict[int, np.ndarray] = {}
        self.memory_mapped_embeddings: Optional[np.ndarray] = None
        
        # Create cache directory if needed
        if self.cache_dir is not None and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
    
    def cache_embeddings(
        self,
        f_gram_ids: List[int],
        embeddings: torch.Tensor,
        verbose: bool = True,
    ) -> None:
        """Cache embeddings for f-grams.
        
        Args:
            f_gram_ids: List of f-gram IDs.
            embeddings: Tensor of embeddings.
            verbose: Whether to show progress bar.
        """
        if self.use_memory_map:
            # Create memory-mapped array if it doesn't exist
            if self.memory_mapped_embeddings is None:
                if self.cache_dir is None:
                    raise ValueError("Cache directory must be provided for memory mapping")
                
                # Create memory-mapped array
                mmap_path = os.path.join(self.cache_dir, "embeddings.npy")
                shape = (len(self.n_gram_extractor.f_grams), self.embedding_dim)
                
                if os.path.exists(mmap_path):
                    # Load existing memory-mapped array
                    self.memory_mapped_embeddings = np.load(mmap_path, mmap_mode="r+")
                else:
                    # Create new memory-mapped array
                    self.memory_mapped_embeddings = np.memmap(
                        mmap_path,
                        dtype=np.float32,
                        mode="w+",
                        shape=shape,
                    )
                    # Initialize with zeros
                    self.memory_mapped_embeddings[:] = 0.0
            
            # Cache embeddings in memory-mapped array
            iterator = enumerate(zip(f_gram_ids, embeddings))
            if verbose:
                iterator = tqdm(iterator, total=len(f_gram_ids), desc="Caching embeddings")
            
            for i, (f_gram_id, embedding) in iterator:
                self.memory_mapped_embeddings[f_gram_id] = embedding.cpu().numpy()
            
            # Flush changes to disk
            if hasattr(self.memory_mapped_embeddings, "flush"):
                self.memory_mapped_embeddings.flush()
        else:
            # Cache embeddings in memory
            iterator = zip(f_gram_ids, embeddings)
            if verbose:
                iterator = tqdm(iterator, total=len(f_gram_ids), desc="Caching embeddings")
            
            for f_gram_id, embedding in iterator:
                self.embeddings[f_gram_id] = embedding.cpu().numpy()
    
    def get_embeddings(
        self,
        f_gram_ids: List[int],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Get embeddings for f-grams.
        
        Args:
            f_gram_ids: List of f-gram IDs.
            device: Device to put embeddings on.
            
        Returns:
            Tensor of embeddings.
        """
        if self.use_memory_map:
            # Get embeddings from memory-mapped array
            if self.memory_mapped_embeddings is None:
                raise ValueError("Memory-mapped embeddings not initialized")
            
            embeddings = torch.tensor(
                self.memory_mapped_embeddings[f_gram_ids],
                dtype=torch.float32,
            )
        else:
            # Get embeddings from memory
            embeddings = torch.stack([
                torch.tensor(self.embeddings[f_gram_id], dtype=torch.float32)
                for f_gram_id in f_gram_ids
            ])
        
        # Move embeddings to device if specified
        if device is not None:
            embeddings = embeddings.to(device)
        
        return embeddings
    
    def get_token_embeddings(
        self,
        token_ids: List[int],
        device: Optional[torch.device] = None,
    ) -> Dict[int, torch.Tensor]:
        """Get embeddings for all f-grams containing each token.
        
        Args:
            token_ids: List of token IDs.
            device: Device to put embeddings on.
            
        Returns:
            Dictionary mapping token positions to tensors of f-gram embeddings.
        """
        # Get f-grams for each token
        token_f_grams = self.n_gram_extractor.get_token_f_grams(token_ids)
        
        # Get embeddings for each token's f-grams
        token_embeddings = {}
        for pos, f_grams in token_f_grams.items():
            if not f_grams:
                continue
            
            # Get f-gram IDs
            f_gram_ids = [self.n_gram_extractor.f_gram_to_id[f_gram] for f_gram in f_grams]
            
            # Get embeddings
            embeddings = self.get_embeddings(f_gram_ids, device)
            
            # Store embeddings
            token_embeddings[pos] = embeddings
        
        return token_embeddings
    
    def save(self, path: str) -> None:
        """Save the cache to a file.
        
        Args:
            path: Path to save the cache.
        """
        if self.use_memory_map:
            # For memory-mapped caches, just save the path to the memory-mapped file
            np.save(path, {
                "use_memory_map": True,
                "cache_dir": self.cache_dir,
                "embedding_dim": self.embedding_dim,
            })
        else:
            # For in-memory caches, save the embeddings
            np.save(path, {
                "use_memory_map": False,
                "embeddings": self.embeddings,
                "embedding_dim": self.embedding_dim,
            })
    
    @classmethod
    def load(
        cls,
        path: str,
        n_gram_extractor: NGramExtractor,
    ) -> "EmbeddingCache":
        """Load a cache from a file.
        
        Args:
            path: Path to load the cache from.
            n_gram_extractor: The NGramExtractor for identifying f-grams.
            
        Returns:
            Loaded EmbeddingCache.
        """
        data = np.load(path, allow_pickle=True).item()
        
        if data["use_memory_map"]:
            # For memory-mapped caches, create a new cache with the same cache directory
            cache = cls(
                n_gram_extractor=n_gram_extractor,
                embedding_dim=data["embedding_dim"],
                cache_dir=data["cache_dir"],
                use_memory_map=True,
            )
            
            # Load memory-mapped embeddings
            mmap_path = os.path.join(data["cache_dir"], "embeddings.npy")
            cache.memory_mapped_embeddings = np.load(mmap_path, mmap_mode="r")
        else:
            # For in-memory caches, create a new cache and load the embeddings
            cache = cls(
                n_gram_extractor=n_gram_extractor,
                embedding_dim=data["embedding_dim"],
                use_memory_map=False,
            )
            
            cache.embeddings = data["embeddings"]
        
        return cache 