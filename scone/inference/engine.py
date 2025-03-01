"""Inference engine for SCONE."""

from typing import Dict, List, Optional, Tuple, Union, Literal

import torch
from transformers import PreTrainedTokenizer

from scone.inference.embedding_cache import EmbeddingCache
from scone.models.language_model import SconeLanguageModel
from scone.tokenization.f_gram_tokenizer import FGramTokenizer
from scone.utils.logging import get_logger

logger = get_logger(__name__)


class SconeInferenceEngine:
    """Inference engine for SCONE.
    
    This class provides functionality for running inference with SCONE models.
    It handles loading models, tokenizers, and embedding caches, and provides
    methods for generating text.
    
    Attributes:
        model: The SCONE language model.
        tokenizer: The tokenizer for the model.
        f_gram_tokenizer: The f-gram tokenizer.
        embedding_cache: Cache for f-gram embeddings.
        device: Device to run inference on.
        quantization: Quantization mode (None, "int8", "int4", or "fp16").
    """
    
    def __init__(
        self,
        model: SconeLanguageModel,
        tokenizer: PreTrainedTokenizer,
        f_gram_tokenizer: FGramTokenizer,
        embedding_cache: EmbeddingCache,
        device: Optional[torch.device] = None,
        quantization: Optional[Literal["int8", "int4", "fp16"]] = None,
    ) -> None:
        """Initialize the SconeInferenceEngine.
        
        Args:
            model: The SCONE language model.
            tokenizer: The tokenizer for the model.
            f_gram_tokenizer: The f-gram tokenizer.
            embedding_cache: Cache for f-gram embeddings.
            device: Device to run inference on.
            quantization: Quantization mode (None, "int8", "int4", or "fp16").
        """
        self.model = model
        self.tokenizer = tokenizer
        self.f_gram_tokenizer = f_gram_tokenizer
        self.embedding_cache = embedding_cache
        self.quantization = quantization
        
        # Set device
        self.device = device
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Apply quantization if specified
        if quantization:
            self._apply_quantization(quantization)
        
        # Move model to device
        self.model.to(self.device)
    
    def _apply_quantization(self, quantization: str) -> None:
        """Apply quantization to the model.
        
        Args:
            quantization: Quantization mode ("int8", "int4", or "fp16").
        """
        if quantization == "int8":
            logger.info("Applying INT8 quantization")
            try:
                from torch.quantization import quantize_dynamic
                self.model = quantize_dynamic(
                    self.model,
                    {torch.nn.Linear},
                    dtype=torch.qint8
                )
            except ImportError:
                logger.warning("INT8 quantization requires PyTorch 1.6.0 or higher. Skipping quantization.")
        
        elif quantization == "int4":
            logger.info("Applying INT4 quantization")
            try:
                # Check if bitsandbytes is available
                import bitsandbytes as bnb
                
                # Replace Linear layers with 4-bit quantized layers
                for name, module in self.model.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        parent_name = name.rsplit(".", 1)[0] if "." in name else ""
                        parent = self.model if parent_name == "" else self.model.get_submodule(parent_name)
                        child_name = name.rsplit(".", 1)[1] if "." in name else name
                        
                        # Create 4-bit quantized layer
                        quantized_module = bnb.nn.Linear4bit(
                            module.in_features,
                            module.out_features,
                            bias=module.bias is not None,
                            compute_dtype=torch.float16,
                        )
                        
                        # Copy weights and bias
                        with torch.no_grad():
                            quantized_module.weight = module.weight
                            if module.bias is not None:
                                quantized_module.bias = module.bias
                        
                        # Replace module
                        setattr(parent, child_name, quantized_module)
            
            except ImportError:
                logger.warning("INT4 quantization requires bitsandbytes. Skipping quantization.")
        
        elif quantization == "fp16":
            logger.info("Applying FP16 quantization")
            # Convert model to half precision
            self.model.half()
        
        else:
            logger.warning(f"Unknown quantization mode: {quantization}. Supported modes: int8, int4, fp16")
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        f_gram_tokenizer_path: Optional[str] = None,
        embedding_cache_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        quantization: Optional[str] = None,
        use_memory_map: bool = True,
    ) -> "SconeInferenceEngine":
        """Load a SconeInferenceEngine from pretrained files.
        
        Args:
            model_path: Path to pretrained model.
            tokenizer_path: Path to tokenizer. If None, will use model_path.
            f_gram_tokenizer_path: Path to f-gram tokenizer. If None, will use model_path.
            embedding_cache_path: Path to embedding cache. If None, will use model_path.
            device: Device to run inference on.
            quantization: Quantization mode (None, "int8", "int4", or "fp16").
            use_memory_map: Whether to use memory mapping for embedding cache.
            
        Returns:
            Loaded SconeInferenceEngine.
        """
        from transformers import AutoTokenizer
        
        # Set default paths
        if tokenizer_path is None:
            tokenizer_path = model_path
        if f_gram_tokenizer_path is None:
            f_gram_tokenizer_path = model_path
        if embedding_cache_path is None:
            embedding_cache_path = f"{model_path}/embedding_cache.npy"
        
        # Load model
        logger.info(f"Loading model from {model_path}")
        model = SconeLanguageModel.from_pretrained(model_path)
        
        # Load tokenizer
        logger.info(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # Load f-gram tokenizer
        logger.info(f"Loading f-gram tokenizer from {f_gram_tokenizer_path}")
        f_gram_tokenizer = FGramTokenizer.from_pretrained(f_gram_tokenizer_path)
        
        # Load embedding cache
        logger.info(f"Loading embedding cache from {embedding_cache_path}")
        embedding_cache = EmbeddingCache.load(
            embedding_cache_path,
            f_gram_tokenizer.n_gram_extractor,
            use_memory_map=use_memory_map,
        )
        
        return cls(
            model=model,
            tokenizer=tokenizer,
            f_gram_tokenizer=f_gram_tokenizer,
            embedding_cache=embedding_cache,
            device=device,
            quantization=quantization,
        )
    
    def generate(
        self,
        text: str,
        max_length: int = 50,
        min_length: int = 0,
        do_sample: bool = True,
        num_beams: int = 1,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        num_return_sequences: int = 1,
    ) -> List[str]:
        """Generate text using the model.
        
        Args:
            text: Input text.
            max_length: Maximum length of generated text.
            min_length: Minimum length of generated text.
            do_sample: Whether to use sampling.
            num_beams: Number of beams for beam search.
            temperature: Temperature for sampling.
            top_k: Top-k sampling parameter.
            top_p: Top-p sampling parameter.
            repetition_penalty: Repetition penalty.
            num_return_sequences: Number of sequences to return.
            
        Returns:
            List of generated texts.
        """
        # Tokenize input text
        encoding = self.f_gram_tokenizer.tokenize(
            text=text,
            return_f_grams=True,
            max_length=None,
            padding=False,
            truncation=False,
        )
        
        input_ids = encoding["input_ids"]
        token_f_grams = encoding["token_f_grams"]
        
        # Get f-gram embeddings
        token_embeddings = {}
        for pos, f_grams in token_f_grams.items():
            if not f_grams:
                continue
            
            # Get f-gram IDs
            f_gram_ids = [
                self.f_gram_tokenizer.n_gram_extractor.f_gram_to_id[f_gram]
                for f_gram in f_grams
            ]
            
            # Get embeddings
            embeddings = self.embedding_cache.get_embeddings(f_gram_ids, self.device)
            
            # Average embeddings
            token_embeddings[pos] = embeddings.mean(dim=0)
        
        # Create f-gram embeddings tensor
        f_gram_embeddings = torch.zeros(
            (1, len(input_ids), self.model.config.hidden_size),
            device=self.device,
        )
        
        for pos, embedding in token_embeddings.items():
            f_gram_embeddings[0, pos] = embedding
        
        # Convert input_ids to tensor
        input_ids = torch.tensor([input_ids], device=self.device)
        
        # Apply half precision for fp16 quantization
        if self.quantization == "fp16":
            f_gram_embeddings = f_gram_embeddings.half()
        
        # Generate text
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                f_gram_embeddings=f_gram_embeddings,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
                num_beams=num_beams,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                num_return_sequences=num_return_sequences,
            )
        
        # Decode output
        output_texts = [
            self.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in output_ids
        ]
        
        return output_texts
    
    def benchmark_inference(
        self,
        text: str,
        max_length: int = 50,
        num_runs: int = 10,
        warmup_runs: int = 2,
    ) -> Dict[str, float]:
        """Benchmark inference speed.
        
        Args:
            text: Input text.
            max_length: Maximum length of generated text.
            num_runs: Number of runs to average over.
            warmup_runs: Number of warmup runs.
            
        Returns:
            Dictionary of benchmark metrics.
        """
        import time
        
        # Tokenize input text
        encoding = self.f_gram_tokenizer.tokenize(
            text=text,
            return_f_grams=True,
            max_length=None,
            padding=False,
            truncation=False,
        )
        
        input_ids = encoding["input_ids"]
        token_f_grams = encoding["token_f_grams"]
        
        # Get f-gram embeddings
        token_embeddings = {}
        for pos, f_grams in token_f_grams.items():
            if not f_grams:
                continue
            
            # Get f-gram IDs
            f_gram_ids = [
                self.f_gram_tokenizer.n_gram_extractor.f_gram_to_id[f_gram]
                for f_gram in f_grams
            ]
            
            # Get embeddings
            embeddings = self.embedding_cache.get_embeddings(f_gram_ids, self.device)
            
            # Average embeddings
            token_embeddings[pos] = embeddings.mean(dim=0)
        
        # Create f-gram embeddings tensor
        f_gram_embeddings = torch.zeros(
            (1, len(input_ids), self.model.config.hidden_size),
            device=self.device,
        )
        
        for pos, embedding in token_embeddings.items():
            f_gram_embeddings[0, pos] = embedding
        
        # Convert input_ids to tensor
        input_ids = torch.tensor([input_ids], device=self.device)
        
        # Apply half precision for fp16 quantization
        if self.quantization == "fp16":
            f_gram_embeddings = f_gram_embeddings.half()
        
        # Warmup runs
        for _ in range(warmup_runs):
            with torch.no_grad():
                self.model.generate(
                    input_ids=input_ids,
                    f_gram_embeddings=f_gram_embeddings,
                    max_length=max_length,
                    do_sample=False,
                )
        
        # Timed runs
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(num_runs):
            with torch.no_grad():
                self.model.generate(
                    input_ids=input_ids,
                    f_gram_embeddings=f_gram_embeddings,
                    max_length=max_length,
                    do_sample=False,
                )
            
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        # Calculate metrics
        total_time = end_time - start_time
        avg_time = total_time / num_runs
        tokens_per_second = max_length / avg_time
        
        return {
            "total_time_seconds": total_time,
            "avg_time_seconds": avg_time,
            "tokens_per_second": tokens_per_second,
            "quantization": self.quantization,
        } 