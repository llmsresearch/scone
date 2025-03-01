"""Hydra configuration for SCONE."""

from dataclasses import dataclass, field
from typing import List, Optional, Union


@dataclass
class DataConfig:
    """Data configuration for SCONE."""
    
    dataset_name: str = "wikitext"
    dataset_config_name: str = "wikitext-103-v1"
    text_column: str = "text"
    max_length: int = 512
    max_n: int = 3
    min_freq: int = 100
    max_f_grams: int = 1000000


@dataclass
class ModelConfig:
    """Model configuration for SCONE."""
    
    base_model_name: str = "gpt2"
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 1024
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    use_f_gram_embeddings: bool = True


@dataclass
class TrainingConfig:
    """Training configuration for SCONE."""
    
    output_dir: str = "./outputs/scone-base"
    num_epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 0
    max_grad_norm: float = 1.0
    logging_steps: int = 100
    save_steps: int = 1000
    eval_steps: int = 1000
    seed: int = 42
    gradient_accumulation_steps: int = 1
    fp16: bool = False


@dataclass
class InferenceConfig:
    """Inference configuration for SCONE."""
    
    use_memory_map: bool = True
    batch_size: int = 32


@dataclass
class SconeConfig:
    """Configuration for SCONE."""
    
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig) 