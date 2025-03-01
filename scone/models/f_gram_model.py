"""F-gram model for SCONE."""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig


class FGramConfig(PretrainedConfig):
    """Configuration class for FGramModel.
    
    This class holds the configuration for the f-gram model, which is used
    to generate contextualized embeddings for f-grams.
    
    Attributes:
        vocab_size: Size of the vocabulary.
        hidden_size: Size of the hidden layers.
        num_hidden_layers: Number of hidden layers.
        num_attention_heads: Number of attention heads.
        intermediate_size: Size of the intermediate layers.
        hidden_act: Activation function for hidden layers.
        hidden_dropout_prob: Dropout probability for hidden layers.
        attention_probs_dropout_prob: Dropout probability for attention probabilities.
        max_position_embeddings: Maximum sequence length.
        type_vocab_size: Size of the token type vocabulary.
        initializer_range: Range for weight initialization.
        layer_norm_eps: Epsilon for layer normalization.
        embedding_size: Size of the embeddings.
    """
    
    model_type = "f_gram"
    
    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        embedding_size: int = 768,
        **kwargs,
    ) -> None:
        """Initialize the FGramConfig.
        
        Args:
            vocab_size: Size of the vocabulary.
            hidden_size: Size of the hidden layers.
            num_hidden_layers: Number of hidden layers.
            num_attention_heads: Number of attention heads.
            intermediate_size: Size of the intermediate layers.
            hidden_act: Activation function for hidden layers.
            hidden_dropout_prob: Dropout probability for hidden layers.
            attention_probs_dropout_prob: Dropout probability for attention probabilities.
            max_position_embeddings: Maximum sequence length.
            type_vocab_size: Size of the token type vocabulary.
            initializer_range: Range for weight initialization.
            layer_norm_eps: Epsilon for layer normalization.
            embedding_size: Size of the embeddings.
        """
        super().__init__(**kwargs)
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.embedding_size = embedding_size


class FGramModel(PreTrainedModel):
    """Model for generating contextualized embeddings for f-grams.
    
    This model takes token sequences as input and generates contextualized
    embeddings for f-grams. It uses a transformer architecture to capture
    the context of each token.
    
    Attributes:
        config: Configuration for the model.
        transformer: Transformer model for generating contextualized embeddings.
        f_gram_embeddings: Embedding layer for f-grams.
    """
    
    config_class = FGramConfig
    base_model_prefix = "f_gram"
    
    def __init__(self, config: FGramConfig) -> None:
        """Initialize the FGramModel.
        
        Args:
            config: Configuration for the model.
        """
        super().__init__(config)
        
        # Import here to avoid circular imports
        from transformers import AutoModel
        
        # Use a smaller transformer model for generating contextualized embeddings
        self.transformer = AutoModel.from_pretrained(
            "bert-base-uncased",
            config=config,
            add_pooling_layer=False,
        )
        
        # Initialize f-gram embeddings
        self.f_gram_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Initialize weights
        self.init_weights()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        f_gram_ids: Optional[torch.Tensor] = None,
        f_gram_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the model.
        
        Args:
            input_ids: Token IDs.
            attention_mask: Attention mask.
            token_type_ids: Token type IDs.
            position_ids: Position IDs.
            f_gram_ids: F-gram IDs.
            f_gram_attention_mask: Attention mask for f-grams.
            output_attentions: Whether to output attentions.
            output_hidden_states: Whether to output hidden states.
            return_dict: Whether to return a dictionary.
            
        Returns:
            Dictionary containing:
                - last_hidden_state: Last hidden state of the transformer.
                - f_gram_embeddings: Embeddings for f-grams.
        """
        # Get contextualized token embeddings from transformer
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        
        # Get last hidden state
        last_hidden_state = transformer_outputs.last_hidden_state
        
        # If f-gram IDs are provided, get f-gram embeddings
        f_gram_embeddings = None
        if f_gram_ids is not None:
            f_gram_embeddings = self.f_gram_embeddings(f_gram_ids)
            
            # Apply attention mask if provided
            if f_gram_attention_mask is not None:
                f_gram_embeddings = f_gram_embeddings * f_gram_attention_mask.unsqueeze(-1)
        
        return {
            "last_hidden_state": last_hidden_state,
            "f_gram_embeddings": f_gram_embeddings,
        }
    
    def get_f_gram_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        f_gram_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        """Get embeddings for f-grams.
        
        Args:
            input_ids: Token IDs.
            attention_mask: Attention mask.
            token_type_ids: Token type IDs.
            position_ids: Position IDs.
            f_gram_ids: F-gram IDs.
            
        Returns:
            Embeddings for f-grams.
        """
        # Get contextualized token embeddings
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            f_gram_ids=f_gram_ids,
        )
        
        return outputs["f_gram_embeddings"] 