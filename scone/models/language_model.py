"""Language model with SCONE embeddings."""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig

from scone.models.f_gram_model import FGramModel


class SconeConfig(PretrainedConfig):
    """Configuration class for SconeLanguageModel.
    
    This class holds the configuration for the SCONE language model, which
    uses contextualized f-gram embeddings to enhance language model performance.
    
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
        f_gram_model_config: Configuration for the f-gram model.
        use_f_gram_embeddings: Whether to use f-gram embeddings.
    """
    
    model_type = "scone"
    
    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
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
        f_gram_model_config: Optional[Dict] = None,
        use_f_gram_embeddings: bool = True,
        **kwargs,
    ) -> None:
        """Initialize the SconeConfig.
        
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
            f_gram_model_config: Configuration for the f-gram model.
            use_f_gram_embeddings: Whether to use f-gram embeddings.
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
        self.f_gram_model_config = f_gram_model_config
        self.use_f_gram_embeddings = use_f_gram_embeddings


class SconeLanguageModel(PreTrainedModel):
    """Language model with SCONE embeddings.
    
    This model uses contextualized f-gram embeddings to enhance language model
    performance. It consists of a base language model and an f-gram model for
    generating contextualized embeddings.
    
    Attributes:
        config: Configuration for the model.
        base_model: Base language model.
        f_gram_model: Model for generating contextualized f-gram embeddings.
        f_gram_projection: Projection layer for f-gram embeddings.
        use_f_gram_embeddings: Whether to use f-gram embeddings.
    """
    
    config_class = SconeConfig
    base_model_prefix = "scone"
    
    def __init__(self, config: SconeConfig) -> None:
        """Initialize the SconeLanguageModel.
        
        Args:
            config: Configuration for the model.
        """
        super().__init__(config)
        
        # Import here to avoid circular imports
        from transformers import AutoModelForCausalLM, AutoConfig
        
        # Initialize base language model
        base_config = AutoConfig.from_pretrained(
            "gpt2",
            vocab_size=config.vocab_size,
            n_positions=config.max_position_embeddings,
            n_embd=config.hidden_size,
            n_layer=config.num_hidden_layers,
            n_head=config.num_attention_heads,
            resid_pdrop=config.hidden_dropout_prob,
            embd_pdrop=config.hidden_dropout_prob,
            attn_pdrop=config.attention_probs_dropout_prob,
            layer_norm_epsilon=config.layer_norm_eps,
        )
        
        self.base_model = AutoModelForCausalLM.from_config(base_config)
        
        # Initialize f-gram model if needed
        self.use_f_gram_embeddings = config.use_f_gram_embeddings
        self.f_gram_model = None
        self.f_gram_projection = None
        
        if self.use_f_gram_embeddings:
            from scone.models.f_gram_model import FGramConfig
            
            # Create f-gram model config if not provided
            if config.f_gram_model_config is None:
                f_gram_config = FGramConfig(
                    vocab_size=config.vocab_size,
                    hidden_size=config.hidden_size // 2,
                    num_hidden_layers=config.num_hidden_layers // 2,
                    num_attention_heads=config.num_attention_heads // 2,
                    intermediate_size=config.intermediate_size // 2,
                    hidden_act=config.hidden_act,
                    hidden_dropout_prob=config.hidden_dropout_prob,
                    attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                    max_position_embeddings=config.max_position_embeddings,
                    type_vocab_size=config.type_vocab_size,
                    initializer_range=config.initializer_range,
                    layer_norm_eps=config.layer_norm_eps,
                    embedding_size=config.hidden_size // 2,
                )
            else:
                f_gram_config = FGramConfig(**config.f_gram_model_config)
            
            # Initialize f-gram model
            self.f_gram_model = FGramModel(f_gram_config)
            
            # Initialize projection layer for f-gram embeddings
            self.f_gram_projection = nn.Linear(
                f_gram_config.hidden_size,
                config.hidden_size,
                bias=False,
            )
        
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
        f_gram_embeddings: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
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
            f_gram_embeddings: Pre-computed f-gram embeddings.
            labels: Labels for language modeling.
            output_attentions: Whether to output attentions.
            output_hidden_states: Whether to output hidden states.
            return_dict: Whether to return a dictionary.
            
        Returns:
            Dictionary containing:
                - loss: Language modeling loss.
                - logits: Logits for next token prediction.
                - hidden_states: Hidden states of the model.
                - attentions: Attention weights.
        """
        # Get f-gram embeddings if needed
        if self.use_f_gram_embeddings and f_gram_embeddings is None and f_gram_ids is not None:
            # Generate f-gram embeddings using f-gram model
            f_gram_outputs = self.f_gram_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                f_gram_ids=f_gram_ids,
                f_gram_attention_mask=f_gram_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
            
            f_gram_embeddings = f_gram_outputs["f_gram_embeddings"]
        
        # Project f-gram embeddings if needed
        if self.use_f_gram_embeddings and f_gram_embeddings is not None:
            f_gram_embeddings = self.f_gram_projection(f_gram_embeddings)
        
        # Get base model embeddings
        base_embeddings = self.base_model.transformer.wte(input_ids)
        
        # Combine base embeddings with f-gram embeddings if available
        if self.use_f_gram_embeddings and f_gram_embeddings is not None:
            combined_embeddings = base_embeddings + f_gram_embeddings
        else:
            combined_embeddings = base_embeddings
        
        # Add position embeddings
        if position_ids is None:
            position_ids = torch.arange(
                0, input_ids.size(1), dtype=torch.long, device=input_ids.device
            ).unsqueeze(0)
        
        position_embeddings = self.base_model.transformer.wpe(position_ids)
        embeddings = combined_embeddings + position_embeddings
        
        # Forward pass through base model
        outputs = self.base_model.transformer(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        
        # Get logits
        hidden_states = outputs.last_hidden_state
        logits = self.base_model.lm_head(hidden_states)
        
        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
        
        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": outputs.hidden_states,
            "attentions": outputs.attentions,
        }
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        f_gram_embeddings: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        do_sample: bool = False,
        early_stopping: bool = False,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        bad_words_ids: Optional[List[List[int]]] = None,
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        length_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        num_return_sequences: Optional[int] = None,
        decoder_start_token_id: Optional[int] = None,
        use_cache: Optional[bool] = None,
        **model_kwargs,
    ) -> torch.Tensor:
        """Generate text using the model.
        
        Args:
            input_ids: Token IDs.
            attention_mask: Attention mask.
            token_type_ids: Token type IDs.
            position_ids: Position IDs.
            f_gram_embeddings: Pre-computed f-gram embeddings.
            max_length: Maximum length of generated text.
            min_length: Minimum length of generated text.
            do_sample: Whether to use sampling.
            early_stopping: Whether to stop early.
            num_beams: Number of beams for beam search.
            temperature: Temperature for sampling.
            top_k: Top-k sampling parameter.
            top_p: Top-p sampling parameter.
            repetition_penalty: Repetition penalty.
            bad_words_ids: IDs of bad words to avoid.
            bos_token_id: ID of beginning of sentence token.
            pad_token_id: ID of padding token.
            eos_token_id: ID of end of sentence token.
            length_penalty: Length penalty.
            no_repeat_ngram_size: Size of n-grams to avoid repeating.
            num_return_sequences: Number of sequences to return.
            decoder_start_token_id: ID of decoder start token.
            use_cache: Whether to use cache.
            
        Returns:
            Generated token IDs.
        """
        # Store f-gram embeddings in model_kwargs
        model_kwargs["f_gram_embeddings"] = f_gram_embeddings
        
        # Generate text using base model
        return self.base_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            max_length=max_length,
            min_length=min_length,
            do_sample=do_sample,
            early_stopping=early_stopping,
            num_beams=num_beams,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            bad_words_ids=bad_words_ids,
            bos_token_id=bos_token_id,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            num_return_sequences=num_return_sequences,
            decoder_start_token_id=decoder_start_token_id,
            use_cache=use_cache,
            **model_kwargs,
        ) 