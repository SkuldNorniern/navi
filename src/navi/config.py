"""Configuration for Navi LLM model."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class NaviConfig:
    """Configuration for the Navi language model."""

    # Model dimensions
    d_model: int = 2560  # Hidden dimension
    n_layers: int = 28   # Number of transformer layers
    n_heads: int = 20    # Number of attention heads
    head_dim: int = 128  # Dimension per attention head
    n_kv_heads: int = 4  # Number of key/value heads for GQA (groups of 5 query heads per KV head)

    # Feed-forward network
    d_ff: int = 4 * 2560  # FFN hidden dimension (4x d_model)

    # Vocabulary
    vocab_size: int = 64000  # Vocabulary size
    max_seq_len: int = 8192  # Maximum sequence length

    # Training parameters
    learning_rate: float = 3e-4
    batch_size: int = 8
    gradient_clip_norm: float = 1.0

    # Dropout rates
    dropout_rate: float = 0.1
    attention_dropout: float = 0.1

    # Normalization
    rms_norm_eps: float = 1e-6

    # RoPE parameters
    rope_theta: float = 10000.0

    def __post_init__(self):
        """Validate configuration parameters."""
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads for GQA"
        assert self.head_dim == self.d_model // self.n_heads, "head_dim should equal d_model // n_heads"


# Predefined configurations
def get_navi_config(model_size: str = "2.7B") -> NaviConfig:
    """Get configuration for different model sizes."""
    if model_size == "2.7B":
        return NaviConfig(
            d_model=2560,
            n_layers=28,
            n_heads=20,
            n_kv_heads=4,
        )
    elif model_size == "2.9B":
        return NaviConfig(
            d_model=2560,  # Keep same as 2.7B for compatibility
            n_layers=32,   # More layers for larger model
            n_heads=20,
            n_kv_heads=4,
        )
    elif model_size == "1.5B":
        return NaviConfig(
            d_model=1536,
            n_layers=16,
            n_heads=12,
            n_kv_heads=4,
        )
    else:
        raise ValueError(f"Unknown model size: {model_size}")


# Default configuration
DEFAULT_CONFIG = get_navi_config("1.5B")
