"""Navi Language Model implementation."""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional, Tuple

from .config import NaviConfig
from .module import TransformerBlock, RMSNorm


class NaviModel(nn.Module):
    """Navi decoder-only language model."""
    config: NaviConfig

    def setup(self):
        self.vocab_size = self.config.vocab_size
        self.d_model = self.config.d_model
        self.n_layers = self.config.n_layers
        self.max_seq_len = self.config.max_seq_len

        # Input embedding layer (tied with output)
        self.embed_tokens = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.d_model,
            name="embed_tokens"
        )

        # Transformer blocks
        self.layers = [
            TransformerBlock(
                d_model=self.config.d_model,
                n_heads=self.config.n_heads,
                n_kv_heads=self.config.n_kv_heads,
                head_dim=self.config.head_dim,
                d_ff=self.config.d_ff,
                max_seq_len=self.config.max_seq_len,
                dropout_rate=self.config.dropout_rate,
                rms_norm_eps=self.config.rms_norm_eps,
                name=f"layer_{i}"
            )
            for i in range(self.n_layers)
        ]

        # Final normalization layer
        self.norm = RMSNorm(eps=self.config.rms_norm_eps, name="norm")

        # Output projection (tied with embeddings)
        self.lm_head = self.embed_tokens  # Weight tying

    def __call__(
        self,
        input_ids: jnp.ndarray,
        training: bool = False,
        attention_mask: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """
        Forward pass through the Navi model.

        Args:
            input_ids: Token indices of shape (batch_size, seq_len)
            training: Whether in training mode
            attention_mask: Optional attention mask

        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape

        # Input embeddings
        x = self.embed_tokens(input_ids)  # (batch_size, seq_len, d_model)

        # Create causal mask for autoregressive generation
        causal_mask = self._create_causal_mask(seq_len)

        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand attention_mask to match causal_mask dimensions
            attention_mask = attention_mask[:, None, None, :]  # (batch, 1, 1, seq)
            causal_mask = jnp.logical_and(causal_mask, attention_mask)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, mask=causal_mask, training=training)

        # Final normalization
        x = self.norm(x)

        # Language model head (tied embeddings)
        logits = self.lm_head.attend(x)  # (batch_size, seq_len, vocab_size)

        return logits

    def _create_causal_mask(self, seq_len: int) -> jnp.ndarray:
        """Create causal attention mask for autoregressive generation."""
        # Create lower triangular mask (causal)
        mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
        # Expand for batch and heads dimensions
        mask = mask[None, None, :, :]  # (1, 1, seq_len, seq_len)
        return mask

    def generate(
        self,
        input_ids: jnp.ndarray,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        rng_key: Optional[jax.random.PRNGKey] = None
    ) -> jnp.ndarray:
        """
        Generate text autoregressively.

        Args:
            input_ids: Initial token sequence
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            rng_key: Random key for sampling

        Returns:
            Generated token sequence
        """
        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)

        generated = input_ids

        for _ in range(max_new_tokens):
            # Get logits for the last token
            logits = self(generated, training=False)
            next_token_logits = logits[:, -1, :]  # (batch_size, vocab_size)

            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Apply top-k sampling
            if top_k is not None:
                top_k_logits, top_k_indices = jax.lax.top_k(next_token_logits, top_k)
                next_token_logits = jnp.where(
                    next_token_logits >= top_k_logits[:, -1:],
                    next_token_logits,
                    -float('inf')
                )

            # Apply top-p (nucleus) sampling
            if top_p is not None:
                sorted_logits = jnp.sort(next_token_logits, descending=True)
                sorted_probs = jax.nn.softmax(sorted_logits)
                cumulative_probs = jnp.cumsum(sorted_probs, axis=-1)

                # Find cutoff
                cutoff_mask = cumulative_probs > top_p
                cutoff_mask = jnp.roll(cutoff_mask, 1, axis=-1)
                cutoff_mask = cutoff_mask.at[:, 0].set(False)

                # Apply cutoff
                next_token_logits = jnp.where(cutoff_mask, -float('inf'), next_token_logits)

            # Sample next token
            rng_key, subkey = jax.random.split(rng_key)
            probs = jax.nn.softmax(next_token_logits)
            next_token = jax.random.categorical(subkey, logits=next_token_logits)

            # Append to sequence
            next_token = next_token[:, None]  # Add sequence dimension
            generated = jnp.concatenate([generated, next_token], axis=-1)

        return generated


def create_navi_model(config: NaviConfig) -> NaviModel:
    """Factory function to create a Navi model with given configuration."""
    return NaviModel(config=config)

