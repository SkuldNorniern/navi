import math
import jax
import jax.numpy as jnp
from flax import linen as nn


# Norm Layer
class RMSNorm(nn.Module):
    eps: float = 1e-6
    @nn.compact
    def __call__(self, x):
        scale = self.param("scale", nn.initializers.ones, (x.shape[-1],))
        ms = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(ms + self.eps)
        return scale * x

# Act Function
class SwiGLU(nn.Module):
    d_ff: int
    @nn.compact
    def __call__(self, x):
        w12 = nn.Dense(2 * self.d_ff, use_bias=False, name="w12")(x)
        g, u = jnp.split(w12, 2, axis=-1)
        y = nn.Dense(x.shape[-1], use_bias=False, name="w_out")(nn.silu(g) * u)
        return y


# Rotary Position Embedding (RoPE)
class RoPE(nn.Module):
    """Rotary Position Embedding for attention."""
    head_dim: int
    max_seq_len: int = 8192
    theta: float = 10000.0

    def setup(self):
        # Precompute frequencies for RoPE
        freqs = 1.0 / (self.theta ** (jnp.arange(0, self.head_dim, 2).astype(jnp.float32) / self.head_dim))
        self.freqs = freqs

    def __call__(self, x: jnp.ndarray, seq_len: Optional[int] = None) -> jnp.ndarray:
        """Apply RoPE to input tensor."""
        if seq_len is None:
            seq_len = x.shape[-3]  # Assume shape is (batch, seq, heads, head_dim)

        # Create position indices
        positions = jnp.arange(seq_len)

        # Compute cos and sin for RoPE
        freqs = jnp.outer(positions, self.freqs)
        cos = jnp.cos(freqs)
        sin = jnp.sin(freqs)

        # Reshape for broadcasting
        cos = cos[:, :, None]  # (seq_len, head_dim//2, 1)
        sin = sin[:, :, None]  # (seq_len, head_dim//2, 1)

        # Split input into even and odd dimensions
        x_even = x[..., ::2]  # (..., seq_len, head_dim//2)
        x_odd = x[..., 1::2]  # (..., seq_len, head_dim//2)

        # Apply RoPE rotation
        x_rotated = jnp.concatenate([
            x_even * cos - x_odd * sin,
            x_even * sin + x_odd * cos
        ], axis=-1)

        return x_rotated


# Grouped Query Attention (GQA)
class GQA(nn.Module):
    """Grouped Query Attention with RoPE."""
    d_model: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    max_seq_len: int = 8192
    dropout_rate: float = 0.1

    def setup(self):
        self.n_groups = self.n_heads // self.n_kv_heads

        # Query, Key, Value projections
        self.q_proj = nn.Dense(self.n_heads * self.head_dim, use_bias=False, name="q_proj")
        self.k_proj = nn.Dense(self.n_kv_heads * self.head_dim, use_bias=False, name="k_proj")
        self.v_proj = nn.Dense(self.n_kv_heads * self.head_dim, use_bias=False, name="v_proj")
        self.out_proj = nn.Dense(self.d_model, use_bias=False, name="out_proj")

        # RoPE for position encoding
        self.rope = RoPE(head_dim=self.head_dim, max_seq_len=self.max_seq_len)

        # Dropout
        self.dropout = nn.Dropout(rate=self.dropout_rate)

    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None, training: bool = False):
        batch_size, seq_len, _ = x.shape

        # Project to queries, keys, values
        q = self.q_proj(x).reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # Apply RoPE to queries and keys
        q = self.rope(q)
        k = self.rope(k)

        # Repeat KV heads for GQA
        if self.n_groups > 1:
            k = jnp.repeat(k, self.n_groups, axis=-2)  # (batch, seq, n_heads, head_dim)
            v = jnp.repeat(v, self.n_groups, axis=-2)  # (batch, seq, n_heads, head_dim)

        # Transpose for attention computation
        q = q.transpose(0, 2, 1, 3)  # (batch, n_heads, seq, head_dim)
        k = k.transpose(0, 2, 1, 3)  # (batch, n_heads, seq, head_dim)
        v = v.transpose(0, 2, 1, 3)  # (batch, n_heads, seq, head_dim)

        # Compute attention scores
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = jnp.matmul(q, k.transpose(0, 1, 3, 2)) * scale  # (batch, n_heads, seq, seq)

        # Apply causal mask if provided
        if mask is not None:
            attn_weights = jnp.where(mask, attn_weights, -float('inf'))

        # Compute attention probabilities
        attn_probs = jax.nn.softmax(attn_weights, axis=-1)
        attn_probs = self.dropout(attn_probs, deterministic=not training)

        # Apply attention to values
        attn_output = jnp.matmul(attn_probs, v)  # (batch, n_heads, seq, head_dim)

        # Transpose back and reshape
        attn_output = attn_output.transpose(0, 2, 1, 3)  # (batch, seq, n_heads, head_dim)
        attn_output = attn_output.reshape(batch_size, seq_len, self.n_heads * self.head_dim)

        # Final projection
        output = self.out_proj(attn_output)

        return output


# Transformer Block
class TransformerBlock(nn.Module):
    """A single transformer block with attention and feed-forward layers."""
    d_model: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    d_ff: int
    max_seq_len: int = 8192
    dropout_rate: float = 0.1
    rms_norm_eps: float = 1e-6

    def setup(self):
        # Attention layer
        self.attention = GQA(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
            max_seq_len=self.max_seq_len,
            dropout_rate=self.dropout_rate
        )

        # Feed-forward network
        self.ffn = SwiGLU(d_ff=self.d_ff)

        # Normalization layers
        self.attn_norm = RMSNorm(eps=self.rms_norm_eps)
        self.ffn_norm = RMSNorm(eps=self.rms_norm_eps)

        # Dropouts
        self.attn_dropout = nn.Dropout(rate=self.dropout_rate)
        self.ffn_dropout = nn.Dropout(rate=self.dropout_rate)

    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None, training: bool = False):
        # Pre-norm attention
        attn_input = self.attn_norm(x)
        attn_output = self.attention(attn_input, mask=mask, training=training)
        attn_output = self.attn_dropout(attn_output, deterministic=not training)

        # Residual connection for attention
        x = x + attn_output

        # Pre-norm feed-forward
        ffn_input = self.ffn_norm(x)
        ffn_output = self.ffn(ffn_input)
        ffn_output = self.ffn_dropout(ffn_output, deterministic=not training)

        # Residual connection for FFN
        x = x + ffn_output

        return x
