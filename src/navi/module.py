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
