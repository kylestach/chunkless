from flax import linen as nn
import jax.numpy as jnp
import jax


class TransformerBlock(nn.Module):
    @nn.compact
    def __call__(self, x, mask):
        skip = x

        x = nn.LayerNorm()(x)
        x = nn.MultiHeadDotProductAttention(num_heads=8, dtype=jnp.float32)(
            x, x, x, mask
        )

        x = x + skip
        skip = x

        x = nn.LayerNorm()(x)
        x = nn.Dense(features=x.shape[-1])(x)
        x = nn.GeGLU()(x)
        x = nn.Dense(features=x.shape[-1])(x)

        x = x + skip

        return x


class TransformerPolicy(nn.Module):
    @nn.compact
    def __call__(self, x):
        return x
