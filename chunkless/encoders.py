from typing import Sequence
import einops
import flax.linen as nn
import jax.numpy as jnp

class Encoder(nn.Module):
    out_features: int

class MLPEncoder(Encoder):
    state_keys: str | Sequence[str] = "state"

    num_layers: int = 2
    hidden_features: int = 256

    norm: str | None = "LayerNorm"
    activation: str | None = "relu"

    @nn.compact
    def __call__(self, x):
        if isinstance(self.state_keys, str):
            x = x[self.state_keys]
        else:
            x = jnp.concatenate([x[k] for k in self.state_keys], axis=-1)

        batch_size, history_len, state_dim = x.shape

        activation = getattr(nn, self.activation)
        norm = getattr(nn, self.norm)

        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden_features)(x)
            x = norm()(x)
            x = activation(x)
        
        assert self.out_features % history_len == 0, f"out_features must be divisible by history_len but instead got {self.out_features} % {history_len}"
        x = nn.Dense(self.out_features // history_len)(x)

        x = einops.rearrange(x, "... t d -> ... (t d)")
        return x
