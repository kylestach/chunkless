from functools import partial
from typing import Any, Literal, Mapping, Sequence, Tuple
import flax.linen as nn
from einops import rearrange, einsum, repeat
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
import optax

from chunkless.diffusion_policy import make_optimizer
from chunkless.normalizers import Normalizer


def _apply_rope(x, *, positions, max_wavelength=10_000):
    """Applies RoPE positions [B, L] to x [B, L, H, D]."""
    freq_exponents = (2.0 / x.shape[-1]) * jnp.arange(x.shape[-1] // 2)
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None] / timescale[None, None, :]
    radians = radians[..., None, :]
    # radians.shape = [...,L,1,d=D/2]
    sin, cos = jnp.sin(radians), jnp.cos(radians)
    x1, x2 = jnp.split(x, 2, axis=-1)
    res = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
    return res


def _apply_alibi(attn_logits):
    # Weights are [B, H, T, S]
    batch_size, num_heads, seq_len, _ = attn_logits.shape
    alibi_coeffs = jnp.logspace(-8, -1, num_heads, base=2)[:, None, None]
    distances = jnp.abs(jnp.arange(seq_len)[:, None] - jnp.arange(seq_len)[None, :])
    return attn_logits - alibi_coeffs * distances


class Attention(nn.Module):
    num_heads: int
    dropout_rate: float = 0.1
    alibi: bool = False

    @nn.compact
    def __call__(self, x, mask, cache=None, deterministic=False):
        seq_len = x.shape[-2]
        x = nn.Dense(features=x.shape[-1] * 3)(x)
        q, k, v = rearrange(x, "b t (qkv h d) -> qkv b t h d", qkv=3, h=self.num_heads)

        if cache is not None:
            cache_start_index = cache["index"]
            k = cache["key"] = jax.lax.dynamic_update_slice(
                cache["key"], k, (0, cache_start_index, 0, 0)
            )
            v = cache["value"] = jax.lax.dynamic_update_slice(
                cache["value"], v, (0, cache_start_index, 0, 0)
            )
            cache["index"] = cache_start_index + x.shape[-2]
            q_index = cache_start_index + jnp.arange(seq_len)
            k_index = jnp.arange(k.shape[1])
        else:
            q_index = jnp.arange(seq_len)
            k_index = jnp.arange(seq_len)

        # Apply RoPE
        q = _apply_rope(q, positions=q_index)
        k = _apply_rope(k, positions=k_index)

        attn_logits = einsum(q, k, "b t h d, b s h d -> b h t s")
        if mask is not None:
            attn_logits = jnp.where(mask, attn_logits, -jnp.inf)

        if self.alibi:
            attn_logits = _apply_alibi(attn_logits)
        attn_weights = jax.nn.softmax(attn_logits, axis=-1)
        attn_weights = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(
            attn_weights
        )

        self.sow("intermediates", "attn_weights", attn_weights, reduce_fn=lambda _, y: y)

        out = einsum(attn_weights, v, "b h t s, b s h d -> b t h d")
        return rearrange(out, "b t h d -> b t (h d)"), cache

    def make_cache(self, x):
        return {
            "key": jnp.zeros(
                (x.shape[0], x.shape[1], self.num_heads, x.shape[-1] // self.num_heads),
                dtype=x.dtype,
            ),
            "value": jnp.zeros(
                (x.shape[0], x.shape[1], self.num_heads, x.shape[-1] // self.num_heads),
                dtype=x.dtype,
            ),
            "index": jnp.array(0, dtype=jnp.int32),
        }


class TransformerMlp(nn.Module):
    mlp_ratio: float = 4.0
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, deterministic=True):
        features = x.shape[-1]
        x = nn.LayerNorm()(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(x)
        x = nn.Dense(features=features * self.mlp_ratio)(x)
        x = nn.GeGLU()(x)
        x = nn.Dense(features=features)(x)
        return x


class TransformerBlock(nn.Module):
    num_heads: int
    mlp_ratio: float = 4.0
    dropout_rate: float = 0.1
    alibi: bool = False

    def setup(self, deterministic=True):
        self.attn = Attention(num_heads=self.num_heads, dropout_rate=self.dropout_rate, alibi=self.alibi)
        self.mlp = TransformerMlp(
            mlp_ratio=self.mlp_ratio, dropout_rate=self.dropout_rate
        )
        self.attn_ln = nn.LayerNorm()
        self.mlp_ln = nn.LayerNorm()

    @nn.compact
    def __call__(self, x, mask, cache=None, deterministic=True):
        skip = x
        x = self.attn_ln(x)
        x, cache = self.attn(x, mask, cache, deterministic=deterministic)
        x = x + skip

        skip = x
        x = self.mlp_ln(x)
        x = self.mlp(x, deterministic=deterministic)
        x = x + skip
        return x, cache

    def make_cache(self, x):
        return self.attn.make_cache(x)


class Transformer(nn.Module):
    d_embed: int
    num_heads: int
    mlp_ratio: float = 4.0
    num_layers: int = 12
    dropout_rate: float = 0.1
    alibi: bool = False

    def setup(self):
        self.layers = [
            TransformerBlock(
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                dropout_rate=self.dropout_rate,
                alibi=self.alibi,
            )
            for _ in range(self.num_layers)
        ]

    def __call__(self, x, cache=None, deterministic=True):
        if cache is None:
            cache = [None] * self.num_layers

        mask = nn.make_causal_mask(x[..., 0])

        for i, layer in enumerate(self.layers):
            x, cache[i] = layer(x, mask, cache[i], deterministic=deterministic)

        return x, cache

    def make_cache(self, x):
        return [layer.make_cache(x) for layer in self.layers]


class CausalPolicyModel(nn.Module):
    action_vocab_size: int
    state_keys: Sequence[str]

    d_embed: int
    num_heads: int
    mlp_ratio: float = 4.0
    num_layers: int = 12
    num_obs_tokens: int = 1
    action_condition_discrete: bool = False
    hl_gauss_std: float = 0.005
    dropout_rate: float = 0.1
    alibi: bool = False

    def setup(self):
        self.proj_obs = nn.Dense(features=self.d_embed * self.num_obs_tokens)
        if self.action_condition_discrete:
            self.action_embed = nn.Embed(
                num_embeddings=self.action_vocab_size, features=self.d_embed
            )
        else:
            self.proj_action_dim = nn.Dense(features=self.d_embed)
        self.transformer = Transformer(
            d_embed=self.d_embed,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            num_layers=self.num_layers,
            dropout_rate=self.dropout_rate,
            alibi=self.alibi,
        )
        self.action_logits = nn.Dense(features=self.action_vocab_size)
        self.start_action = self.param(
            "start_action", lambda k: jnp.zeros(self.d_embed)
        )

    def __call__(
        self, *, obs=None, action=None, start_token=None, cache=None, init_cache=False, deterministic=True
    ):
        embeds = []
        action_start_idx = 0
        history_len = 0
        remove_last = 0

        if obs is not None:
            obs = jnp.concatenate([obs[k] for k in self.state_keys], axis=-1)
            embeds.append(
                rearrange(
                    self.proj_obs(obs), "b t (k d) -> b t k d", k=self.num_obs_tokens
                )
            )
            action_start_idx = self.num_obs_tokens
            history_len = obs.shape[1]

        if start_token is not None:
            embeds.append(
                repeat(
                    self.start_action,
                    "d -> b t 1 d",
                    b=obs.shape[0] if obs is not None else 1,
                    t=obs.shape[1] if obs is not None else 1,
                )
            )
        if action is not None:
            if self.action_condition_discrete:
                action = jnp.digitize(
                    jnp.clip(action, -1, 1), jnp.linspace(-1, 1, self.action_vocab_size)
                )
                action_embed = self.action_embed(action[..., None])
            else:
                action_embed = self.proj_action_dim(action[..., None])
            # Add dummy action tokens if necessary to match history length
            if action_embed.shape[1] == history_len - 1:
                remove_last = action_embed.shape[2]
                action_embed = jnp.concatenate(
                    [
                        action_embed,
                        jnp.zeros(
                            (action_embed.shape[0], 1, *action_embed.shape[2:]),
                            action_embed.dtype,
                        ),
                    ],
                    axis=1,
                )
            embeds.append(action_embed)

        embeds = jnp.concat(embeds, axis=-2)
        tokens_per_step = embeds.shape[-2]
        embeds = rearrange(embeds, "b t k d -> b (t k) d")
        embeds = embeds[..., : embeds.shape[-2] - remove_last, :]

        if cache is None and init_cache:
            cache = self.make_cache(embeds)

        outs, cache = self.transformer(embeds, cache=cache, deterministic=deterministic)

        outs = jnp.concatenate([outs, jnp.zeros_like(outs[:, :remove_last])], axis=1)
        outs = rearrange(outs, "b (t k) d -> b t k d", k=tokens_per_step)
        outs = outs[..., action_start_idx:, :]

        outs = self.action_logits(outs)
        outs = jax.nn.log_softmax(outs, axis=-1)

        return outs, cache

    def loss(self, *, batch):
        pred_logits, _ = self(
            obs=batch, action=batch["action"], start_token=True, deterministic=False
        )
        pred_logits = pred_logits[..., :-1, :]
        action_targets = batch["action"]
        action_target_cdf = jax.scipy.stats.norm.cdf(
            jnp.linspace(
                -1 + 1 / (pred_logits.shape[-1] - 1),
                1 - 1 / (pred_logits.shape[-1] - 1),
                pred_logits.shape[-1] - 1,
            ),
            loc=action_targets[..., None],
            scale=self.hl_gauss_std,
        )
        action_target_cdf = jnp.concatenate(
            [
                jnp.zeros_like(action_target_cdf[..., :1]),
                action_target_cdf,
                jnp.ones_like(action_target_cdf[..., -1:]),
            ],
            axis=-1,
        )
        action_target_pdf = jnp.diff(action_target_cdf, axis=-1)
        loss = -jnp.sum(pred_logits * action_target_pdf, axis=-1)
        return loss

    def make_cache(self, x):
        return self.transformer.make_cache(x)


def _predict_fn(
    *,
    network,
    params,
    action_dim,
    obs_history,
    action_history,
    rng,
    temperature=None,
    start_index=None,
):
    def select(logits, rng, temperature=None):
        logits = logits[:, -1:, :1, :]
        if temperature is None:
            action_token = jnp.argmax(logits, axis=-1)
        else:
            rng, key = jax.random.split(rng)
            action_token = jax.random.categorical(key, logits / temperature, axis=-1)
        return action_token, rng

    obs_tokens = network.num_obs_tokens
    if start_index is None:
        start_index = obs_history.shape[1] - 1
    start_index = (
        obs_tokens + 1 + (action_history.shape[-1] + obs_tokens + 1) * start_index
    )

    # Prefill
    logits, cache = network.apply(
        {"params": params},
        action=action_history,
        obs=obs_history,
        start_token=True,
        cache=None,
        init_cache=True,
    )
    cache = jax.tree.map(
        lambda x: {**x, "index": start_index},
        cache,
        is_leaf=lambda x: isinstance(x, dict) and "index" in x,
    )
    action_token_values = jnp.linspace(-1, 1, logits.shape[-1])

    action_token, rng = select(logits, rng, temperature=temperature)
    next_action = action_token_values[action_token]

    pred_actions = [next_action]
    for _ in range(action_dim - 1):
        logits, cache = network.apply(
            {"params": params}, action=next_action, cache=cache
        )
        action_token, rng = select(logits, rng, temperature=temperature)
        next_action = action_token_values[action_token]
        pred_actions.append(next_action)

    return jnp.squeeze(jnp.stack(pred_actions, axis=-1), axis=(1, 2))


def causal_train_step(train_state, batch, rng):
    def loss_fn(params):
        loss = train_state.apply_fn(
            {"params": params}, batch=batch, method="loss", rngs={"dropout": rng}
        )
        return loss.mean()

    loss, grads = jax.value_and_grad(loss_fn)(train_state.params)
    train_state = train_state.apply_gradients(grads=grads)

    metrics = {
        "loss": loss,
        "learning_rate": train_state.opt_state.hyperparams["learning_rate"],
        "ema_rate": train_state.opt_state.hyperparams["ema_rate"],
        "grad_norm": optax.global_norm(grads),
    }
    return train_state, metrics


class CausalPolicy:
    def __init__(
        self,
        *,
        obs_type: Literal["lowdim", "image"],
        abstract_obs: Any,
        abstract_action: jax.ShapeDtypeStruct,
        seed: int = 0,
        optimizer_config: dict,
        stats: Mapping,
        normalize_rules: Sequence[Tuple[str, Literal["mean_std", "min_max", "none"]]],
        state_keys: Sequence[str] | None = None,
        model_kwargs: dict = {},
    ):
        self._normalizer = Normalizer(stats, normalize_rules)
        rng = jax.random.PRNGKey(seed)

        rng, params_rng = jax.random.split(rng)
        network = CausalPolicyModel(
            action_vocab_size=512,
            state_keys=state_keys,
            **model_kwargs,
            # d_embed=d_embed,
            # num_heads=num_heads,
            # mlp_ratio=mlp_ratio,
            # num_layers=num_layers,
            # num_obs_tokens=num_obs_tokens,
            # action_condition_discrete=action_condition_discrete,
            # hl_gauss_std=hl_gauss_std,
        )
        network_params = jax.jit(network.init)(
            params_rng,
            obs=abstract_obs,
            action=abstract_action,
            start_token=True,
            cache=None,
        )["params"]

        self._network = network

        self._train_state = TrainState.create(
            apply_fn=self._network.apply,
            params=network_params,
            tx=make_optimizer(**optimizer_config),
        )
        self._predict = jax.jit(
            partial(
                _predict_fn, network=self._network, action_dim=abstract_action.shape[-1]
            )
        )
        self._rng = rng
        self._jit_train_step = jax.jit(causal_train_step)
        self._num_history_steps = jax.tree.leaves(abstract_obs)[0].shape[1]

    def train_step(self, batch):
        batch = self._normalizer.normalize(batch)
        self._rng, subrng = jax.random.split(self._rng)
        self._train_state, metrics = self._jit_train_step(
            self._train_state, batch, subrng
        )
        return metrics

    def predict(self, obs, *, use_ema, action_history, temperature=None):
        obs = self._normalizer.normalize(obs)
        action_history = self._normalizer.normalize(action_history, key="action")
        batch_size = jax.tree.leaves(obs)[0].shape[0]
        history_length = jax.tree.leaves(obs)[0].shape[1]

        assert (
            history_length == action_history.shape[1] + 1
        ), f"Obs history length {history_length} must be one more than action history shape {action_history.shape}"

        # Pad to max history length
        obs = jax.tree.map(
            lambda x: jnp.pad(
                x, ((0, 0), (0, self._num_history_steps - x.shape[1]), (0, 0))
            ),
            obs,
        )
        action_history = jnp.pad(
            action_history,
            (
                (0, 0),
                (0, self._num_history_steps - action_history.shape[1] - 1),
                (0, 0),
            ),
        )

        if use_ema:
            params = self._train_state.opt_state.inner_state["ema"].ema_params
        else:
            params = self.train_state.params

        self._rng, subrng = jax.random.split(self._rng)

        action = self._predict(
            obs_history=obs,
            action_history=action_history,
            params=params,
            rng=subrng,
            temperature=temperature,
            start_index=history_length - 1,
        )
        action = jax.device_get(action)

        action = self._normalizer.unnormalize(action, key="action")
        return action
