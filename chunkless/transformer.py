from typing import Literal, Optional, Callable, Any
import functools

from flax.nnx.nnx.module import first_from
import jax
from jax import Array, lax
import jax.numpy as jnp

import einops

from flax import nnx
from flax.typing import (
    Dtype,
    Shape,
    Initializer,
    PrecisionLike,
    DotGeneralT,
)
from flax.nnx.nnx.nn.attention import dot_product_attention_weights
from flax.nnx.nnx.nn.linear import default_kernel_init
import optax


AttentionType = Literal["dot_product", "rope", "alibi"]
PositionalEncodingType = Literal["sinusoidal", "learned", "rope", "alibi"]


def sin_embed_init(key, shape, dtype):
    # sin/cos positional encoding
    max_len, d_model = shape

    embed = jnp.zeros((max_len, d_model), dtype=dtype)
    pos = jnp.arange(max_len)[:, None]
    dim = jnp.arange(d_model // 2)[None, :]
    ts = 10000 ** -(2 * dim / d_model)
    embed = jnp.concatenate(
        [
            jnp.sin(pos * ts),
            jnp.cos(pos * ts),
        ],
        axis=-1,
    )
    return embed


def rope_attention_weights(
    query: Array,
    key: Array,
    **kwargs,
) -> Array:
    """Computes attention weights according to RoPE."""

    # rotate qkv
    def rotate(x):
        dim = x.shape[-1]
        idcs = jnp.arange(x.shape[-2])
        ts = 10000 ** -(2 * jnp.linspace(0, 1, x.shape[-1] // 2) / dim)
        ct = jnp.cos(ts * idcs[:, None])
        st = jnp.sin(ts * idcs[:, None])
        return jnp.concatenate(
            [
                x[..., : dim // 2] * ct - x[..., dim // 2 :] * st,
                x[..., : dim // 2] * st + x[..., dim // 2 :] * ct,
            ],
            axis=-1,
        )

    return dot_product_attention_weights(
        rotate(query),
        rotate(key),
        **kwargs,
    )


def rope_attention(
    query: Array,
    key: Array,
    value: Array,
    bias: Optional[Array] = None,
    mask: Optional[Array] = None,
    broadcast_dropout: bool = True,
    dropout_rng: Optional[Array] = None,
    dropout_rate: float = 0.0,
    deterministic: bool = False,
    dtype: Optional[Dtype] = None,
    precision: PrecisionLike = None,
    module: Optional[nnx.Module] = None,
):
    """Computes Rotary Positional Encoding (RoPE) from query, key, and value.

    See `nnx.dot_product_attention` for more details.
    """
    query, key, value = promote_dtype((query, key, value), dtype=dtype)  # type: ignore[bad-unpacking]
    dtype = query.dtype
    assert key.ndim == query.ndim == value.ndim, "q, k, v must have same rank."
    assert (
        query.shape[:-3] == key.shape[:-3] == value.shape[:-3]
    ), "q, k, v batch dims must match."
    assert (
        query.shape[-2] == key.shape[-2] == value.shape[-2]
    ), "q, k, v num_heads must match."
    assert key.shape[-3] == value.shape[-3], "k, v lengths must match."

    # compute attention weights
    attn_weights = rope_attention_weights(
        query,
        key,
        bias=bias,
        mask=mask,
        broadcast_dropout=broadcast_dropout,
        dropout_rng=dropout_rng,
        dropout_rate=dropout_rate,
        deterministic=deterministic,
        dtype=dtype,
        precision=precision,
        module=module,
    )

    # return weighted sum over values for each query position
    return jnp.einsum("...hqk,...khd->...qhd", attn_weights, value, precision=precision)


def alibi_attention(
    query: Array,
    key: Array,
    value: Array,
    bias: Optional[Array] = None,
    mask: Optional[Array] = None,
    broadcast_dropout: bool = True,
    dropout_rng: Optional[Array] = None,
    dropout_rate: float = 0.0,
    deterministic: bool = False,
    dtype: Optional[Dtype] = None,
    precision: PrecisionLike = None,
    module: Optional[nnx.Module] = None,
):
    """Computes Attention with Linear Biases (ALiBi) from query, key, and value.

    See `nnx.dot_product_attention` for more details.
    """
    num_heads, query_len, _ = query.shape[-3:]
    kv_len = key.shape[-2]

    q_index = jnp.arange(query_len)
    kv_index = jnp.arange(kv_len)
    alibi_base = q_index[..., None] - kv_index[..., None, :]

    alibi_slopes = 2 ** jnp.linspace(1, 8, num_heads)
    alibi_bias = alibi_slopes[:, None, None] * alibi_base[None, :, :]

    bias = bias + alibi_bias if bias is not None else alibi_bias

    return nnx.dot_product_attention(
        query,
        key,
        value,
        bias=bias,
        mask=mask,
        broadcast_dropout=broadcast_dropout,
        dropout_rng=dropout_rng,
        dropout_rate=dropout_rate,
        deterministic=deterministic,
        dtype=dtype,
        precision=precision,
        module=module,
    )


class MultiHeadAttention(nnx.Module):
    """
    Multi-head attention with KV cache and multi-step cache prefill (e.g. for prompts).
    """

    def __init__(
        self,
        num_heads: int,
        in_features: int,
        qkv_features: int | None = None,
        out_features: int | None = None,
        *,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        broadcast_dropout: bool = True,
        dropout_rate: float = 0.0,
        deterministic: bool | None = None,
        precision: PrecisionLike = None,
        kernel_init: Initializer = default_kernel_init,
        out_kernel_init: Initializer | None = None,
        bias_init: Initializer = nnx.initializers.zeros_init(),
        out_bias_init: Initializer | None = None,
        use_bias: bool = True,
        attention_fn: Callable[..., Array] = nnx.dot_product_attention,
        decode: bool | None = None,
        normalize_qk: bool = False,
        # Deprecated, will be removed.
        qkv_dot_general: DotGeneralT | None = None,
        out_dot_general: DotGeneralT | None = None,
        qkv_dot_general_cls: Any = None,
        out_dot_general_cls: Any = None,
        rngs: nnx.Rngs,
    ):
        self.num_heads = num_heads
        self.in_features = in_features
        self.qkv_features = qkv_features if qkv_features is not None else in_features
        self.out_features = out_features if out_features is not None else in_features
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.broadcast_dropout = broadcast_dropout
        self.dropout_rate = dropout_rate
        self.deterministic = deterministic
        self.precision = precision
        self.kernel_init = kernel_init
        self.out_kernel_init = out_kernel_init
        self.bias_init = bias_init
        self.out_bias_init = out_bias_init
        self.use_bias = use_bias
        self.attention_fn = attention_fn
        self.decode = decode
        self.normalize_qk = normalize_qk
        self.qkv_dot_general = qkv_dot_general
        self.out_dot_general = out_dot_general
        self.qkv_dot_general_cls = qkv_dot_general_cls
        self.out_dot_general_cls = out_dot_general_cls

        if self.qkv_features % self.num_heads != 0:
            raise ValueError(
                f"Memory dimension ({self.qkv_features}) must be divisible by "
                f"'num_heads' heads ({self.num_heads})."
            )

        self.head_dim = self.qkv_features // self.num_heads

        linear_general = functools.partial(
            nnx.LinearGeneral,
            in_features=self.in_features,
            out_features=(self.num_heads, self.head_dim),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            use_bias=self.use_bias,
            precision=self.precision,
            dot_general=self.qkv_dot_general,
            dot_general_cls=self.qkv_dot_general_cls,
        )
        # project inputs_q to multi-headed q/k/v
        # dimensions are then [batch..., length, n_heads, n_features_per_head]
        self.query = linear_general(rngs=rngs)
        self.key = linear_general(rngs=rngs)
        self.value = linear_general(rngs=rngs)

        self.query_ln: nnx.LayerNorm | None
        self.key_ln: nnx.LayerNorm | None
        if self.normalize_qk:
            # Normalizing query and key projections stabilizes training with higher
            # LR. See ViT-22B paper http://arxiv.org/abs/2302.05442 for analysis.
            self.query_ln = nnx.LayerNorm(
                self.head_dim,
                use_bias=False,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                rngs=rngs,
            )
            self.key_ln = nnx.LayerNorm(
                self.head_dim,
                use_bias=False,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                rngs=rngs,
            )
        else:
            self.query_ln = None
            self.key_ln = None

        self.out = nnx.LinearGeneral(
            in_features=(self.num_heads, self.head_dim),
            out_features=self.out_features,
            axis=(-2, -1),
            kernel_init=self.out_kernel_init or self.kernel_init,
            bias_init=self.out_bias_init or self.bias_init,
            use_bias=self.use_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            dot_general=self.out_dot_general,
            dot_general_cls=self.out_dot_general_cls,
            rngs=rngs,
        )
        self.rngs = rngs if dropout_rate > 0.0 else None

        self.cached_key: nnx.Cache[Array] | None = None
        self.cached_value: nnx.Cache[Array] | None = None
        self.cache_index: nnx.Cache[Array] | None = None

    def __call__(
        self,
        inputs_q: Array,
        inputs_k: Array | None = None,
        inputs_v: Array | None = None,
        *,
        mask: Array | None = None,
        deterministic: bool | None = None,
        rngs: nnx.Rngs | None = None,
        sow_weights: bool = False,
        decode: bool | None = None,
    ):
        """Applies multi-head dot product attention on the input data.

        Projects the inputs into multi-headed query, key, and value vectors,
        applies dot-product attention and project the results to an output vector.

        If both inputs_k and inputs_v are None, they will both copy the value of
        inputs_q (self attention).
        If only inputs_v is None, it will copy the value of inputs_k.

        Args:
          inputs_q: input queries of shape `[batch_sizes..., length, features]`.
          inputs_k: key of shape `[batch_sizes..., length, features]`. If None,
            inputs_k will copy the value of inputs_q.
          inputs_v: values of shape `[batch_sizes..., length, features]`. If None,
            inputs_v will copy the value of inputs_k.
          mask: attention mask of shape `[batch_sizes..., num_heads, query_length,
            key/value_length]`. Attention weights are masked out if their
            corresponding mask value is `False`.
          deterministic: if false, the attention weight is masked randomly using
            dropout, whereas if true, the attention weights are deterministic. The
            ``deterministic`` flag passed into the call method will take precedence
            over the ``deterministic`` flag passed into the constructor.
          rngs: rng key. The rng key passed into the call method will take
            precedence over the rng key passed into the constructor.
          sow_weights: if ``True``, the attention weights are sowed into the
            'intermediates' collection.
          decode: whether to prepare and use an autoregressive cache. The ``decode``
            flag passed into the call method will take precedence over the ``decode``
            flag passed into the constructor.

        Returns:
          output of shape `[batch_sizes..., length, features]`.
        """
        if rngs is None:
            rngs = self.rngs

        if inputs_k is None:
            if inputs_v is not None:
                raise ValueError(
                    "`inputs_k` cannot be None if `inputs_v` is not None. "
                    "To have both `inputs_k` and `inputs_v` be the same value, pass in the "
                    "value to `inputs_k` and leave `inputs_v` as None."
                )
            inputs_k = inputs_q
        if inputs_v is None:
            inputs_v = inputs_k

        if inputs_q.shape[-1] != self.in_features:
            raise ValueError(
                f"Incompatible input dimension, got {inputs_q.shape[-1]} "
                f"but module expects {self.in_features}."
            )

        query = self.query(inputs_q)
        key = self.key(inputs_k)
        value = self.value(inputs_v)

        if self.normalize_qk:
            assert self.query_ln is not None and self.key_ln is not None
            # Normalizing query and key projections stabilizes training with higher
            # LR. See ViT-22B paper http://arxiv.org/abs/2302.05442 for analysis.
            query = self.query_ln(query)
            key = self.key_ln(key)

        # During fast autoregressive decoding, we feed one position at a time,
        # and cache the keys and values step by step.
        decode = first_from(
            decode,
            self.decode,
            error_msg="""No `decode` argument was provided to MultiHeadAttention
        as either a __call__ argument, class attribute, or nnx.flag.""",
        )

        if decode:
            input_length = inputs_q.shape[-2]

            if (
                self.cached_key is None
                or self.cached_value is None
                or self.cache_index is None
            ):
                raise ValueError(
                    "Autoregressive cache not initialized, call ``init_cache`` first."
                )
            (
                *batch_dims,
                max_length,
                num_heads,
                depth_per_head,
            ) = self.cached_key.value.shape
            # shape check of cached keys against query input
            expected_shape = tuple(batch_dims) + (
                input_length,
                num_heads,
                depth_per_head,
            )
            if expected_shape != query.shape:
                raise ValueError(
                    "Autoregressive cache shape error, "
                    "expected query shape %s instead got %s."
                    % (expected_shape, query.shape)
                )
            # update key, value caches with our new 1d spatial slices
            cur_index = self.cache_index.value
            zero = jnp.array(0, dtype=lax.dtype(cur_index.dtype))
            indices = (zero,) * len(batch_dims) + (cur_index, zero, zero)
            key = lax.dynamic_update_slice(self.cached_key.value, key, indices)
            value = lax.dynamic_update_slice(self.cached_value.value, value, indices)
            self.cached_key.value = key
            self.cached_value.value = value
            self.cache_index.value += input_length
            # causal mask for cached decoder self-attention:
            # our single query position should only attend to those key
            # positions that have already been generated and cached,
            # not the remaining zero elements.
            mask = nnx.combine_masks(
                mask,
                jnp.broadcast_to(
                    jnp.arange(max_length)[None, :]
                    <= jnp.arange(input_length)[:, None] + cur_index,
                    tuple(batch_dims) + (1, input_length, max_length),
                ),
            )

        if self.dropout_rate > 0.0:  # Require `deterministic` only if using dropout.
            deterministic = first_from(
                deterministic,
                self.deterministic,
                error_msg="""No `deterministic` argument was provided to MultiHeadAttention
          as either a __call__ argument, class attribute, or nnx.flag.""",
            )
            if not deterministic:
                if rngs is None:
                    raise ValueError(
                        "'rngs' must be provided if 'dropout_rng' is not given."
                    )
                dropout_rng = rngs.dropout()
            else:
                dropout_rng = None
        else:
            deterministic = True
            dropout_rng = None

        # apply attention
        x = self.attention_fn(
            query,
            key,
            value,
            mask=mask,
            dropout_rng=dropout_rng,
            dropout_rate=self.dropout_rate,
            broadcast_dropout=self.broadcast_dropout,
            deterministic=deterministic,
            dtype=self.dtype,
            precision=self.precision,
            module=self if sow_weights else None,
        )
        # back to the original inputs dimensions
        out = self.out(x)
        return out

    def init_cache(self, input_shape: Shape, dtype: Dtype = jnp.float32):
        """Initializes cache for fast autoregressive decoding. When
        ``decode=True``, this method must be called first before performing
        forward inference.

        Example usage::

          >>> from flax import nnx
          >>> import jax.numpy as jnp
          ...
          >>> rngs = nnx.Rngs(42)
          ...
          >>> x = jnp.ones((1, 3))
          >>> model_nnx = nnx.MultiHeadAttention(
          ...   num_heads=2,
          ...   in_features=3,
          ...   qkv_features=6,
          ...   out_features=6,
          ...   decode=True,
          ...   rngs=rngs,
          ... )
          ...
          >>> # out_nnx = model_nnx(x)  <-- throws an error because cache isn't initialized
          ...
          >>> model_nnx.init_cache(x.shape)
          >>> out_nnx = model_nnx(x)
        """
        cache_shape = (*input_shape[:-1], self.num_heads, self.head_dim)
        self.cached_key = nnx.Cache(jnp.zeros(cache_shape, dtype))
        self.cached_value = nnx.Cache(jnp.zeros(cache_shape, dtype))
        self.cache_index = nnx.Cache(jnp.array(0, dtype=jnp.int32))


class AdditivePositionalEncoding(nnx.Module):
    def __init__(self, max_len, d_model, learned: bool, rngs: nnx.Rngs):
        self.learned = learned
        self.embed = nnx.Param(
            sin_embed_init(rngs.params(), (max_len, d_model), jnp.float32),
        )
        self.cache_index: nnx.Cache[Array] | None = None

    def __call__(self, x, *, decode: bool = False):
        if decode:
            assert self.cache_index is not None, "cache_index must be initialized"
            embed = self.embed[self.cache_index.value]
            self.cache_index.value += x.shape[-2]
        else:
            embed = self.embed[: x.shape[-2]]

        if not self.learned:
            embed = lax.stop_gradient(embed)

        return x + embed

    def init_cache(self, input_shape):
        self.cache_index = nnx.Cache(jnp.zeros((), jnp.uint32))


class TransformerLayer(nnx.Module):
    def __init__(
        self,
        *,
        d_model,
        n_heads=None,
        d_ff=None,
        ratio_ff=4.0,
        d_head=64,
        self_dropout: float = 0.0,
        cross_dropout: float = 0.0,
        d_cross_attn=None,
        attention_fn: Callable = nnx.dot_product_attention,
        rngs: nnx.Rngs,
    ):
        if n_heads is None:
            n_heads = d_model // d_head
        if d_ff is None:
            d_ff = int(d_model * ratio_ff)
        self.self_attn = MultiHeadAttention(
            num_heads=n_heads,
            in_features=d_model,
            dropout_rate=self_dropout,
            rngs=rngs,
            attention_fn=attention_fn,
        )
        self.ffn = nnx.Sequential(
            nnx.Linear(d_model, d_ff, rngs=rngs),
            nnx.gelu,
            nnx.Linear(d_ff, d_model, rngs=rngs),
        )
        self.ln1 = nnx.LayerNorm(d_model, rngs=rngs)
        self.ln2 = nnx.LayerNorm(d_model, rngs=rngs)

        if d_cross_attn is not None:
            self.cross_attn = MultiHeadAttention(
                num_heads=n_heads,
                in_features=d_model,
                dropout_rate=cross_dropout,
                rngs=rngs,
                attention_fn=attention_fn,
            )
            self.ln3 = nnx.LayerNorm(d_model, rngs=rngs)
            self.ln4 = nnx.LayerNorm(d_model, rngs=rngs)

    def __call__(
        self,
        x,
        y=None,
        *,
        self_mask: Literal["causal"] | Array | None = None,
        cross_mask: Array | None = None,
        decode: bool = False,
        rngs: nnx.Rngs,
    ):
        if self_mask == "causal":
            self_mask = nnx.make_causal_mask(x[..., 0])

        x = x + self.self_attn(
            self.ln1(x),
            decode=decode,
            mask=self_mask,
            rngs=rngs,
        )

        if hasattr(self, "cross_attn"):
            x = x + self.cross_attn(
                self.ln3(x),
                self.ln4(y),
                decode=False,
                mask=cross_mask,
                rngs=rngs,
            )

        x = x + self.ffn(self.ln2(x), rngs=rngs)

        return x

    def init_cache(self, input_shape: Shape, *, dtype: Dtype = jnp.float32):
        self.self_attn.init_cache(input_shape, dtype=dtype)


class TransformerBackbone(nnx.Module):
    def __init__(
        self,
        *,
        d_model,
        n_heads=None,
        d_ff=None,
        ratio_ff=4.0,
        d_head=64,
        self_dropout: float = 0.0,
        cross_dropout: float = 0.0,
        n_layers=6,
        d_cross_attn=None,
        attention_fn: Callable = nnx.dot_product_attention,
        rngs: nnx.Rngs,
    ):
        self.layers = [
            TransformerLayer(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                ratio_ff=ratio_ff,
                d_head=d_head,
                self_dropout=self_dropout,
                cross_dropout=cross_dropout,
                d_cross_attn=d_cross_attn,
                attention_fn=attention_fn,
                rngs=rngs,
            )
            for _ in range(n_layers)
        ]

    def __call__(self, x, y=None, *, rngs: nnx.Rngs, **kwargs):
        for layer in self.layers:
            x = layer(x, y, rngs=rngs, **kwargs)
        return x

    def init_cache(self, input_shape: Shape, *, dtype: Dtype = jnp.float32):
        for layer in self.layers:
            layer.init_cache(input_shape, dtype=dtype)


class DiscreteSequenceTransformer(nnx.Module):
    def __init__(
        self,
        *,
        d_model: int,
        vocab_size: int,
        max_len: int,
        rngs: nnx.Rngs,
        position_encoding_type: PositionalEncodingType = "learned",
        **kwargs,
    ):
        self.embedding = nnx.Embed(vocab_size, d_model, rngs=rngs)

        if position_encoding_type == "learned":
            self.position_encoding = AdditivePositionalEncoding(
                max_len, d_model, learned=True, rngs=rngs
            )
            self.attention_fn = nnx.dot_product_attention
        elif position_encoding_type == "sinusoidal":
            self.position_encoding = AdditivePositionalEncoding(
                max_len, d_model, learned=False, rngs=rngs
            )
            self.attention_fn = nnx.dot_product_attention
        elif position_encoding_type == "rope":
            self.attention_fn = rope_attention
        elif position_encoding_type == "alibi":
            self.attention_fn = alibi_attention

        self.backbone = TransformerBackbone(
            rngs=rngs,
            attention_fn=self.attention_fn,
            d_model=d_model,
            **kwargs,
        )
        self.d_model = d_model

    def __call__(self, x, y=None, *, rngs: nnx.Rngs, **kwargs):
        x = self.embedding(x)

        if hasattr(self, "position_encoding"):
            x = self.position_encoding(x, decode=kwargs.get("decode", False))

        for layer in self.backbone.layers:
            x = layer(x, y, **kwargs, rngs=rngs)
        return einops.einsum(x, self.embedding.embedding.value, "... d, v d -> ... v")

    def step_decode(self, x, y=None, *, temperature: float = 1.0, rngs: nnx.Rngs):
        output_logits = self(x[..., None], y, decode=True, rngs=rngs)

        new_token = jax.lax.select(
            temperature == 0,
            jnp.argmax(output_logits, axis=-1),
            jax.random.categorical(rngs.sample(), output_logits / temperature),
        )

        return jnp.squeeze(new_token, axis=-1)

    def decode(
        self,
        start_token,
        y=None,
        *,
        prompt_tokens: Array | None = None,
        temperature: float = 1.0,
        decode_length: int,
        rngs: nnx.Rngs,
    ):
        def _step_decode(carry, _, *, rngs):
            module: "DiscreteSequenceTransformer"
            token: Array
            module, token = carry
            new_token = module.step_decode(
                token, y=y, temperature=temperature, rngs=rngs
            )
            return (module, new_token), new_token

        if prompt_tokens is not None:
            assert (
                start_token is None
            ), "start_token must be None if prompt_tokens given"

            start_token_logits = self(prompt_tokens, y, decode=True, rngs=rngs)[
                ..., -1, :
            ]

            start_token = jax.lax.select(
                temperature == 0,
                jnp.argmax(start_token_logits, axis=-1),
                jax.random.categorical(rngs.sample(), start_token_logits / temperature),
            )

            # One fewer token to decode, since the prompt tokens output the first token
            decode_length -= 1
        else:
            assert (
                start_token is not None
            ), "start_token must be provided if prompt_tokens is None"

        scan_fn = nnx.scan(
            _step_decode, out_axes=-1,
        )

        _, output = scan_fn(
            (self, start_token),
            jnp.arange(decode_length),
            rngs=rngs,
        )

        if prompt_tokens is not None:
            output = jnp.concatenate([start_token[..., None], output], axis=-1)

        return output

    def loss_with_logits(
        self,
        gt_sequence,
        y=None,
        *,
        rngs: nnx.Rngs,
        self_mask=None,
        cross_mask=None,
        label_probs=None,
    ):
        inputs = gt_sequence[..., :-1]
        labels = gt_sequence[..., 1:]
        output_logits = self(
            inputs,
            y,
            decode=False,
            rngs=rngs,
            self_mask=self_mask,
            cross_mask=cross_mask,
        )

        if label_probs is None:
            loss = optax.softmax_cross_entropy_with_integer_labels(
                output_logits, labels
            )
        else:
            loss = optax.softmax_cross_entropy(output_logits, label_probs)

        return (
            jnp.mean(loss),
            output_logits,
        )

    def loss(
        self,
        gt_sequence,
        y=None,
        *,
        rngs: nnx.Rngs,
        self_mask=None,
        cross_mask=None,
        label_probs=None,
    ):
        return self.loss_with_logits(
            gt_sequence,
            y,
            rngs=rngs,
            self_mask=self_mask,
            cross_mask=cross_mask,
            label_probs=label_probs,
        )[0]

    def init_cache(self, input_shape: Shape, *, dtype: Dtype = jnp.float32):
        if hasattr(self, "position_encoding") and hasattr(
            self.position_encoding, "init_cache"
        ):
            self.position_encoding.init_cache(input_shape)
        self.backbone.init_cache((*input_shape, self.d_model), dtype=dtype)


class DiscreteTransformerContinuousInput(nnx.Module):
    def __init__(
        self,
        *,
        d_model: int,
        vocab_size: int,
        max_len: int,
        rngs: nnx.Rngs,
        position_encoding_type: PositionalEncodingType = "learned",
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.discrete_to_continuous = lambda x: 2 * x / (vocab_size - 1) - 1
        self.continuous_to_discrete = lambda x: (x + 1) / 2 * (vocab_size - 1)
        self.input_embedding = nnx.Sequential(
            nnx.Linear(1, d_model, rngs=rngs),
            nnx.gelu,
            nnx.Linear(d_model, d_model, rngs=rngs),
        )
        self.output_embedding = nnx.Embed(vocab_size, d_model, rngs=rngs)

        if position_encoding_type == "learned":
            self.position_encoding = AdditivePositionalEncoding(
                max_len, d_model, learned=True, rngs=rngs
            )
            self.attention_fn = nnx.dot_product_attention
        elif position_encoding_type == "sinusoidal":
            self.position_encoding = AdditivePositionalEncoding(
                max_len, d_model, learned=False, rngs=rngs
            )
            self.attention_fn = nnx.dot_product_attention
        elif position_encoding_type == "rope":
            self.attention_fn = rope_attention
        elif position_encoding_type == "alibi":
            self.attention_fn = alibi_attention

        self.backbone = TransformerBackbone(
            rngs=rngs,
            attention_fn=self.attention_fn,
            d_model=d_model,
            **kwargs,
        )
        self.d_model = d_model

    def __call__(
        self, x, y=None, *, rngs: nnx.Rngs, skip_input_embedding=False, **kwargs
    ):
        x = self.input_embedding(x[..., None])

        if hasattr(self, "position_encoding"):
            x = self.position_encoding(x, decode=kwargs.get("decode", False))

        for layer in self.backbone.layers:
            x = layer(x, y, **kwargs, rngs=rngs)
        return einops.einsum(
            x, self.output_embedding.embedding.value, "... d, v d -> ... v"
        )

    def step_decode(self, x, y=None, *, temperature: float = 1.0, rngs: nnx.Rngs):
        output_logits = self(
            x[..., None], y, decode=True, rngs=rngs, skip_input_embedding=True
        )

        new_token = jax.lax.select(
            temperature == 0,
            jnp.argmax(output_logits, axis=-1),
            jax.random.categorical(rngs.sample(), output_logits / temperature),
        )

        return self.discrete_to_continuous(jnp.squeeze(new_token, axis=-1))

    def decode(
        self,
        start_token,
        y=None,
        *,
        prompt_tokens: Array | None = None,
        temperature: float = 1.0,
        decode_length: int,
        rngs: nnx.Rngs,
    ):
        def _step_decode(carry, _, *, rngs):
            module: "DiscreteSequenceTransformer"
            token: Array
            module, token = carry
            new_token = module.step_decode(
                token, y=y, temperature=temperature, rngs=rngs
            )
            return (module, new_token), new_token

        if prompt_tokens is not None:
            assert (
                start_token is None
            ), "start_token must be None if prompt_tokens given"

            start_token_logits = self(prompt_tokens, y, decode=True, rngs=rngs)[
                ..., -1, :
            ]

            start_token = jax.lax.select(
                temperature == 0,
                jnp.argmax(start_token_logits, axis=-1),
                jax.random.categorical(rngs.sample(), start_token_logits / temperature),
            )

            # One fewer token to decode, since the prompt tokens output the first token
            decode_length -= 1
        else:
            assert (
                start_token is not None
            ), "start_token must be provided if prompt_tokens is None"

        scan_fn = nnx.scan(
            _step_decode, length=decode_length, in_axes=(nnx.Carry, None), out_axes=-1,
        )

        _, output = scan_fn(
            (self, start_token),
            None,
            rngs=rngs,
        )

        if prompt_tokens is not None:
            output = jnp.concatenate([start_token[..., None], output], axis=-1)

        return output

    def loss(
        self,
        gt_sequence,
        y=None,
        *,
        rngs: nnx.Rngs,
        self_mask=None,
        cross_mask=None,
    ):
        inputs = gt_sequence[..., :-1]
        labels = gt_sequence[..., 1:]
        output_logits = self(
            inputs,
            y,
            decode=False,
            rngs=rngs,
            self_mask=self_mask,
            cross_mask=cross_mask,
        )

        def hl_gauss_probs(x, sigma):
            bins = jnp.linspace(-1, 1, self.vocab_size - 1)
            cdf = jax.scipy.stats.norm.cdf(bins, loc=x[..., None], scale=sigma)
            cdf = jnp.concatenate([jnp.zeros_like(cdf[..., :1]), cdf, jnp.ones_like(cdf[..., -1:])], axis=-1)
            pdf = jnp.diff(cdf)
            return pdf

        target_probs = hl_gauss_probs(labels, sigma=0.03)
        return jnp.mean(optax.softmax_cross_entropy(output_logits, target_probs))

    def init_cache(self, input_shape: Shape, *, dtype: Dtype = jnp.float32):
        if hasattr(self, "position_encoding") and hasattr(
            self.position_encoding, "init_cache"
        ):
            self.position_encoding.init_cache(input_shape)
        self.backbone.init_cache((*input_shape, self.d_model), dtype=dtype)
