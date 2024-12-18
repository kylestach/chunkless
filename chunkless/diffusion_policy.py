from functools import partial
from typing import Any, Callable, Dict, Literal, Mapping, NamedTuple, Sequence, Tuple
import chex
import einops
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.struct import dataclass
from flax.training.train_state import TrainState
import numpy as np
import optax
from chunkless.normalizers import Normalizer
from chunkless.spec import ModuleSpec


def mish(x):
    return x * jnp.tanh(nn.softplus(x))


class FiLM(nn.Module):
    @nn.compact
    def __call__(self, x, y):
        a = nn.Dense(features=x.shape[-1])(y)
        b = nn.Dense(features=x.shape[-1])(y)
        return a * x + b


class ConvBlock(nn.Module):
    num_groups: int = 8
    kernel_size: int = 5

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=x.shape[-1], kernel_size=(self.kernel_size,))(x)
        x = nn.GroupNorm(num_groups=self.num_groups)(x)
        x = mish(x)
        return x


class DiffusionResidualBlock(nn.Module):
    @nn.compact
    def __call__(self, x, y):
        residual = x

        x = ConvBlock(num_groups=8, kernel_size=5)(x)
        x = FiLM()(x, y)
        x = ConvBlock(num_groups=8, kernel_size=5)(x)

        return x + residual


class UNetDiffusionBackbone(nn.Module):
    base_num_features: int = 128
    num_down_blocks: int = 3

    @nn.compact
    def __call__(self, x, obs_encoding):
        num_features = x.shape[-1]

        obs_encoding = einops.rearrange(obs_encoding, "... d -> ... () d")

        # Project to base number of features
        x = nn.Dense(features=self.base_num_features)(x)
        encoder_skip_features = []

        # Downsample
        for i in range(self.num_down_blocks):
            x = DiffusionResidualBlock()(x, obs_encoding)
            x = DiffusionResidualBlock()(x, obs_encoding)
            encoder_skip_features.append(x)
            if i < self.num_down_blocks - 1:
                x = nn.Conv(features=x.shape[-1] * 2, kernel_size=(3,), strides=(2,))(x)

        x = DiffusionResidualBlock()(x, obs_encoding)
        x = DiffusionResidualBlock()(x, obs_encoding)

        # Upsample
        for i in range(self.num_down_blocks):
            x = jnp.concatenate([x, encoder_skip_features.pop()], axis=-1)
            x = DiffusionResidualBlock()(x, obs_encoding)
            x = DiffusionResidualBlock()(x, obs_encoding)
            if i < self.num_down_blocks - 1:
                x = nn.ConvTranspose(
                    features=x.shape[-1] // 2, kernel_size=(3,), strides=2
                )(x)

        # Project to output
        x = ConvBlock()(x)
        x = nn.Dense(features=num_features)(x)

        return x


def sinusoidal_embedding(t, num_features):
    half_dim = num_features // 2
    emb = jnp.log(10000) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim) * -emb)
    emb = t[..., None] * emb
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
    return emb


class TimeEmbedding(nn.Module):
    features: int = 128

    @nn.compact
    def __call__(self, t):
        x = sinusoidal_embedding(t, self.features)
        x = nn.Dense(features=self.features * 4)(x)
        x = mish(x)
        x = nn.Dense(features=self.features)(x)
        return x


@dataclass
class CosineNoiseSchedule:
    def __call__(self, t):
        alpha = jnp.cos(jnp.pi / 2 * t)
        alpha = jnp.clip(alpha, 0.01, 1)
        beta = jnp.sqrt(1 - alpha**2)
        return alpha, beta


@dataclass
class DDIMScheduler:
    noise_schedule: CosineNoiseSchedule
    prediction_type: Literal["eps", "v", "x0"] = "eps"

    def pred_target(self, x0, eps, t):
        alpha, beta = self.noise_schedule(t)
        if self.prediction_type == "v":
            return alpha * eps - beta * x0
        elif self.prediction_type == "x0":
            return (eps - beta * x0) / alpha
        elif self.prediction_type == "eps":
            return eps
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")

    def pred_eps(self, xt, value, t):
        alpha, beta = self.noise_schedule(t)
        if self.prediction_type == "v":
            return (xt - (alpha * xt - beta * value)) / beta
        elif self.prediction_type == "x0":
            return (value - beta * xt) / alpha
        elif self.prediction_type == "eps":
            return value
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")

    def step(self, xt, prediction, t0, t1, mode="ddim", noise=None, x0_bounds=None):
        alpha0, beta0 = self.noise_schedule(t0)
        alpha1, beta1 = self.noise_schedule(t1)

        if mode == "ddim":
            eta = 0
        elif mode == "ddpm":
            eta = ((beta0 / beta1) ** 2 * (1 - (alpha1 / alpha0) ** 2)) ** 0.5
            assert noise is not None, "DDPM requires noise during sampling"
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Get noise epsilon from prediction
        noise_pred = self.pred_eps(xt, prediction, t1)

        # Predicted x0
        # xt = alpha_t * x0 + beta_t * noise => x0 = (xt - beta_t * noise) / alpha_t
        x0_pred = (xt - beta1 * noise_pred) / alpha1

        if x0_bounds is not None:
            x0_pred = jnp.clip(x0_pred, -x0_bounds, x0_bounds)

        # Direction to xt
        x0_dir = jnp.sqrt(jnp.clip(1 - alpha0**2 - eta**2, 0, 1)) * noise_pred

        # Step
        x = alpha0 * x0_pred + x0_dir

        if noise is not None:
            x = x + eta * noise

        return x

    def add_noise(self, x, noise, t):
        while len(t.shape) < len(x.shape):
            t = t[..., None]
        alpha, beta = self.noise_schedule(t)
        return alpha * x + beta * noise

    def loss(self, x0, prediction, noise, t, padding=None):
        target = jax.vmap(self.pred_target)(x0, noise, t)
        error = (prediction - target) ** 2
        if padding is None:
            error = jnp.mean(error)
        else:
            mask = (1 - padding)[..., None]
            error = jnp.mean(error * mask) / jnp.mean(mask)
        return error


class DiffusionNetwork(nn.Module):
    noise_scheduler: DDIMScheduler
    backbone_spec: ModuleSpec
    encoder_spec: ModuleSpec
    action_shape: Tuple[int, ...]
    obs_history_size: int

    def setup(self):
        self.encoder = self.encoder_spec.instantiate()
        self.backbone = self.backbone_spec.instantiate()
        self.t_embedding = TimeEmbedding(features=self.encoder.out_features)

    def __call__(self, x, obs, t):
        encoding = self.encode_obs(obs)
        return self.predict(x, encoding, t)

    def predict(self, x, obs_encoding, t):
        return self.backbone(x, (obs_encoding + self.t_embedding(t)))

    def denoise_step(self, x, obs_encoding, *, t0, t1, **kwargs):
        prediction = self.predict(x, obs_encoding, t1)
        return self.remove_noise(x, prediction, t0, t1, **kwargs)

    def remove_noise(self, x, prediction, t0, t1, **kwargs):
        return self.noise_scheduler.step(x, prediction, t0, t1, **kwargs)

    def encode_obs(self, obs):
        return self.encoder(obs)

    def loss(self, batch, rng: jax.Array):
        encoding = self.encode_obs(batch)

        x0 = batch["action"]
        rng_t, rng_x = jax.random.split(rng, 2)
        t = jax.random.uniform(rng_t, (x0.shape[0],), minval=0.0, maxval=1.0)
        noise = jax.random.normal(rng_x, x0.shape)

        xt = self.noise_scheduler.add_noise(x0, noise, t)

        prediction = self.predict(xt, encoding, t)
        loss = self.noise_scheduler.loss(x0, prediction, noise, t, padding=batch.get("action_is_pad", None))

        return loss


@partial(jax.jit, static_argnames=["mode", "apply_fn"])
def _denoise_step(
    params,
    obs_encoding,
    action,
    t0,
    t1,
    *,
    mode="ddim",
    rng=None,
    apply_fn: Callable,
    x0_bounds=None,
):
    if mode == "ddpm":
        assert rng is not None, "DDPM requires noise during sampling"
        noise = jax.random.normal(rng, action.shape)
    else:
        noise = None

    return apply_fn(
        {"params": params},
        action,
        obs_encoding,
        t0=t0,
        t1=t1,
        mode=mode,
        noise=noise,
        method="denoise_step",
        x0_bounds=x0_bounds,
    )


@partial(
    jax.jit,
    static_argnames=["mode", "apply_fn", "num_prediction_timesteps", "action_shape"],
)
def _multi_obs_denoise_step(
    params,
    obs_encoding,
    action,
    t0,
    t1,
    *,
    mode="ddim",
    rng=None,
    apply_fn: Callable,
    x0_bounds=None,
    num_prediction_timesteps: int,
    action_shape: Tuple[int, ...],
):
    """
    Observations are (B, K+H-1, *O)
    For H=3, K=4, one element is:
    | O0 | O1 | O2 | O3 | O4 | O5 |
    Then batched observations look like
    | O0 | O1 | O2 |
    | O1 | O2 | O3 |
    | O2 | O3 | O4 |
    | O3 | O4 | O5 |
    Each diffusion step predicts:
    | A0 | A1 | A2 | A3 |
    | A1 | A2 | A3 | A4 |
    | A2 | A3 | A4 | A5 |
    | A3 | A4 | A5 | A6 |
    To average together, offset them correctly:
    | A0 | A1 | A2 | A3 |    |    |    |
    |    | A1 | A2 | A3 | A4 |    |    |
    |    |    | A2 | A3 | A4 | A5 |    |
    |    |    |    | A3 | A4 | A5 | A6 |
    """
    if mode == "ddpm":
        assert rng is not None, "DDPM requires noise during sampling"
        noise = jax.random.normal(rng, action.shape)
    else:
        noise = None

    chex.assert_shape(obs_encoding, (None, num_prediction_timesteps, None))
    assert (
        num_prediction_timesteps >= 1
    ), "Must predict at least one timestep but got {}".format(num_prediction_timesteps)

    # Add padding to the end of predictions
    def _pad_and_roll(x):
        chex.assert_shape(x, (num_prediction_timesteps, *action_shape))
        x = jnp.pad(x, ((0, 0), (0, num_prediction_timesteps - 1), (0, 0)))
        x = jax.vmap(partial(jnp.roll, axis=0))(x, jnp.arange(num_prediction_timesteps))
        return x

    predictions_mask = jnp.ones((num_prediction_timesteps, *action_shape))
    predictions_mask = _pad_and_roll(predictions_mask)

    @jax.vmap
    def average_predictions(predictions):
        chex.assert_shape(predictions, (num_prediction_timesteps, *action_shape))

        predictions = _pad_and_roll(predictions)

        return jnp.sum(predictions * predictions_mask, axis=0) / jnp.sum(
            predictions_mask, axis=0
        )

    @jax.vmap
    def extract_actions(all_actions):
        return jax.vmap(
            lambda a: jax.lax.dynamic_slice(all_actions, (a, 0), action_shape)
        )(jnp.arange(num_prediction_timesteps))

    predictions = apply_fn(
        {"params": params},
        extract_actions(action),
        obs_encoding,
        t1,
        method="predict",
    )
    predictions = average_predictions(predictions)
    return apply_fn(
        {"params": params},
        action,
        predictions,
        t0=t0,
        t1=t1,
        mode=mode,
        noise=noise,
        method="remove_noise",
        x0_bounds=x0_bounds,
    )


def multi_obs_predict_fn(
    network: DiffusionNetwork,
    params,
    obs,
    num_denoising_steps,
    rng: jax.Array,
    mode="ddim",
    x0_bounds=None,
    gt_action=None,
    gt_action_mask=None,
):
    batch_size, num_obs = jax.tree.leaves(obs)[0].shape[:2]
    num_prediction_timesteps = num_obs - network.obs_history_size + 1
    full_sequence_length = network.action_shape[0] + num_prediction_timesteps - 1

    batched_obs_idcs = (
        jnp.arange(num_prediction_timesteps)[:, None]
        + jnp.arange(network.obs_history_size)[None, :]
    )
    batched_obs = jax.tree_map(lambda x: x[:, batched_obs_idcs], obs)

    # Run encoder on (flattened) batched observations
    batched_obs = jax.tree.map(
        lambda o: einops.rearrange(o, "b k ... -> (b k) ..."), batched_obs
    )
    obs_encoding = network.apply({"params": params}, batched_obs, method="encode_obs")
    obs_encoding = einops.rearrange(
        obs_encoding, "(b k) ... -> b k ...", k=num_prediction_timesteps
    )

    # Run diffusion on the whole sequence jointly
    original_noise = jax.random.normal(
        rng, (batch_size, full_sequence_length, network.action_shape[1])
    )
    noisy_action = original_noise

    ts = np.linspace(0, 1, num_denoising_steps + 1)
    for t0, t1 in zip(ts[:-1][::-1], ts[1:][::-1]):
        rng, subrng = jax.random.split(rng)

        # Condition on previous ground truth actions if provided
        if gt_action is not None:
            noisy_action = jnp.where(
                gt_action_mask,
                network.noise_scheduler.add_noise(gt_action, original_noise, t1),
                noisy_action,
            )

        noisy_action = _multi_obs_denoise_step(
            params,
            obs_encoding,
            noisy_action,
            t0,
            t1,
            mode=mode,
            rng=subrng,
            apply_fn=network.apply,
            x0_bounds=x0_bounds,
            num_prediction_timesteps=num_prediction_timesteps,
            action_shape=network.action_shape,
        )

    return jax.device_get(noisy_action[:, num_prediction_timesteps - 1 :])


def predict_fn(
    network: DiffusionNetwork,
    params,
    obs,
    num_denoising_steps,
    rng: jax.Array,
    mode="ddim",
    x0_bounds=None,
    gt_action=None,
    gt_action_mask=None,
):
    batch_size = jax.tree.leaves(obs)[0].shape[0]
    original_noise = jax.random.normal(rng, (batch_size, *network.action_shape))
    noisy_action = original_noise

    obs_encoding = network.apply({"params": params}, obs, method="encode_obs")

    ts = np.linspace(0, 1, num_denoising_steps + 1)
    for t0, t1 in zip(ts[:-1][::-1], ts[1:][::-1]):
        rng, subrng = jax.random.split(rng)

        # Condition on previous ground truth actions if provided
        if gt_action is not None:
            noisy_action = jnp.where(
                gt_action_mask,
                network.noise_scheduler.add_noise(gt_action, original_noise, t1),
                noisy_action,
            )

        noisy_action = _denoise_step(
            params,
            obs_encoding,
            noisy_action,
            t0,
            t1,
            mode=mode,
            rng=subrng,
            apply_fn=network.apply,
            x0_bounds=x0_bounds,
        )

    return jax.device_get(noisy_action)


def make_diffusion_policy(
    obs_type: Literal["lowdim", "image"],
    backbone_type: Literal["unet", "mlp"],
    prediction_type: Literal["eps", "v", "x0"],
    state_keys: Sequence[str],
    action_shape: Tuple[int, ...],
    obs_history_size: int,
):
    if obs_type == "lowdim":
        encoder_spec = ModuleSpec.from_name(
            "chunkless.encoders.MLPEncoder",
            dict(state_keys=state_keys, out_features=256, num_layers=0),
        )
    else:
        raise ValueError(f"Unknown observation type: {obs_type}")

    if backbone_type == "unet":
        backbone_spec = ModuleSpec.from_name(
            "chunkless.diffusion_policy.UNetDiffusionBackbone", {}
        )
    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}")

    noise_scheduler = DDIMScheduler(
        CosineNoiseSchedule(), prediction_type=prediction_type
    )

    network = DiffusionNetwork(
        backbone_spec=backbone_spec,
        encoder_spec=encoder_spec,
        noise_scheduler=noise_scheduler,
        action_shape=action_shape,
        obs_history_size=obs_history_size,
    )

    return (
        network,
        partial(multi_obs_predict_fn, network=network),
        # partial(predict_fn, network=network),
    )


class EMAState(NamedTuple):
    step: int
    ema_params: optax.Params


def ema_params(ema_rate: optax.ScalarOrSchedule):
    def init_fn(params):
        return EMAState(step=0, ema_params=params)

    def update_fn(updates, opt_state, params=None):
        if callable(ema_rate):
            decay = ema_rate(opt_state.step)
        else:
            decay = ema_rate

        return updates, EMAState(
            step=opt_state.step + 1,
            ema_params=optax.tree_utils.tree_update_moment(
                params, opt_state.ema_params, decay=decay, order=1
            ),
        )

    return optax.GradientTransformation(init_fn, update_fn)


def ema_rate_schedule(
    inv_gamma: float, power: float, min_value: float = 0.0, max_value: float = 0.999
):
    def _schedule(step):
        value = 1 - (1 + step / inv_gamma) ** -power
        return jnp.clip(value, min_value, max_value)

    return _schedule


def make_optimizer(
    *,
    num_train_steps,
    warmup_steps,
    learning_rate=1e-4,
    weight_decay=1e-6,
    b1=0.95,
    b2=0.999,
    eps=1e-8,
    ema_inv_gamma=1.0,
    ema_power=2 / 3,
    ema_min_value=0.0,
    ema_max_value=0.9999,
):
    @optax.inject_hyperparams
    def _make_optimizer(learning_rate, ema_rate):
        return optax.named_chain(
            (
                "adam",
                optax.adamw(learning_rate, weight_decay=weight_decay, b1=b1, b2=b2, eps=eps),
            ),
            (
                "ema",
                ema_params(ema_rate),
            ),
        )

    return _make_optimizer(
        optax.warmup_cosine_decay_schedule(
            0, learning_rate, warmup_steps, num_train_steps
        ),
        ema_rate=ema_rate_schedule(
            ema_inv_gamma, ema_power, ema_min_value, ema_max_value
        ),
    )


def diffusion_train_step(train_state, batch, rng):
    def loss_fn(params):
        loss = train_state.apply_fn({"params": params}, batch, rng, method="loss")
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(train_state.params)
    train_state = train_state.apply_gradients(grads=grads)

    metrics = {
        "loss": loss,
        "learning_rate": train_state.opt_state.hyperparams["learning_rate"],
        "ema_rate": train_state.opt_state.hyperparams["ema_rate"],
        "grad_norm": optax.global_norm(grads),
    }
    return train_state, metrics


class DiffusionPolicy:
    def __init__(
        self,
        *,
        obs_type: Literal["lowdim", "image"],
        backbone_type: Literal["unet", "mlp"],
        prediction_type: Literal["eps", "v", "x0"] = "eps",
        abstract_obs: Any,
        abstract_action: jax.ShapeDtypeStruct,
        action_start_index: int,
        action_end_index: int,
        seed=0,
        optimizer_config: dict,
        stats: Mapping,
        normalize_rules: Sequence[Tuple[str, Literal["mean_std", "min_max", "none"]]],
        state_keys: Sequence[str] | None = None,
        sample_clip_bounds: float | None = None,
    ):
        self._normalizer = Normalizer(stats, normalize_rules)

        network, predict_fn = make_diffusion_policy(
            obs_type,
            backbone_type,
            prediction_type,
            state_keys=state_keys,
            action_shape=abstract_action.shape,
            obs_history_size=jax.tree.leaves(abstract_obs)[0].shape[0],
        )
        optimizer = make_optimizer(**optimizer_config)
        rng, params_rng = jax.random.split(jax.random.PRNGKey(seed))
        params = network.init(
            params_rng,
            self._normalizer.normalize(abstract_action[None], key="action"),
            self._normalizer.normalize(jax.tree_map(lambda x: x[None], abstract_obs)),
            jnp.zeros((1,)),
        )["params"]

        self._train_state = TrainState.create(
            apply_fn=network.apply, params=params, tx=optimizer
        )
        self._network = network
        self._predict = predict_fn
        self._rng = rng
        self._jit_train_step = jax.jit(diffusion_train_step)

        self._action_start_index = action_start_index
        self._action_end_index = action_end_index
        self._sample_clip_bounds = sample_clip_bounds

        self._action_shape = abstract_action.shape

    def train_step(self, batch):
        batch = self._normalizer.normalize(batch)
        self._rng, subrng = jax.random.split(self._rng)
        self._train_state, metrics = self._jit_train_step(
            self._train_state, batch, subrng
        )
        return metrics

    def predict(
        self,
        obs,
        *,
        use_ema,
        num_denoising_steps,
        action_history=None,
        action_history_mask=None,
    ):
        obs = self._normalizer.normalize(obs)
        batch_size = jax.tree.leaves(obs)[0].shape[0]
        if use_ema:
            params = self._train_state.opt_state.inner_state["ema"].ema_params
        else:
            params = self._train_state.params
        self._rng, subrng = jax.random.split(self._rng)

        predict_kwargs = dict(
            params=params,
            obs=obs,
            rng=subrng,
            x0_bounds=self._sample_clip_bounds,
            mode="ddpm",
            num_denoising_steps=num_denoising_steps,
        )

        if action_history is not None:
            chex.assert_shape(
                [action_history_mask, action_history], (batch_size, *self._action_shape)
            )

            action_history = self._normalizer.normalize(action_history, key="action")

            if action_history_mask is None:
                action_history_mask = np.ones(
                    (batch_size, *self._action_shape), dtype=bool
                )

            predict_kwargs["gt_action"] = action_history
            predict_kwargs["gt_action_mask"] = action_history_mask

        action = self._predict(**predict_kwargs)[
            :, self._action_start_index : self._action_end_index
        ]

        action = self._normalizer.unnormalize(action, key="action")
        return action
