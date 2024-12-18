from typing import Any, Callable, Sequence, Tuple

import einops
import gymnasium
import gymnasium.vector.utils.numpy_utils as gnpu
import jax
import numpy as np

from gymnasium import spaces
from gymnasium import ObservationWrapper
import wandb

from chunkless.policy_wrappers import PolicyWrapper


class RemapKeysWrapper(ObservationWrapper):
    def __init__(self, env, remapping: Sequence[Tuple[str, str]]):
        super().__init__(env)
        self.remapping = remapping
        self.observation_space = spaces.Dict(
            {
                remap_key: self.env.observation_space[orig_key]
                for remap_key, orig_key in self.remapping
            }
        )

    def observation(self, observation):
        return {
            remap_key: observation[orig_key] for remap_key, orig_key in self.remapping
        }

class FreezeOnTerminationWrapper(gymnasium.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.termination_step = None

    def reset(self, **kwargs):
        if self.termination_step is not None:
            return self.termination_step[0], self.termination_step[-1]
        else:
            return super().reset(**kwargs)

    def hard_reset(self, **kwargs):
        self.termination_step = None
        return super().reset(**kwargs)

    def step(self, action):
        if self.termination_step is not None:
            return self.termination_step

        _, _, terminated, truncated, _ = step = super().step(action)

        if terminated or truncated:
            self.termination_step = step

        return step


class NoRenderWrapper(gymnasium.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def render(self):
        return None


def do_rollout_vectorized(
    policy,
    env: gymnasium.vector.VectorEnv,
    *,
    policy_wrapper_fn: Callable[[Any], PolicyWrapper],
    max_rollout_length=None,
    num_videos=0,
    video_fps=10,
    progress_callback: Callable[[float], None] = None,
):
    policy_wrapper = policy_wrapper_fn(policy)
    obs = gnpu.concatenate(env.observation_space, tuple(o for o, _ in env.call("hard_reset")), out=env.observation_space.sample())

    if num_videos > 0:
        video_frames = []
    else:
        video_frames = None

    max_reward = -np.inf

    ep_lens = np.ones(env.num_envs, dtype=int)

    trajectory = []

    alive_envs = np.ones(env.num_envs, dtype=bool)

    for i in range(max_rollout_length):
        action = policy_wrapper.predict(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        trajectory.append({"observation": obs, "next_observation": next_obs, "action": action, "reward": reward, "terminated": terminated, "truncated": truncated, "valid": alive_envs})
        obs = next_obs

        if video_frames is not None:
            video_frames.append(np.stack(env.call("render")[:num_videos]))

        alive_envs = ~np.logical_or(terminated, truncated)
        max_reward = np.maximum(max_reward, reward)
        ep_lens = np.where(
            alive_envs,
            ep_lens + 1,
            ep_lens,
        )

        if progress_callback is not None:
            progress_callback((i + 1) / max_rollout_length)

    trajectory = jax.tree.map(lambda *xs: np.stack(xs, axis=1), *trajectory)

    info = {
        "is_success": np.mean(info["is_success"]),
        "ep_len": np.mean(ep_lens),
        "max_reward": np.mean(max_reward),
        "final_reward": np.mean(reward),
    }

    if num_videos > 0:
        video_frames = np.stack(video_frames, axis=1)
        video_frames = einops.rearrange(video_frames, "n t h w c -> n t c h w")
        info["video"] = wandb.Video(video_frames, fps=video_fps, format="mp4")

    return info, trajectory
