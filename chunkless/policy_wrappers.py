from collections import deque

import einops
import jax
import numpy as np


class PolicyWrapper:
    def __init__(self, policy, obs_history_size, policy_kwargs):
        self.policy = policy
        self.policy_kwargs = policy_kwargs
        self.obs_queue = deque([], maxlen=obs_history_size)
        self.obs_history_size = obs_history_size

    def _record_obs(self, obs):
        while len(self.obs_queue) < self.obs_history_size:
            self.obs_queue.append(obs)
        self.obs_queue.append(obs)

    def reset(self):
        self.obs_queue.clear()

    def get_obs(self):
        return jax.tree.map(lambda *xs: np.stack(xs, axis=1), *self.obs_queue)

    def predict(self, obs):
        return self.policy.predict(obs, **self.policy_kwargs)


class TemporalEnsemblingWrapper(PolicyWrapper):
    def __init__(self, policy, obs_history_size, action_chunk_size, policy_kwargs):
        super().__init__(policy, obs_history_size, policy_kwargs)
        self.action_chunk_size = action_chunk_size
        self.action_queue = deque([], maxlen=action_chunk_size)

    def reset(self):
        super().reset()
        self.action_queue.clear()

    def predict(self, obs):
        self._record_obs(obs)
        chunk = super().predict(self.get_obs())
        chunk = einops.rearrange(chunk, "... t d -> t ... d")
        self.action_queue.append(chunk[: self.action_chunk_size])

        return np.mean(
            np.stack(
                [self.action_queue[-i - 1][i] for i in range(len(self.action_queue))]
            ),
            axis=0,
        )


class ActionChunkWrapper(PolicyWrapper):
    def __init__(self, policy, obs_history_size, action_chunk_size, policy_kwargs):
        super().__init__(policy, obs_history_size, policy_kwargs)
        self.action_chunk_size = action_chunk_size
        self.action_queue = deque([], maxlen=action_chunk_size)

    def reset(self):
        super().reset()
        self.action_queue.clear()

    def predict(self, obs):
        self._record_obs(obs)
        if len(self.action_queue) == 0:
            chunk = super().predict(self.get_obs())
            chunk = einops.rearrange(chunk, "... t d -> t ... d")
            self.action_queue.extend(list(chunk[: self.action_chunk_size]))
        return self.action_queue.popleft()


class CausalPolicyWrapper(PolicyWrapper):
    def __init__(self, policy, history_size, policy_kwargs):
        super().__init__(policy, history_size, policy_kwargs)
        self.past_actions = deque([], maxlen=history_size - 1)
    
    def reset(self):
        super().reset()
        self.past_actions.clear()

    def predict(self, obs):
        self.obs_queue.append(obs)
        batch_size = jax.tree.leaves(obs)[0].shape[0]
        action_history = np.stack(self.past_actions, axis=1) if len(self.past_actions) > 0 else np.zeros((batch_size, 0, 2))
        action = self.policy.predict(
            self.get_obs(),
            action_history=action_history,
            **self.policy_kwargs
        )
        self.past_actions.append(action)
        return action


class PastConditionalDiffusionWrapper(PolicyWrapper):
    """
    Run diffusion policy conditioned on past actions by doing joint diffusion over past and future actions but teacher-forcing the past actions. Runs without action chunking.
    """

    def __init__(
        self,
        policy,
        obs_history_size,
        ac_history_size,
        action_chunk_size,
        policy_kwargs,
    ):
        super().__init__(policy, obs_history_size, policy_kwargs)
        self.action_chunk_size = action_chunk_size
        self.ac_history_size = ac_history_size
        self.past_actions = deque([], maxlen=ac_history_size)
        self.prev_action = None

    def reset(self):
        super().reset()
        self.past_actions.clear()
        self.prev_action = None

    def predict(self, obs):
        self._record_obs(obs)
        if self.prev_action is not None:
            self.past_actions.append(self.prev_action)

        ac_dim = self.policy._action_shape[-1]
        batch_size = jax.tree.leaves(obs)[0].shape[0]

        action_history = np.zeros((batch_size, self.action_chunk_size, ac_dim))
        action_history_mask = np.zeros(
            (batch_size, self.action_chunk_size, ac_dim), dtype=bool
        )
        for i, action in enumerate(reversed(self.past_actions)):
            action_history[:, self.ac_history_size - i - 1] = action
            action_history_mask[:, self.ac_history_size - i - 1, :] = True

        action_chunk = self.policy.predict(
            self.get_obs(),
            action_history=action_history,
            action_history_mask=action_history_mask,
            **self.policy_kwargs
        )
        self.prev_action = action_chunk[:, 0]
        return self.prev_action
