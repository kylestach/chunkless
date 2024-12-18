from functools import partial
import os
from typing import Iterator, Union
import gymnasium
import jax
import jax.numpy as jnp
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader, Sampler
import tqdm

from absl import flags, app
from ml_collections import config_flags, ConfigDict

import orbax.checkpoint as ocp
import tensorflow as tf

from gym_pusht.envs.pusht import PushTEnv
import wandb

from chunkless.policy_wrappers import CausalPolicyWrapper
from chunkless.dataset import NpzTrajectoryDataset, EpisodeAwareInfiniteSampler
from chunkless.causal_policy import CausalPolicy
from chunkless.gym import (
    FreezeOnTerminationWrapper,
    NoRenderWrapper,
    do_rollout_vectorized,
    RemapKeysWrapper,
)


def main(_):
    config = flags.FLAGS.config.to_dict()
    model_config = config["model"]

    # # Do configuration here

    # desc_pairs = []

    # def _build_description(keypath, x, y):
    #     keystr = ".".join([k.key for k in keypath])
    #     if x != y:
    #         desc_pairs.append(f"{keystr}={x}")

    # jax.tree_util.tree_map_with_path(_build_description, config, get_default_config())

    # Make the environment
    def make_env(i):
        env = PushTEnv(
            obs_type="environment_state_agent_pos",
            render_mode="rgb_array",
            visualization_width=128,
            visualization_height=128,
        )
        env = RemapKeysWrapper(
            env,
            [
                ("observation.environment_state", "environment_state"),
                ("observation.state", "agent_pos"),
            ],
        )
        if i >= config["num_rollouts"] - config["num_videos"]:
            env = NoRenderWrapper(env)
        env = FreezeOnTerminationWrapper(env)
        return env

    env = gymnasium.vector.AsyncVectorEnv(
        [partial(make_env, i) for i in range(config["num_rollouts"])]
    )

    # Make the dataset
    dataset = LeRobotDataset(
        repo_id="lerobot/pusht_keypoints",
        delta_timestamps={
            "observation.state": [
                config["dt"] * k
                for k in range(1 - model_config["num_history_steps"], 1)
            ],
            "observation.environment_state": [
                config["dt"] * k
                for k in range(1 - model_config["num_history_steps"], 1)
            ],
            # History actions for observations (-3, -2, -1, 0) are (-4, -3, -2, -1)
            "action": [
                config["dt"] * k
                for k in range(1 - model_config["num_history_steps"], 1)
            ],
        },
    )

    if config["load_aux_trajectories"] is not None:
        stats = dataset.stats
        dataset = NpzTrajectoryDataset(
            config["load_aux_trajectories"],
            n_obs_history=model_config["num_history_steps"],
            n_ac_history=model_config["num_history_steps"],
            action_chunk_size=model_config["num_history_steps"],
            exec_horizon=1,
        )
        dataset.episode_data_index = {
            "from": dataset.ep_starts,
            "to": dataset.ep_ends,
        }
        dataset.stats = stats

    def _np_collate(batch_items):
        return jax.tree.map(lambda *xs: np.stack(xs), *batch_items)

    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        collate_fn=_np_collate,
        sampler=EpisodeAwareInfiniteSampler(
            dataset.episode_data_index,
            drop_n_first_frames=model_config["num_history_steps"] - 1,
            drop_n_last_frames=0,
            shuffle=True,
        ),
        num_workers=32,
    )
    data_it = iter(dataloader)

    # Make the policy
    sample_batch = next(data_it)

    policy = CausalPolicy(
        obs_type="lowdim",
        abstract_obs={
            "observation.state": sample_batch["observation.state"],
            "observation.environment_state": sample_batch[
                "observation.environment_state"
            ],
        },
        abstract_action=sample_batch["action"],
        seed=0,
        optimizer_config={
            "num_train_steps": config["num_train_steps"],
            "warmup_steps": config["warmup_steps"],
            **config["optimizer"],
        },
        stats=dataset.stats,
        normalize_rules=[
            ("action", "min_max"),
            ("observation.state", "min_max"),
            ("observation.environment_state", "min_max"),
            ("*", "none"),
        ],
        state_keys=["observation.state", "observation.environment_state"],
        model_kwargs=model_config["model_kwargs"],
    )

    def make_name(run_format, config):
        config_flat = {}

        def _insert(path, x):
            config_flat["/".join([p.key for p in path])] = x

        jax.tree_util.tree_map_with_path(_insert, config)
        return run_format.format(**config_flat)

    if "wandb_run_format" in config:
        name = make_name(config["wandb_run_format"], config)
    else:
        name = None

    wandb.init(project=config["wandb_project"], name=name, config=config)
    # if len(desc_pairs) > 0:
    #     wandb.run.name = wandb.run.name + f"({', '.join(desc_pairs)})"

    inference_schemes = [
        (
            "causal",
            lambda policy: CausalPolicyWrapper(
                policy,
                history_size=model_config["num_history_steps"],
                policy_kwargs={"temperature": None, "use_ema": True},
            ),
        ),
    ]

    if config["checkpoint_dir"]:
        checkpoint_mgr = ocp.CheckpointManager(
            tf.io.gfile.join(
                config["checkpoint_dir"], f"{wandb.run.name}-{wandb.run.id}"
            ),
            item_names=["params", "opt_state", "config"],
            options=ocp.CheckpointManagerOptions(max_to_keep=1),
        )
    else:
        checkpoint_mgr = None

    with tqdm.trange(config["num_train_steps"], dynamic_ncols=True) as pbar:
        pbar.set_description(f"Training")
        metrics = []
        for i in pbar:
            try:
                batch = next(data_it)
            except StopIteration:
                data_it = iter(dataloader)
                batch = next(data_it)

            # Do a training step
            metrics.append(policy.train_step(batch))
            pbar.set_description(f"Loss: {metrics[-1]['loss']:.4f}")

            # Policy evaluation
            if (i + 1) % 5000 == 0:
                info = {}
                for name, scheme in inference_schemes:
                    prefix = f"rollout/{name}"
                    rollout_kwargs = dict(
                        policy=policy,
                        env=env,
                        policy_wrapper_fn=scheme,
                        max_rollout_length=config["max_episode_length"],
                        video_fps=10,
                        progress_callback=lambda t: pbar.set_description(
                            f"Evaluating {name} [{int(100*t)}%]"
                        ),
                        num_videos=config["num_videos"],
                    )
                    rollout_info, _ = do_rollout_vectorized(**rollout_kwargs)
                    info = info | {f"{prefix}.{k}": v for k, v in rollout_info.items()}
                pbar.set_description(f"Training")

                wandb.log(
                    info,
                    commit=False,
                    step=i + 1,
                )

            # Prediction metrics
            if (i + 1) % 500 == 0:
                pred_metrics = {}
                target_action = batch["action"][:, -1]
                obs = {k: v for k, v in batch.items() if k.startswith("observation") and not k.endswith("is_pad")}
                pred_action = policy.predict(
                    obs,
                    action_history=batch["action"][:, :-1],
                    use_ema=True,
                    temperature=0.7,
                )
                pred_metrics[f"predict/pred_mse"] = np.mean(
                    (pred_action - target_action) ** 2
                )
                wandb.log(
                    pred_metrics,
                    commit=False,
                    step=i + 1,
                )

            # W&B logging
            if (i + 1) % 10 == 0:
                metrics = jax.tree.map(
                    lambda *xs: np.mean(np.stack(xs), axis=0), *metrics
                )
                wandb.log(
                    {f"train/{k}": v for k, v in metrics.items()},
                    commit=True,
                    step=i + 1,
                )
                metrics = []

            # Save checkpoint
            if checkpoint_mgr is not None and (
                (i + 1) % 10000 == 0 or i == config["num_train_steps"] - 1
            ):
                checkpoint_mgr.save(
                    i,
                    args=ocp.args.Composite(
                        params=ocp.args.StandardSave(policy._train_state.params),
                        opt_state=ocp.args.StandardSave(policy._train_state.opt_state),
                        config=ocp.args.JsonSave(config),
                    ),
                )

    if checkpoint_mgr is not None:
        checkpoint_mgr.wait_until_finished()

    if config["save_trajectories"] is not None:
        out_trajectories = {}
        out_ep_starts = []
        out_ep_ends = []
        num_steps_so_far = 0

        num_trajectories = 10000
        num_rollouts = num_trajectories // config["num_rollouts"]

        for _ in tqdm.trange(
            num_rollouts, dynamic_ncols=True, desc="Collecting trajectories"
        ):
            _, trajectories = do_rollout_vectorized(
                policy,
                env,
                policy_wrapper_fn=inference_schemes[0][1],
                max_rollout_length=config["max_episode_length"],
                video_fps=10,
                num_videos=0,
            )
            for i in range(trajectories["action"].shape[0]):
                valid = trajectories["valid"][i]
                trajectory = jax.tree.map(lambda x: x[i][valid], trajectories)

                # Flatten
                trajectory = {
                    **{
                        f"observation.{k.replace('observation.', '')}": v
                        for k, v in trajectory["observation"].items()
                    },
                    **{
                        f"next_observation.{k.replace('observation.', '')}": v
                        for k, v in trajectory["next_observation"].items()
                    },
                    "action": trajectory["action"],
                    "reward": trajectory["reward"],
                    "terminated": trajectory["terminated"],
                    "truncated": trajectory["truncated"],
                }

                if trajectory["terminated"][-1]:
                    for k, v in trajectory.items():
                        if k not in out_trajectories:
                            out_trajectories[k] = []
                        out_trajectories[k].append(v)
                    out_ep_starts.append(num_steps_so_far)
                    num_steps_so_far += len(trajectory["action"])
                    out_ep_ends.append(num_steps_so_far)

        out_trajectories = {k: np.concatenate(v) for k, v in out_trajectories.items()}
        out_ep_starts = np.array(out_ep_starts)
        out_ep_ends = np.array(out_ep_ends)
        print(
            f"Collected {len(out_ep_starts)}/{num_rollouts*config['num_rollouts']} valid trajectories "
        )

        np.savez(
            config["save_trajectories"],
            **out_trajectories,
            ep_starts=out_ep_starts,
            ep_ends=out_ep_ends,
        )


if __name__ == "__main__":
    config_flags.DEFINE_config_file("config", "causal_config.py")
    app.run(main)
