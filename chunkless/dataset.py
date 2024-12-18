import numpy as np
from torch.utils.data import Dataset, Sampler
from typing import Iterator, Union
import torch


class NpzTrajectoryDataset:
    def __init__(
        self, path, n_obs_history, n_ac_history, action_chunk_size, exec_horizon
    ):
        data = np.load(path)
        data = {k: np.asarray(v) for k, v in data.items()}
        self.trajectories = data
        num_frames = len(data["action"])
        num_episodes = len(data["ep_starts"])

        frame_id = np.arange(num_frames)

        ep_starts = self.trajectories.pop("ep_starts")
        ep_ends = self.trajectories.pop("ep_ends")
        self.ep_starts = ep_starts
        self.ep_ends = ep_ends

        is_start = np.isin(frame_id, ep_starts)
        is_end = np.roll(is_start, -1)
        traj_id = np.cumsum(is_start) - 1
        traj_length_by_traj = ep_ends - ep_starts

        self.traj_id_by_frame = traj_id
        self.traj_length_by_frame = traj_length_by_traj[traj_id]
        self.traj_start_by_frame = ep_starts[traj_id]
        self.traj_end_by_frame = ep_ends[traj_id]
        self.timestep_by_frame = np.arange(num_frames) - self.traj_start_by_frame
        self.time_remaining_by_frame = (
            self.traj_length_by_frame - self.timestep_by_frame
        )

        self.n_obs_history = n_obs_history
        self.n_ac_history = n_ac_history
        self.action_chunk_size = action_chunk_size
        self.exec_horizon = exec_horizon

    def __len__(self):
        return len(self.trajectories["action"])

    def __getitem__(self, idx):
        ep_start = self.traj_start_by_frame[idx]
        ep_end = self.traj_end_by_frame[idx]
        return {
            k: v[np.clip(np.arange(idx + 1 - self.n_obs_history, idx + 1), ep_start, ep_end - 1)]
            for k, v in self.trajectories.items()
            if k.startswith("observation")
        } | {
            "action": self.trajectories["action"][
                np.clip(
                    idx + 1 - self.n_ac_history + np.arange(self.action_chunk_size),
                    ep_start,
                    ep_end - 1,
                )
            ],
        }

    def weights(self):
        return self.time_remaining_by_frame > self.exec_horizon


class EpisodeAwareInfiniteSampler(Sampler):
    def __init__(
        self,
        episode_data_index: dict,
        episode_indices_to_use: Union[list, None] = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        shuffle: bool = False,
    ):
        """Sampler that optionally incorporates episode boundary information.

        Args:
            episode_data_index: Dictionary with keys 'from' and 'to' containing the start and end indices of each episode.
            episode_indices_to_use: List of episode indices to use. If None, all episodes are used.
                                    Assumes that episodes are indexed from 0 to N-1.
            drop_n_first_frames: Number of frames to drop from the start of each episode.
            drop_n_last_frames: Number of frames to drop from the end of each episode.
            shuffle: Whether to shuffle the indices.
        """
        indices = []
        for episode_idx, (start_index, end_index) in enumerate(
            zip(episode_data_index["from"], episode_data_index["to"], strict=True)
        ):
            if episode_indices_to_use is None or episode_idx in episode_indices_to_use:
                indices.extend(
                    range(
                        start_index.item() + drop_n_first_frames,
                        end_index.item() - drop_n_last_frames,
                    )
                )

        self.indices = indices
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[int]:
        while True:
            yield self.indices[torch.randint(0, len(self.indices), ())]

