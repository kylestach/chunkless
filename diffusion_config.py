from ml_collections import ConfigDict
from ml_collections.config_dict import placeholder

def get_config():
    return ConfigDict({
        "wandb_project": "jax-diffusion-policy",
        "dt": 0.1,
        "num_train_steps": 200000,
        "max_episode_length": 300,
        "model": {
            "state_encoder_num_layers": 0,
            "state_encoder_flatten_first": False,
            "num_history_steps": 2,
            "num_ac_history_steps": 2,
            "action_chunk_size": 16,
            "action_chunk_exec_size": 8,
        },
        "optimizer": {
            "learning_rate": 1e-4,
            "weight_decay": 1e-6,
            "b1": 0.95,
            "b2": 0.999,
            "ema_inv_gamma": 1.0,
            "ema_power": 0.75,
            "ema_min_value": 0.0,
            "ema_max_value": 0.9999,
        },
        "batch_size": 256,
        "warmup_steps": 1000,
        "sample_clip_bounds": 1,
        "num_denoising_steps": 100,
        "num_rollouts": 100,
        "save_trajectories": placeholder(str),
        "num_videos": 4,
        "checkpoint_dir": None,
        "load_aux_trajectories": placeholder(str),
    })