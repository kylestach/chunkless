from ml_collections import ConfigDict
from ml_collections.config_dict import placeholder, FieldReference

def get_config():
    d_embed = FieldReference(512, field_type=int)

    return ConfigDict({
        "wandb_project": "jax-diffusion-policy",
        "dt": 0.1,
        "num_train_steps": 200000,
        "max_episode_length": 300,
        "model": {
            "num_history_steps": 8,
            "model_kwargs": {
                "num_obs_tokens": 1,
                "d_embed": d_embed,
                "num_layers": 12,
                "num_heads": d_embed // 64,
                "mlp_ratio": 4,
                "action_condition_discrete": False,
                "hl_gauss_std": 0.005,
                "dropout_rate": 0.1,
                "alibi": False,
            }
        },
        "optimizer": {
            "learning_rate": 1e-3,
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