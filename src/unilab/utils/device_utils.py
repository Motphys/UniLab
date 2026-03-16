import torch

from unilab.base import registry


def get_default_device() -> str:
    """Detect the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_env_dims(env_name: str, sim_backend: str = "mujoco") -> tuple[int, int]:
    """Get observation and action dimensions from environment."""
    env = registry.make(env_name, num_envs=1, sim_backend=sim_backend)
    obs_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    assert obs_shape is not None and action_shape is not None
    obs_dim = obs_shape[0]
    action_dim = action_shape[0]
    env.close()  # type: ignore[attr-defined]
    return obs_dim, action_dim
