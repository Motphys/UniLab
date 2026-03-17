import torch

from unilab.base import registry


def get_default_device() -> str:
    """Detect the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_env_dims(
    env_name: str, sim_backend: str = "mujoco", env_cfg_override: dict | None = None
) -> tuple[int, int]:
    """Get observation and action dimensions from environment."""
    env = registry.make(env_name, num_envs=1, sim_backend=sim_backend, env_cfg_override=env_cfg_override)
    obs_dim = sum(env.obs_groups_spec.values())
    action_shape = env.action_space.shape
    assert action_shape is not None
    action_dim = action_shape[0]
    env.close()  # type: ignore[attr-defined]
    return obs_dim, action_dim
