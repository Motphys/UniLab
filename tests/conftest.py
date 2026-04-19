"""Shared fixtures for UniLab tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import pytest
import torch

from unilab.base import registry
from unilab.base.base import ABEnv, EnvCfg
from unilab.ipc.replay_buffer import ReplayBuffer
from unilab.ipc.shared_onpolicy_storage import SharedOnPolicyStorage
from unilab.ipc.weight_sync import SharedWeightSync

# ---------------------------------------------------------------------------
# Dummy flat env — no MuJoCo required
# ---------------------------------------------------------------------------
_DUMMY_OBS_DIM = 8
_DUMMY_ACT_DIM = 3
_DUMMY_ENV_NAME = "DummyFlatTest"


@dataclass
class _DummyCfg(EnvCfg):
    pass


class _DummyEnv(ABEnv):
    """Minimal env stub: random obs, zero reward, never done."""

    def __init__(self, cfg: _DummyCfg, num_envs: int = 1, backend_type: str = "mujoco"):
        self._cfg = cfg
        self._num_envs = num_envs
        self._obs_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(_DUMMY_OBS_DIM,), dtype=np.float32
        )
        self._act_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(_DUMMY_ACT_DIM,), dtype=np.float32
        )

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def cfg(self) -> EnvCfg:
        return self._cfg

    @property
    def observation_space(self) -> gym.Space:
        return self._obs_space

    @property
    def action_space(self) -> gym.Space:
        return self._act_space

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {"actor": _DUMMY_OBS_DIM}

    def close(self) -> None:
        pass

    @property
    def state(self):
        return None

    def init_state(self):
        return None

    def step(self, actions: np.ndarray):
        return None


def _register_dummy_env() -> None:
    if not registry.contains(_DUMMY_ENV_NAME):
        registry.register_env_config(_DUMMY_ENV_NAME, _DummyCfg)
        registry.register_env(_DUMMY_ENV_NAME, _DummyEnv, "mujoco")


_register_dummy_env()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mp_ctx():
    return torch.multiprocessing.get_context("spawn")


@pytest.fixture
def tiny_replay_buffer():
    buf = ReplayBuffer(
        capacity=128, obs_dim=_DUMMY_OBS_DIM, action_dim=_DUMMY_ACT_DIM, device="cpu"
    )
    yield buf


@pytest.fixture
def tiny_storage():
    storage = SharedOnPolicyStorage(
        num_envs=4,
        num_steps=10,
        obs_dim=_DUMMY_OBS_DIM,
        action_dim=_DUMMY_ACT_DIM,
        num_slots=2,
        create=True,
    )
    yield storage
    storage.cleanup()


@pytest.fixture
def tiny_weight_shapes():
    """Small MLP param shapes dict — linear(8,16) + bias, linear(16,3) + bias."""
    return {
        "layer1.weight": torch.Size([16, 8]),
        "layer1.bias": torch.Size([16]),
        "layer2.weight": torch.Size([3, 16]),
        "layer2.bias": torch.Size([3]),
    }


@pytest.fixture
def mock_env_name() -> str:
    return _DUMMY_ENV_NAME


@pytest.fixture
def default_go1_reward_config():
    """Default reward config for Go1 testing."""
    return {
        "scales": {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -5.0,
            "ang_vel_xy": -0.1,
            "base_height": -100.0,
            "action_rate": -0.005,
            "similar_to_default": -0.1,
            "contact": 0.24,
        },
        "tracking_sigma": 0.25,
        "base_height_target": 0.3,
    }


@pytest.fixture
def default_go2_reward_config():
    """Default reward config for Go2 testing."""
    return {
        "scales": {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -5.0,
            "ang_vel_xy": -0.1,
            "base_height": -100.0,
            "action_rate": -0.005,
            "similar_to_default": -0.1,
            "contact": 0.24,
            "swing_feet_z": 4.0,
        },
        "tracking_sigma": 0.25,
        "base_height_target": 0.3,
    }


@pytest.fixture
def default_g1_reward_config():
    """Default reward config for G1 testing."""
    return {
        "scales": {
            "tracking_lin_vel": 2.0,
            "tracking_ang_vel": 0.25,
            "forward_progress": 0.0,
            "under_speed": -0.2,
            "upper_body_pose": -0.05,
            "penalty_feet_ori": 0.0,
            "feet_phase": 1.0,
            "feet_phase_contrast": 1.0,
            "feet_phase_contact": 0.5,
            "feet_double_stance": -0.5,
            "lin_vel_z": -1.0,
            "ang_vel_xy": -0.2,
            "base_height": -120.0,
            "orientation": -2.5,
            "action_rate": -0.005,
            "pose": -0.05,
        },
        "tracking_sigma": 0.25,
        "gait_frequency": 1.5,
        "feet_phase_swing_height": 0.09,
        "feet_phase_tracking_sigma": 0.008,
        "base_height_target": 0.765,
        "min_forward_speed_for_gait_reward": 0.05,
        "min_base_height": 0.5,
        "max_tilt_deg": 35.0,
        "pose_weights": [
            0.01,
            1.0,
            5.0,
            0.01,
            5.0,
            5.0,
            0.01,
            1.0,
            5.0,
            0.01,
            5.0,
            5.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
        ],
    }


@pytest.fixture
def default_allegro_reward_config():
    """Default reward config for AllegroInhandRotation testing."""
    return {
        "scales": {
            "rotate": 1.25,
            "obj_linvel": -0.3,
            "pose_diff": -0.3,
            "torque": -0.1,
            "work": -2.0,
            "drop": 0.0,
        },
        "angvel_clip_min": -0.5,
        "angvel_clip_max": 0.5,
        "reset_z_threshold": 0.125,
    }


@pytest.fixture
def default_g1_walk_flat_reward_config():
    """Default reward config for G1 SAC testing."""
    return {
        "scales": {
            "tracking_lin_vel": 2.0,
            "tracking_ang_vel": 1.5,
            "penalty_ang_vel_xy": -1.0,
            "penalty_orientation": -10.0,
            "penalty_action_rate": -2.0,
            "pose": -0.5,
            "penalty_feet_ori": -25.0,
            "feet_phase": 5.0,
            "alive": 10.0,
        },
        "tracking_sigma": 0.25,
        "base_height_target": 0.754,
        "min_base_height": 0.3,
        "max_tilt_deg": 65.0,
        "gait_frequency": 1.5,
        "feet_phase_swing_height": 0.09,
        "feet_phase_tracking_sigma": 0.008,
        "close_feet_threshold": 0.15,
        "pose_weights": [
            0.01,
            1.0,
            5.0,
            0.01,
            5.0,
            5.0,
            0.01,
            1.0,
            5.0,
            0.01,
            5.0,
            5.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
        ],
    }
