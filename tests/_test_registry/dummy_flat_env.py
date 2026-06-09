"""DummyFlatTest env registration — importable both from pytest conftest (parent
process) and from the spawn collector subprocesses via
``UNILAB_EXTRA_REGISTRY_PACKAGES`` + ``ensure_registries``.

The env returns a ``NpEnvState``-shaped object on each step so that off-policy
collector loops (which expect ``state.obs / state.reward / state.terminated /
state.truncated / state.info``) can drive it through a complete learn cycle.
The dynamics are intentionally trivial: random obs, zero reward, never-done.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np

from unilab.base import registry
from unilab.base.base import ABEnv, EnvCfg, EnvPlayCapabilities

_DUMMY_OBS_DIM = 8
_DUMMY_ACT_DIM = 3
DUMMY_ENV_NAME = "DummyFlatTest"


@dataclass
class _DummyCfg(EnvCfg):
    pass


@dataclass
class _DummyState:
    """Minimal mirror of ``NpEnvState`` for the test env.

    Kept local to avoid coupling tests to NpEnv's internals while still
    satisfying the duck-typed contract used by collector workers.
    """

    obs: dict[str, np.ndarray]
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    info: dict[str, Any] = field(default_factory=dict)
    final_observation: dict[str, np.ndarray] | None = None

    def replace(self, **updates: Any) -> "_DummyState":
        return dataclasses.replace(self, **updates)


class _DummyEnv(ABEnv):
    """Minimal env stub: random obs, zero reward, never done.

    Matches the duck-typed contract expected by ``OffPolicyRunner`` and
    ``APPORunner`` collectors:

    * ``state`` must be ``None`` before ``init_state()`` is called.
    * ``init_state()`` populates the first observation.
    * ``step(actions)`` returns a state with ``obs / reward / terminated /
      truncated / info`` and updates ``self.state`` in place.
    * ``obs`` is a dict with at least the ``"obs"`` key (see
      ``unilab.base.observations.split_obs_dict``).
    """

    def __init__(self, cfg: _DummyCfg, num_envs: int = 1, backend_type: str = "mujoco"):
        self._cfg = cfg
        self._num_envs = num_envs
        self._obs_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(_DUMMY_OBS_DIM,), dtype=np.float32
        )
        self._act_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(_DUMMY_ACT_DIM,), dtype=np.float32
        )
        self._state: _DummyState | None = None
        self._rng = np.random.default_rng(0)
        # Optional NaN guard hook (set via set_nan_guard); see issue #584.
        self._nan_guard = None

    # ------------------------------------------------------------------ env shape
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
        return {"obs": _DUMMY_OBS_DIM}

    @property
    def play_capabilities(self) -> EnvPlayCapabilities:
        return EnvPlayCapabilities(supports_physics_state_playback=False)

    # ------------------------------------------------------------------ lifecycle
    @property
    def state(self):
        return self._state

    def init_state(self):
        obs = {
            "obs": self._rng.standard_normal((self._num_envs, _DUMMY_OBS_DIM)).astype(np.float32),
        }
        self._state = _DummyState(
            obs=obs,
            reward=np.zeros(self._num_envs, dtype=np.float32),
            terminated=np.zeros(self._num_envs, dtype=bool),
            truncated=np.zeros(self._num_envs, dtype=bool),
            info={},
            final_observation=None,
        )
        return self._state

    def step(self, actions: np.ndarray):
        if self._state is None:
            self.init_state()
        # NaN-guard hook for ctrl signal (mirrors NpEnv.step contract).
        if self._nan_guard is not None and hasattr(self._nan_guard, "check_ctrl"):
            self._nan_guard.check_ctrl(np.asarray(actions, dtype=np.float32))
        obs = {
            "obs": self._rng.standard_normal((self._num_envs, _DUMMY_OBS_DIM)).astype(np.float32),
        }
        self._state = _DummyState(
            obs=obs,
            reward=np.zeros(self._num_envs, dtype=np.float32),
            terminated=np.zeros(self._num_envs, dtype=bool),
            truncated=np.zeros(self._num_envs, dtype=bool),
            info={},
            final_observation=None,
        )
        return self._state

    def close(self) -> None:
        self._state = None

    # ------------------------------------------------------------------ NaN guard
    def set_nan_guard(self, guard) -> None:
        self._nan_guard = guard


def register() -> None:
    if not registry.contains(DUMMY_ENV_NAME):
        registry.register_env_config(DUMMY_ENV_NAME, _DummyCfg)
        registry.register_env(DUMMY_ENV_NAME, _DummyEnv, "mujoco")


# Auto-register on import so spawn subprocesses pick it up via
# ensure_registries(packages=["tests._test_registry"]) without further wiring.
register()
