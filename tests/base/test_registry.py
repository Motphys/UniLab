"""Tests for the environment registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import pytest

import unilab.base.registry as registry_mod
from unilab.base.base import ABEnv, EnvCfg

# ---------------------------------------------------------------------------
# Helpers: local env stubs registered only for these tests
# ---------------------------------------------------------------------------
_TEST_ENV_A = "_TestRegistryEnvA"
_TEST_ENV_B = "_TestRegistryEnvB_Dup"
_TEST_ENV_C = "_TestRegistryEnvC_DefaultOrder"


@dataclass
class _TestCfgA(EnvCfg):
    pass


@dataclass
class _TestCfgB(EnvCfg):
    pass


class _TestEnvA(ABEnv):
    def __init__(self, cfg, num_envs=1, backend_type="mujoco"):
        self._cfg = cfg
        self._num_envs = num_envs

    @property
    def num_envs(self):
        return self._num_envs

    @property
    def cfg(self):
        return self._cfg

    @property
    def observation_space(self):
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

    @property
    def action_space(self):
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {"actor": 4}

    @property
    def state(self):
        return None

    def init_state(self):
        return None

    def step(self, actions):
        return None

    def close(self):
        pass


class _TestEnvMotrix(_TestEnvA):
    pass


# Register once at module level (idempotent guard)
if not registry_mod.contains(_TEST_ENV_A):
    registry_mod.register_env_config(_TEST_ENV_A, _TestCfgA)
    registry_mod.register_env(_TEST_ENV_A, _TestEnvA, "mujoco")

if not registry_mod.contains(_TEST_ENV_B):
    registry_mod.register_env_config(_TEST_ENV_B, _TestCfgB)

if not registry_mod.contains(_TEST_ENV_C):
    registry_mod.register_env_config(_TEST_ENV_C, _TestCfgA)
    registry_mod.register_env(_TEST_ENV_C, _TestEnvMotrix, "motrix")
    registry_mod.register_env(_TEST_ENV_C, _TestEnvA, "mujoco")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_envcfg_decorator_registers():
    _name = "_TestDecoratorEnv"
    if not registry_mod.contains(_name):

        @registry_mod.envcfg(_name)
        @dataclass
        class _DecCfg(EnvCfg):
            pass

    assert registry_mod.contains(_name)


def test_envcfg_decorator_registers_plain_factory_and_returns_it_unchanged():
    _name = "_TestDecoratorFactoryEnv"

    def make_cfg() -> EnvCfg:
        return _TestCfgA()

    registered = registry_mod.envcfg(_name)(make_cfg)

    assert registered is make_cfg
    assert registry_mod.materialize_env_config(_name) == _TestCfgA()


def test_materialize_env_config_rejects_invalid_factory_output():
    _name = "_TestInvalidFactoryOutput"

    def make_invalid_cfg():
        return {"sim_dt": 0.01}

    registry_mod.register_env_config(_name, make_invalid_cfg)

    with pytest.raises(
        TypeError,
        match=r"_TestInvalidFactoryOutput.*make_invalid_cfg.*dict.*EnvCfg",
    ):
        registry_mod.materialize_env_config(_name)


def test_register_env_config_rejects_non_callable():
    with pytest.raises(TypeError, match="config factory must be callable"):
        registry_mod.register_env_config("_TestNonCallableFactory", None)  # type: ignore[arg-type]


def test_env_decorator_registers():
    _name = "_TestDecoratorEnv2"
    if not registry_mod.contains(_name):
        registry_mod.register_env_config(_name, _TestCfgA)

        @registry_mod.env(_name, "mujoco")
        class _DecEnv(_TestEnvA):
            pass

    listed = registry_mod.list_registered_envs()
    assert _name in listed
    assert "mujoco" in listed[_name]["available_backends"]


def test_env_decorator_registers_plain_factory_and_returns_it_unchanged():
    _name = "_TestDecoratorEnvFactory"
    registry_mod.register_env_config(_name, _TestCfgA)

    def make_env(cfg: EnvCfg, num_envs: int = 1, backend_type: str = "mujoco") -> ABEnv:
        return _TestEnvA(cfg, num_envs=num_envs, backend_type=backend_type)

    registered = registry_mod.env(_name, "mujoco")(make_env)

    assert registered is make_env
    assert registry_mod._envs[_name].env_factory_dict["mujoco"] is make_env


def test_contains_before_and_after_registration():
    _name = "_TestContainsDynamic"
    assert not registry_mod.contains(_name)
    registry_mod.register_env_config(_name, _TestCfgA)
    assert registry_mod.contains(_name)


def test_duplicate_env_config_raises():
    """Registering the same env_name twice raises ValueError."""
    with pytest.raises(ValueError, match="already registered"):
        registry_mod.register_env_config(_TEST_ENV_A, _TestCfgA)


def test_find_available_sim_backend():
    backend = registry_mod.find_available_sim_backend(_TEST_ENV_A)
    assert backend == "mujoco"


def test_find_available_sim_backend_prefers_explicit_default_order():
    backend = registry_mod.find_available_sim_backend(_TEST_ENV_C)
    assert backend == "mujoco"


def test_find_available_sim_backend_missing_raises():
    with pytest.raises(ValueError):
        registry_mod.find_available_sim_backend("__nonexistent_env__")


def test_make_returns_correct_type():
    env = registry_mod.make(_TEST_ENV_A, sim_backend="mujoco")
    assert isinstance(env, _TestEnvA)


def test_make_unregistered_raises():
    with pytest.raises(ValueError):
        registry_mod.make("__nonexistent_env__")


def test_list_registered_envs_includes_registered():
    listed = registry_mod.list_registered_envs()
    assert _TEST_ENV_A in listed
    assert listed[_TEST_ENV_A]["config_factory"] == "_TestCfgA"
    assert "mujoco" in listed[_TEST_ENV_A]["available_backends"]


# ---------------------------------------------------------------------------
# register_env() validation branches
# ---------------------------------------------------------------------------


def test_register_env_invalid_backend_raises():
    """register_env() with an unsupported sim_backend must raise ValueError."""
    _name = "_TestInvalidBackendEnv"
    if not registry_mod.contains(_name):
        registry_mod.register_env_config(_name, _TestCfgA)
    with pytest.raises(ValueError, match="Unsupported simulation backend"):
        registry_mod.register_env(_name, _TestEnvA, "isaacgym")


def test_register_env_without_config_raises():
    """register_env() when config is not yet registered must raise ValueError."""
    with pytest.raises(ValueError, match="not registered"):
        registry_mod.register_env("__neverregistered__", _TestEnvA, "mujoco")


def test_register_env_duplicate_backend_raises():
    """Registering the same env + backend combination twice raises ValueError."""
    with pytest.raises(ValueError, match="already registered"):
        # _TEST_ENV_A + mujoco was already registered in module setup
        registry_mod.register_env(_TEST_ENV_A, _TestEnvA, "mujoco")


def test_register_env_rejects_non_callable_factory():
    _name = "_TestNonCallableEnvFactory"
    registry_mod.register_env_config(_name, _TestCfgA)

    with pytest.raises(TypeError, match=r"_TestNonCallableEnvFactory.*mujoco.*callable.*object"):
        registry_mod.register_env(_name, object(), "mujoco")  # type: ignore[arg-type]


def test_find_available_sim_backend_no_env_factory_raises():
    """find_available_sim_backend() raises when config exists but no env factory is registered."""
    # _TEST_ENV_B has config but no env factory (registered above without env)
    with pytest.raises(ValueError, match="does not support any simulation backend"):
        registry_mod.find_available_sim_backend(_TEST_ENV_B)


# ---------------------------------------------------------------------------
# make() validation branches
# ---------------------------------------------------------------------------


def test_make_auto_selects_backend():
    """make() with sim_backend=None selects the explicit default backend."""
    env = registry_mod.make(_TEST_ENV_A, sim_backend=None)
    assert isinstance(env, _TestEnvA)


def test_make_auto_selects_default_backend_independent_of_registration_order():
    env = registry_mod.make(_TEST_ENV_C, sim_backend=None)
    assert isinstance(env, _TestEnvA)


def test_make_calls_plain_factory_with_cfg_num_envs_and_backend():
    @dataclass
    class _CallableFactoryCfg(EnvCfg):
        ctrl_dt: float = 0.02

    _name = "_TestCallableEnvFactory"
    received: dict[str, object] = {}

    def make_env(cfg: EnvCfg, num_envs: int = 1, backend_type: str = "mujoco") -> ABEnv:
        received.update(cfg=cfg, num_envs=num_envs, backend_type=backend_type)
        return _TestEnvA(cfg, num_envs=num_envs, backend_type=backend_type)

    registry_mod.register_env_config(_name, _CallableFactoryCfg)
    registered = registry_mod.register_env(_name, make_env, "motrix")

    made = registry_mod.make(
        _name,
        sim_backend=None,
        env_cfg_override={"ctrl_dt": 0.05},
        num_envs=7,
    )

    assert registered is make_env
    assert isinstance(made, _TestEnvA)
    assert received["cfg"] is made.cfg
    assert received["num_envs"] == 7
    assert received["backend_type"] == "motrix"
    assert made.cfg.ctrl_dt == pytest.approx(0.05)


def test_make_rejects_invalid_factory_output_at_registry_boundary():
    _name = "_TestInvalidEnvFactoryOutput"

    def make_invalid_env(cfg: EnvCfg, num_envs: int = 1, backend_type: str = "mujoco") -> object:
        return object()

    registry_mod.register_env_config(_name, _TestCfgA)
    registry_mod.register_env(_name, make_invalid_env, "mujoco")  # type: ignore[arg-type]

    with pytest.raises(
        TypeError,
        match=r"_TestInvalidEnvFactoryOutput.*mujoco.*make_invalid_env.*object.*ABEnv",
    ):
        registry_mod.make(_name, sim_backend="mujoco")


def test_make_unsupported_backend_raises():
    """make() with an unsupported backend name raises ValueError."""
    with pytest.raises(ValueError, match="does not support simulation backend"):
        registry_mod.make(_TEST_ENV_A, sim_backend="motrix")


def test_make_no_env_factory_raises():
    """make() when no env factory is registered (only config) raises ValueError."""
    with pytest.raises(ValueError, match="does not support any simulation backend"):
        registry_mod.make(_TEST_ENV_B, sim_backend=None)


def test_make_with_valid_cfg_override():
    """make() applies valid config overrides."""

    @dataclass
    class _CfgWithField(EnvCfg):
        ctrl_dt: float = 0.02

    _name = "_TestOverrideEnv"
    if not registry_mod.contains(_name):
        registry_mod.register_env_config(_name, _CfgWithField)
        registry_mod.register_env(_name, _TestEnvA, "mujoco")

    env = registry_mod.make(_name, sim_backend="mujoco", env_cfg_override={"ctrl_dt": 0.05})
    assert env.cfg.ctrl_dt == 0.05


def test_make_materializes_fresh_function_owned_config_for_each_call():
    @dataclass
    class _FactoryCfg(EnvCfg):
        ctrl_dt: float = 0.02

    _name = "_TestFunctionFactoryOverrideEnv"

    def make_cfg() -> EnvCfg:
        return _FactoryCfg()

    registry_mod.register_env_config(_name, make_cfg)
    registry_mod.register_env(_name, _TestEnvA, "mujoco")

    overridden = registry_mod.make(
        _name,
        sim_backend="mujoco",
        env_cfg_override={"ctrl_dt": 0.05},
    )
    default = registry_mod.make(_name, sim_backend="mujoco")

    assert overridden.cfg.ctrl_dt == 0.05
    assert default.cfg.ctrl_dt == 0.02
    assert overridden.cfg is not default.cfg


def test_make_with_invalid_cfg_override_raises():
    """make() with a config key that doesn't exist raises ValueError."""
    with pytest.raises(ValueError, match="has no attribute"):
        registry_mod.make(_TEST_ENV_A, sim_backend="mujoco", env_cfg_override={"__bogus_key__": 1})
