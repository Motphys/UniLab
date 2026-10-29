"""Tests for the optional algo-capabilities extension point in env_contract."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.rl._fakes import FakeVecEnv, _FakePlayCapabilities, _FakeSpace
from unilab.base.contracts.env import (
    EnvAlgoCapabilities,
    EnvAlgoCapabilitiesProtocol,
    EnvProtocol,
    EnvStateProtocol,
    SupportsAlgoCapabilitiesProtocol,
    SupportsCompletionReceiptsProtocol,
    get_algo_capabilities,
    get_produces_completion_receipts,
)


def _make_env(algo_capabilities=None) -> FakeVecEnv:
    return FakeVecEnv(
        num_envs=2, obs_groups_spec={"obs": 4}, action_dim=3, algo_capabilities=algo_capabilities
    )


def test_get_algo_capabilities_default_when_env_does_not_provide():
    env = _make_env()
    assert not isinstance(env, SupportsAlgoCapabilitiesProtocol)

    caps = get_algo_capabilities(env)
    assert isinstance(caps, EnvAlgoCapabilitiesProtocol)
    assert caps.action_low is None
    assert caps.action_high is None
    assert caps.joint_names is None


def test_get_produces_completion_receipts_defaults_to_fail_closed():
    env = _make_env()
    assert not isinstance(env, SupportsCompletionReceiptsProtocol)
    # Undeclared envs keep the safe direction: CUDA ingress requires a receipt.
    assert get_produces_completion_receipts(env) is True


def test_get_produces_completion_receipts_honors_sync_declaration():
    class _SyncEnv:
        produces_completion_receipts = False

    class _AsyncEnv:
        @property
        def produces_completion_receipts(self):
            return True

    assert isinstance(_SyncEnv(), SupportsCompletionReceiptsProtocol)
    assert get_produces_completion_receipts(_SyncEnv()) is False
    assert get_produces_completion_receipts(_AsyncEnv()) is True


def test_get_produces_completion_receipts_rejects_non_bool_declaration():
    class _BadEnv:
        produces_completion_receipts = "no"

    with pytest.raises(TypeError, match="produces_completion_receipts"):
        get_produces_completion_receipts(_BadEnv())


def test_get_algo_capabilities_returns_provided_values():
    provided = EnvAlgoCapabilities(
        action_low=np.array([-1.0, -2.0, -3.0], dtype=np.float32),
        joint_names=("hip", "knee", "ankle"),
    )
    env = _make_env(algo_capabilities=provided)
    assert isinstance(env, SupportsAlgoCapabilitiesProtocol)

    caps = get_algo_capabilities(env)
    assert caps is provided
    np.testing.assert_array_equal(caps.action_low, provided.action_low)
    assert caps.action_high is None
    assert caps.joint_names == ("hip", "knee", "ankle")


def test_runtime_checkable_isinstance_semantics():
    assert isinstance(EnvAlgoCapabilities(), EnvAlgoCapabilitiesProtocol)

    class _Full:
        action_low = np.zeros((2,), dtype=np.float32)
        action_high = np.ones((2,), dtype=np.float32)
        joint_names = ("a", "b")
        reward_component_names = ("alive",)

    class _MissingField:
        action_low = np.zeros((2,), dtype=np.float32)

    # runtime_checkable protocols check attribute presence only.
    assert isinstance(_Full(), EnvAlgoCapabilitiesProtocol)
    assert not isinstance(_MissingField(), EnvAlgoCapabilitiesProtocol)
    assert not isinstance(object(), EnvAlgoCapabilitiesProtocol)
    assert not isinstance(object(), SupportsAlgoCapabilitiesProtocol)


def test_default_carrier_is_frozen():
    caps = EnvAlgoCapabilities()
    try:
        caps.action_low = np.zeros((1,), dtype=np.float32)  # type: ignore[misc]
    except AttributeError:
        pass
    else:  # pragma: no cover
        raise AssertionError("EnvAlgoCapabilities must be frozen")


def test_env_state_protocol_accepts_tensor_carriers() -> None:
    class _TensorState:
        obs = {"obs": torch.zeros(2, 3)}
        reward = torch.zeros(2)
        terminated = torch.zeros(2, dtype=torch.bool)
        truncated = torch.zeros(2, dtype=torch.bool)
        info: dict = {}
        final_observation = None

    assert isinstance(_TensorState(), EnvStateProtocol)


def test_env_protocol_requires_authoritative_device() -> None:
    class _NoDeviceEnv:
        num_envs = 1
        obs_groups_spec = {"obs": 1}
        observation_space = _FakeSpace((1,))
        action_space = _FakeSpace((1,))
        state = None
        cfg = object()
        play_capabilities = _FakePlayCapabilities()

        def init_state(self):
            return None

        def step(self, actions):
            return None

        def reset(self, env_indices):
            return {}, {}

        def set_nan_guard(self, guard):
            return None

        def close(self):
            return None

    assert not isinstance(_NoDeviceEnv(), EnvProtocol)
