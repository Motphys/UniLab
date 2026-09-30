"""Core lifecycle tests for the tensor-native TorchEnv owner."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, cast
from unittest.mock import MagicMock

import gymnasium as gym
import numpy as np
import pytest
import torch
from unisim.backend.base import (
    SimBackend,
    TensorDataPlane,
    TensorExecution,
    TensorLifecycleCapabilities,
)

from unilab.base.base import EnvCfg, EnvPlayCapabilities
from unilab.base.scene import SceneCfg
from unilab.base.torch_env import TorchEnv, TorchEnvState
from unilab.training.tensor_diagnostics import NanGuardCfg, TensorNanGuard


@dataclass
class _StubCfg(EnvCfg):
    max_episode_seconds: float | None = 1.0
    ctrl_dt: float = 0.1
    sim_dt: float = 0.1
    tensor_runtime: bool = False


def _backend() -> MagicMock:
    backend = MagicMock(spec=SimBackend)
    backend.backend_type = "mujoco"
    backend.num_actuators = 4
    backend.tensor_execution.return_value = TensorExecution.HOST_BRIDGE
    backend.get_tensor_capabilities.return_value = TensorLifecycleCapabilities(
        execution=TensorExecution.HOST_BRIDGE,
        state_fields=frozenset({"qpos", "qvel"}),
        stepping=True,
        selected_reset=True,
        packed_host_bridge=True,
        data_plane=TensorDataPlane.HOST_BRIDGE,
        stream_event_ownership="test",
        torch_devices=("cpu",),
    )
    backend.step_tensor.return_value = None
    backend.get_play_capabilities.return_value = EnvPlayCapabilities(
        supports_physics_state_playback=False
    )
    return backend


def _unsupported_backend() -> MagicMock:
    backend = _backend()
    backend.tensor_execution.return_value = TensorExecution.UNSUPPORTED
    backend.get_tensor_capabilities.return_value = TensorLifecycleCapabilities(
        execution=TensorExecution.UNSUPPORTED
    )
    backend.step.return_value = None
    return backend


class _StubTorchEnv(TorchEnv):
    OBS_SPEC = {"obs": 3, "critic": 2}

    def __init__(
        self,
        num_envs: int = 3,
        *,
        cfg: EnvCfg | None = None,
        backend: SimBackend | None = None,
        device: str = "cpu",
        terminate: bool = True,
    ):
        super().__init__(cfg or _StubCfg(), backend or _backend(), num_envs, device=device)
        self.terminate = terminate
        self.reset_rows: list[torch.Tensor] = []

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return self.OBS_SPEC

    @property
    def action_space(self) -> gym.Space:
        return gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)

    def apply_action(self, actions: torch.Tensor, state: TorchEnvState) -> torch.Tensor:
        return actions * 2.0

    def update_state(self, state: TorchEnvState) -> TorchEnvState:
        return state.replace(
            obs={
                "obs": torch.ones((self.num_envs, 3), dtype=torch.float32, device=self.device),
                "critic": torch.full(
                    (self.num_envs, 2), 0.5, dtype=torch.float32, device=self.device
                ),
            },
            reward=torch.full((self.num_envs,), 2.0, dtype=torch.float32, device=self.device),
            terminated=torch.full(
                (self.num_envs,),
                self.terminate,
                dtype=torch.bool,
                device=self.device,
            ),
            truncated=torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device),
        )

    def reset(
        self, env_indices: torch.Tensor | None = None
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        rows = self._normalize_reset_indices(env_indices)
        self.reset_rows.append(rows.clone())
        count = rows.numel()
        return (
            {
                "obs": torch.full((count, 3), -1.0, dtype=torch.float32, device=self.device),
                "critic": torch.full((count, 2), -0.5, dtype=torch.float32, device=self.device),
            },
            {"row_metric": torch.arange(count, dtype=torch.float32, device=self.device)},
        )


def _actions(env: _StubTorchEnv) -> torch.Tensor:
    return torch.zeros((env.num_envs, 4), dtype=torch.float32)


def test_scene_visual_model_file_reads_backend_getter() -> None:
    backend = _backend()
    backend.get_scene_visual_model_file.return_value = "/visual.xml"
    env = _StubTorchEnv(backend=backend)

    assert env.get_scene_visual_model_file() == "/visual.xml"


def test_init_state_uses_tensor_dict_obs_and_selected_reset() -> None:
    env = _StubTorchEnv()
    state = env.init_state()

    assert isinstance(state, TorchEnvState)
    assert set(state.obs) == {"obs", "critic"}
    assert all(isinstance(value, torch.Tensor) for value in state.obs.values())
    assert state.reward.shape == (3,)
    assert state.reward.dtype == torch.float32
    assert state.terminated.dtype == torch.bool
    assert state.truncated.dtype == torch.bool
    assert state.info["steps"].dtype == torch.int64
    torch.testing.assert_close(state.obs["obs"], torch.full((3, 3), -1.0))
    torch.testing.assert_close(state.obs["critic"], torch.full((3, 2), -0.5))


def test_state_replace_preserves_shared_buffers_by_reference() -> None:
    env = _StubTorchEnv()
    state = env.init_state()
    replacement = state.replace(reward=torch.ones(3))

    assert replacement is not state
    assert replacement.obs is state.obs
    assert replacement.info is state.info
    assert replacement.reward is not state.reward


def test_step_uses_backend_tensor_contract_and_autoreset() -> None:
    backend = _backend()
    env = _StubTorchEnv(backend=backend)
    env.init_state()
    env.reset_rows.clear()
    actions = _actions(env)

    state = env.step(actions)

    backend.step_tensor.assert_called_once()
    assert backend.step_tensor.call_args.args[1] == 1
    torch.testing.assert_close(backend.step_tensor.call_args.args[0], actions * 2.0)
    assert [rows.tolist() for rows in env.reset_rows] == [[0, 1, 2]]
    torch.testing.assert_close(state.reward, torch.full((3,), 2.0))
    torch.testing.assert_close(state.obs["obs"], torch.full((3, 3), -1.0))
    assert state.final_observation is not None
    torch.testing.assert_close(state.final_observation["obs"], torch.ones((3, 3)))
    torch.testing.assert_close(state.info["steps"], torch.zeros(3, dtype=torch.int64))
    assert state.info["row_metric"].tolist() == [0.0, 1.0, 2.0]
    assert state.info["timing"]["reset_done_count"] == 3.0
    for key in (
        "apply_action_ms",
        "apply_action_cpu_ms",
        "step_core_ms",
        "step_core_cpu_ms",
        "update_state_ms",
        "update_state_cpu_ms",
        "reset_done_ms",
        "reset_done_cpu_ms",
        "env_step_other_cpu_ms",
    ):
        assert isinstance(state.info["timing"][key], float)
        assert state.info["timing"][key] >= 0.0
    assert "final_observation" not in state.info
    assert "_final_observation" not in state.info


def test_legacy_runtime_uses_public_numpy_backend_step() -> None:
    backend = _unsupported_backend()
    env = _StubTorchEnv(cfg=_StubCfg(tensor_runtime=False), backend=backend)
    env.init_state()
    actions = torch.ones((env.num_envs, 4), dtype=torch.float32)

    env.step(actions)

    backend.step_tensor.assert_not_called()
    backend.step.assert_called_once()
    np.testing.assert_array_equal(backend.step.call_args.args[0], actions.numpy() * 2.0)
    assert backend.step.call_args.args[1] == 1


def test_unsupported_backend_tensor_runtime_fails_closed() -> None:
    cfg = _StubCfg()
    cfg.tensor_runtime = True
    env = _StubTorchEnv(cfg=cfg, backend=_unsupported_backend())

    with pytest.raises(ValueError, match="does not declare a tensor lifecycle"):
        env.init_state()


def test_step_without_done_does_not_create_final_observation() -> None:
    env = _StubTorchEnv(terminate=False)
    env.init_state()
    env.reset_rows.clear()
    state = env.step(_actions(env))

    assert env.reset_rows == []
    assert state.final_observation is None
    assert state.info["timing"]["reset_done_count"] == 0.0


def test_partial_autoreset_preserves_non_done_rows() -> None:
    class _PartialEnv(_StubTorchEnv):
        def update_state(self, state: TorchEnvState) -> TorchEnvState:
            result = super().update_state(state)
            terminated = torch.tensor([True, False, True])
            return result.replace(terminated=terminated)

    env = _PartialEnv()
    env.init_state()
    env.reset_rows.clear()
    state = env.step(_actions(env))

    assert [rows.tolist() for rows in env.reset_rows] == [[0, 2]]
    torch.testing.assert_close(state.obs["obs"], torch.tensor([[-1.0] * 3, [1.0] * 3, [-1.0] * 3]))
    torch.testing.assert_close(state.info["steps"], torch.tensor([0, 1, 0], dtype=torch.int64))
    assert state.final_observation is not None
    torch.testing.assert_close(state.final_observation["obs"][[0, 2]], torch.ones((2, 3)))


def test_manual_autoreset_disabled_keeps_terminal_state() -> None:
    env = _StubTorchEnv()
    env.init_state()
    env.reset_rows.clear()
    env.set_autoreset(False)
    state = env.step(_actions(env))

    assert env.reset_rows == []
    assert state.final_observation is None
    torch.testing.assert_close(state.obs["obs"], torch.ones((3, 3)))
    torch.testing.assert_close(state.info["steps"], torch.ones(3, dtype=torch.int64))


def test_reset_indices_are_normalized_and_validated() -> None:
    env = _StubTorchEnv()
    assert env._normalize_reset_indices(None).tolist() == [0, 1, 2]
    assert env._normalize_reset_indices(torch.tensor([2, 0])).tolist() == [2, 0]
    with pytest.raises(ValueError, match="unique"):
        env._normalize_reset_indices(torch.tensor([0, 0]))
    with pytest.raises(ValueError, match=r"\[0, 3\)"):
        env._normalize_reset_indices(torch.tensor([3]))


def test_timeout_computes_truncation_and_selected_reset() -> None:
    env = _StubTorchEnv(cfg=_StubCfg(max_episode_seconds=0.2), terminate=False)
    env.init_state()
    env.reset_rows.clear()

    first = env.step(_actions(env))
    second = env.step(_actions(env))

    assert not bool(first.truncated.any())
    assert bool(second.truncated.any())
    assert [rows.tolist() for rows in env.reset_rows] == [[0, 1, 2]]


def test_numpy_actions_and_wrong_device_fail_closed() -> None:
    env = _StubTorchEnv()

    with pytest.raises(TypeError, match="torch.Tensor"):
        env.step(cast(Any, np.zeros((env.num_envs, 4), dtype=np.float32)))
    assert env.reset_rows == []
    assert env.state is None
    with pytest.raises(ValueError, match="shape"):
        env.step(torch.zeros((env.num_envs, 5)))
    with pytest.raises(TypeError, match="dtype"):
        env.step(torch.zeros((env.num_envs, 4), dtype=torch.float64))
    with pytest.raises(ValueError, match="contiguous"):
        env.step(torch.zeros((env.num_envs, 8))[:, ::2])
    with pytest.raises(ValueError, match="NaN or Inf"):
        env.step(torch.full((env.num_envs, 4), torch.nan))


def test_nonfinite_reward_fails_closed_without_sanitization() -> None:
    env = _StubTorchEnv()
    env.init_state()

    original_update = env.update_state

    def bad_update(state: TorchEnvState) -> TorchEnvState:
        result = original_update(state)
        return result.replace(reward=torch.full_like(result.reward, torch.nan))

    env.update_state = bad_update  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="reward contains NaN or Inf"):
        env.step(_actions(env))


def test_invalid_backend_control_fails_closed() -> None:
    class _BadControlEnv(_StubTorchEnv):
        def apply_action(self, actions: torch.Tensor, state: TorchEnvState) -> torch.Tensor:
            return actions[:, :3]

    env = _BadControlEnv()
    env.init_state()
    with pytest.raises(ValueError, match="control shape"):
        env.step(_actions(env))


def test_nonfinite_backend_control_fails_closed() -> None:
    class _NonfiniteControlEnv(_StubTorchEnv):
        def apply_action(self, actions: torch.Tensor, state: TorchEnvState) -> torch.Tensor:
            return actions * torch.inf

    env = _NonfiniteControlEnv()
    env.init_state()
    with pytest.raises(ValueError, match="control contains NaN or Inf"):
        env.step(_actions(env))


def test_tensor_nan_guard_detects_and_dumps_at_explicit_host_boundary(tmp_path) -> None:
    class _NonfiniteObsEnv(_StubTorchEnv):
        def update_state(self, state: TorchEnvState) -> TorchEnvState:
            result = super().update_state(state)
            obs = {key: value.clone() for key, value in result.obs.items()}
            obs["critic"][1, 0] = torch.nan
            return result.replace(obs=obs)

    env = _NonfiniteObsEnv()
    env.init_state()
    guard = TensorNanGuard(
        NanGuardCfg(enabled=True, buffer_size=2, max_envs_to_dump=2, output_dir=str(tmp_path)),
        num_envs=env.num_envs,
        supports_state_playback=False,
    )
    env.set_nan_guard(guard)

    with pytest.raises(ValueError, match="obs\\['critic'\\] contains NaN or Inf"):
        env.step(_actions(env))
    assert list(tmp_path.glob("nan_dump_*.npz")) == []

    class _FiniteObsEnv(_StubTorchEnv):
        pass

    finite_env = _FiniteObsEnv(terminate=False)
    finite_env.init_state()
    finite_guard = TensorNanGuard(
        NanGuardCfg(enabled=True, buffer_size=2, output_dir=str(tmp_path / "unused")),
        num_envs=finite_env.num_envs,
        supports_state_playback=False,
    )
    finite_env.set_nan_guard(finite_guard)
    finite_state = finite_env.step(_actions(finite_env))
    assert finite_state.info["timing"]["env_step_total_ms"] >= 0.0
    assert list(tmp_path.glob("nan_dump_*.npz")) == []

    def finite_update(state: TorchEnvState) -> TorchEnvState:
        return state.replace(
            obs={
                "obs": torch.zeros((3, 3), dtype=torch.float32),
                "critic": torch.zeros((3, 2), dtype=torch.float32),
            },
            reward=torch.tensor([0.0, torch.nan, 0.0]),
            terminated=torch.zeros(3, dtype=torch.bool),
            truncated=torch.zeros(3, dtype=torch.bool),
        )

    finite_env.update_state = finite_update  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="reward contains NaN or Inf"):
        finite_env.step(_actions(finite_env))
    unused_files = [
        path for path in (tmp_path / "unused").glob("nan_dump_*.npz") if not path.is_symlink()
    ]
    assert len(unused_files) == 1
    dump = np.load(unused_files[0])
    assert dump["states"].shape == (0, 1, 0)
    assert dump["meta_nan_env_ids"].tolist() == [1]
    assert (tmp_path / "unused" / "nan_dump_latest.npz").is_symlink()
    assert dump["meta_num_envs_total"].item() == finite_env.num_envs
    assert dump["meta_supports_state_playback"].item() is False


def test_tensor_nan_guard_keeps_detection_input_strict() -> None:
    guard = TensorNanGuard(NanGuardCfg(enabled=True), num_envs=3, supports_state_playback=False)

    with pytest.raises(TypeError, match="observations must be a mapping"):
        guard.check(cast(Any, [torch.zeros((3, 2))]), torch.zeros(3))
    with pytest.raises(TypeError, match="obs\\['obs'\\] must be a torch.Tensor"):
        guard.check({"obs": cast(Any, np.zeros((3, 2), dtype=np.float32))}, torch.zeros(3))
    with pytest.raises(TypeError, match="reward must be a torch.Tensor"):
        guard.check({"obs": torch.zeros((3, 2))}, cast(Any, np.zeros(3, dtype=np.float32)))
    with pytest.raises(TypeError, match="ctrl must be a torch.Tensor"):
        guard.check_ctrl(cast(Any, np.zeros((3, 2), dtype=np.float32)))
    with pytest.raises(TypeError, match="nan_env_ids must be a torch.Tensor"):
        guard.dump(cast(Any, np.array([0])), "", 0)

    ids = guard.check(
        {"obs": torch.tensor([[0.0, torch.nan], [1.0, 2.0], [3.0, 4.0]])},
        torch.tensor([0.0, torch.nan, 2.0]),
    )
    assert isinstance(ids, torch.Tensor)
    assert ids.dtype == torch.int64
    assert ids.tolist() == [1]


def test_tensor_nan_guard_rejects_legacy_guard_and_resolves_model_lazily() -> None:
    backend = _backend()
    backend.get_scene_model_file.return_value = "/backend.xml"
    env = _StubTorchEnv(cfg=_StubCfg(scene=SceneCfg(model_file="/scene.xml")), backend=backend)
    assert backend.get_scene_model_file.call_count == 0

    with pytest.raises(TypeError, match="TorchEnv nan guard must be TensorNanGuard"):
        env.set_nan_guard(cast(Any, MagicMock()))

    cfg = _StubCfg(scene=SceneCfg(model_file=""))
    backend_without_scene = _backend()
    backend_without_scene.get_scene_model_file.return_value = "/backend.xml"
    lazy_env = _StubTorchEnv(cfg=cfg, backend=backend_without_scene)
    assert backend_without_scene.get_scene_model_file.call_count == 0
    backend_without_scene.get_scene_model_file.return_value = "/changed.xml"
    lazy_guard = TensorNanGuard(NanGuardCfg(enabled=True), 3, False)
    lazy_env.set_nan_guard(lazy_guard)
    lazy_env.init_state()
    assert backend_without_scene.get_scene_model_file.call_count == 0


def test_backend_capability_mismatch_fails_closed() -> None:
    backend = _backend()
    capabilities = backend.get_tensor_capabilities.return_value
    backend.get_tensor_capabilities.return_value = dataclasses.replace(capabilities, stepping=False)
    env = _StubTorchEnv(backend=backend)
    with pytest.raises(ValueError, match="stepping and selected reset"):
        env.init_state()

    backend.tensor_execution.return_value = TensorExecution.DEVICE_RESIDENT
    env = _StubTorchEnv(backend=backend)
    with pytest.raises(ValueError, match="does not match its capabilities"):
        env.init_state()


def test_invalid_final_observation_fails_closed() -> None:
    env = _StubTorchEnv()
    env.init_state()

    original_update = env.update_state

    def bad_update(state: TorchEnvState) -> TorchEnvState:
        result = original_update(state)
        return result.replace(
            final_observation={
                "obs": torch.ones((2, 3)),
                "critic": torch.ones((2, 2)),
            }
        )

    env.update_state = bad_update  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="final_observation.*shape"):
        env.step(_actions(env))


def test_device_resident_backend_requires_cuda() -> None:
    backend = _backend()
    backend.tensor_execution.return_value = TensorExecution.DEVICE_RESIDENT
    backend.get_tensor_capabilities.return_value = TensorLifecycleCapabilities(
        execution=TensorExecution.DEVICE_RESIDENT,
        state_fields=frozenset({"qpos", "qvel"}),
        stepping=True,
        selected_reset=True,
        data_plane=TensorDataPlane.DIRECT,
        stream_event_ownership="test",
        torch_devices=("cpu",),
    )
    cfg = _StubCfg()
    cfg.tensor_runtime = True
    env = _StubTorchEnv(cfg=cfg, backend=backend, device="cpu")
    with pytest.raises(ValueError, match="requires a CUDA device"):
        env.init_state()


def test_backend_device_must_match_capability() -> None:
    backend = _backend()
    backend.get_tensor_capabilities.return_value = TensorLifecycleCapabilities(
        execution=TensorExecution.HOST_BRIDGE,
        state_fields=frozenset({"qpos", "qvel"}),
        stepping=True,
        selected_reset=True,
        packed_host_bridge=True,
        data_plane=TensorDataPlane.HOST_BRIDGE,
        stream_event_ownership="test",
        torch_devices=("cuda:0",),
    )
    env = _StubTorchEnv(backend=backend, device="cpu")
    with pytest.raises(ValueError, match="did not accept Torch device"):
        env.init_state()


def test_backend_reset_timing_hook_is_merged() -> None:
    class _TimedEnv(_StubTorchEnv):
        def _collect_reset_backend_timing_ms(self) -> dict[str, float]:
            return {"set_state_internal_gap_ms": 2.5}

    env = _TimedEnv()
    env.init_state()
    state = env.step(_actions(env))
    assert state.info["timing"]["set_state_internal_gap_ms"] == 2.5


def test_training_state_round_trip_and_close() -> None:
    backend = _backend()
    env = _StubTorchEnv(backend=backend)
    payload = {"version": 1, "step_counter": 17}
    env.import_training_state(payload)
    assert env.export_training_state() == payload

    with pytest.raises(ValueError, match="non-negative"):
        env.import_training_state({"version": 1, "step_counter": -1})

    env.close()
    backend.cleanup_scene_assets.assert_called_once_with()
