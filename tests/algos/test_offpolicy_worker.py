from __future__ import annotations

import queue
import threading

import pytest
import torch

import unilab.algos.offpolicy.worker as worker_module
from unilab.algos.common.collector_timing import extract_env_step_breakdown_timing_ms
from unilab.algos.offpolicy.worker import (
    _publish_inference_tick,
    _wait_for_inference_tick,
    compute_collector_active_steps_per_sec,
    resolve_offpolicy_actor_priv_info,
    sample_offpolicy_actions,
)


def test_collector_binds_backend_device_before_env_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, ...]] = []

    class _EnvMaterializedError(RuntimeError):
        pass

    def configure(backend_type: str, device: str | None) -> str:
        events.append(("bind", backend_type, str(device)))
        return str(device)

    def make_env(name: str, **kwargs):
        events.append(("make", name, str(kwargs["sim_backend"])))
        raise _EnvMaterializedError

    from unilab.base import registry

    monkeypatch.setattr(worker_module, "configure_backend_process_device", configure)
    monkeypatch.setattr(worker_module, "apply_torch_thread_runtime", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker_module, "ensure_registries", lambda: None)
    monkeypatch.setattr(worker_module, "apply_training_seed", lambda *args, **kwargs: None)
    monkeypatch.setattr(registry, "make", make_env)

    with pytest.raises(_EnvMaterializedError):
        worker_module._run_collector(
            stop_event=None,
            env_name="DummyEnv",
            num_envs=2,
            replay_buffer=None,
            inference_slot=None,
            inference_request_queue=None,
            inference_response_queue=None,
            algo_type="sac",
            metrics_queue=None,
            sim_backend="mjwarp",
            backend_device="cuda:3",
            env_cfg_override=None,
            seed=None,
            trace_enabled=False,
            trace_thread_time=False,
        )

    assert events == [
        ("bind", "mjwarp", "cuda:3"),
        ("make", "DummyEnv", "mjwarp"),
    ]


def test_inference_request_publish_timeout_is_explicit() -> None:
    requests: queue.Queue[int] = queue.Queue(maxsize=1)
    requests.put_nowait(0)

    with pytest.raises(TimeoutError, match="publishing off-policy inference tick 1"):
        _publish_inference_tick(requests, 1, threading.Event(), timeout=0.01)


def test_inference_response_wait_timeout_is_explicit() -> None:
    with pytest.raises(TimeoutError, match="waiting for off-policy inference tick 0"):
        _wait_for_inference_tick(queue.Queue(), 0, threading.Event(), timeout=0.01)


class _DummyActor:
    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, torch.Tensor, bool]] = []

    def explore(
        self,
        obs: torch.Tensor,
        dones: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        assert dones is not None
        self.calls.append((obs.clone(), dones.clone(), deterministic))
        return torch.ones(obs.shape[0], 3, dtype=obs.dtype)


def test_compute_collector_active_steps_per_sec_includes_active_phases_only() -> None:
    steps_per_sec = compute_collector_active_steps_per_sec(
        {
            "inference_request_ms": 1.0,
            "learner_action_wait_ms": 100.0,
            "env_step_ms": 10.0,
            "replay_write_ms": 3.0,
            "bookkeeping_ms": 100.0,
        },
        num_envs=32,
    )

    assert steps_per_sec == pytest.approx(32 / 0.014)


def test_extract_env_step_breakdown_timing_ms_maps_env_owned_keys_only() -> None:
    timing = extract_env_step_breakdown_timing_ms(
        {
            "timing": {
                "step_core_ms": 1.5,
                "update_state_ms": 2.5,
                "reset_done_ms": 0.25,
                "apply_action_ms": 9.0,
            }
        }
    )

    assert timing == {
        "env_step_backend_ms": 1.5,
        "env_step_update_state_ms": 2.5,
        "env_step_reset_done_ms": 0.25,
    }


def test_compute_collector_active_steps_per_sec_returns_none_without_active_time() -> None:
    assert (
        compute_collector_active_steps_per_sec(
            {"bookkeeping_ms": 100.0},
            num_envs=32,
        )
        is None
    )


@pytest.mark.parametrize("algo_type", ["sac", "td3", "flashsac"])
def test_sample_offpolicy_actions_uses_actor_explore(algo_type: str) -> None:
    actor = _DummyActor()
    obs = torch.zeros(4, 5)
    dones = torch.zeros(4)

    actions = sample_offpolicy_actions(
        actor=actor,
        algo_type=algo_type,
        obs_torch=obs,
        prev_dones_torch=dones,
    )

    assert len(actor.calls) == 1
    assert actor.calls[0][2] is False
    assert actions.shape == (4, 3)


def test_sample_offpolicy_actions_rejects_unknown_algo() -> None:
    actor = _DummyActor()

    with pytest.raises(ValueError, match="learner action sampling"):
        sample_offpolicy_actions(
            actor=actor,
            algo_type="unknown",
            obs_torch=torch.zeros(2, 4),
            prev_dones_torch=torch.zeros(2),
        )


class _DummyHoraActor:
    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, torch.Tensor, bool]] = []

    def explore(
        self,
        obs: torch.Tensor,
        priv_info: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        self.calls.append((obs.clone(), priv_info.clone(), deterministic))
        return torch.ones(obs.shape[0], 3, dtype=obs.dtype)


def test_sample_offpolicy_actions_passes_hora_priv_info() -> None:
    actor = _DummyHoraActor()
    obs = torch.zeros(4, 5)
    priv_info = torch.randn(4, 2)

    actions = sample_offpolicy_actions(
        actor=actor,
        algo_type="hora_sac",
        obs_torch=obs,
        prev_dones_torch=torch.zeros(4),
        priv_info_torch=priv_info,
    )

    assert actions.shape == (4, 3)
    assert len(actor.calls) == 1
    torch.testing.assert_close(actor.calls[0][1], priv_info)


def test_resolve_offpolicy_actor_priv_info_prefers_explicit_info() -> None:
    import numpy as np

    obs = np.zeros((2, 3), dtype=np.float32)
    critic_tail = np.ones((2, 2), dtype=np.float32)
    critic = np.concatenate([obs, critic_tail], axis=1)
    explicit = np.full((2, 2), 7.0, dtype=np.float32)

    resolved = resolve_offpolicy_actor_priv_info(
        algo_type="hora_sac",
        obs_np=obs,
        critic_np=critic,
        info={"critic_info": explicit},
    )

    np.testing.assert_allclose(resolved, explicit)


def test_resolve_offpolicy_actor_priv_info_uses_critic_tail() -> None:
    import numpy as np

    obs = np.zeros((2, 3), dtype=np.float32)
    critic_tail = np.arange(4, dtype=np.float32).reshape(2, 2)
    critic = np.concatenate([obs, critic_tail], axis=1)

    resolved = resolve_offpolicy_actor_priv_info(
        algo_type="hora_sac",
        obs_np=obs,
        critic_np=critic,
        info={},
    )

    np.testing.assert_allclose(resolved, critic_tail)
