"""Deterministic semantic-parity matrix for state-update graph execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import torch
from unisim import GraphCompletionToken, GraphLifecycleState

from tests.envs.test_manager_based_rl_env import _CommandCfg, _make_cfg, _make_env
from unilab.base.env_update_graph import validate_env_update_graph_manifest

MODES = ("eager", "deferred", "cuda_deferred", "graph")


class FakeEnvUpdateGraphRuntime:
    """Deterministic replay double that replays the extracted owner body."""

    def __init__(self) -> None:
        self.stream_id = 17
        self.capture_id = "gate-u-env-update-capture"
        self.layout: dict[str, Any] | None = None
        self.replays = 0

    def capture(self, verdict, body, *, current_layout):
        # Recording the callable is sufficient for this deterministic double;
        # executing it would duplicate the first eager boundary.
        del verdict, body
        self.layout = current_layout
        return self.capture_id, current_layout

    def replay(self, receipt, body):
        del receipt
        body()
        self.replays += 1
        return GraphCompletionToken(
            contract_version=1,
            lifecycle_state=GraphLifecycleState.READY,
            operation="env_update",
            sequence=self.replays,
            stream_id=self.stream_id,
            capture_id=self.capture_id,
        )


def _move_tensor(value: Any, target: torch.device) -> Any:
    return value.to(device=target) if torch.is_tensor(value) else value


def _move_mapping(mapping: dict[Any, Any], target: torch.device) -> dict[Any, Any]:
    moved: dict[Any, Any] = {}
    for key, value in mapping.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device=target)
        elif isinstance(value, list):
            moved[key] = [_move_tensor(item, target) for item in value]
        elif isinstance(value, dict):
            moved[key] = _move_mapping(value, target)
        else:
            moved[key] = value
    return moved


def _move_carriers(env: Any, target: torch.device) -> None:
    """Move owner/public carriers without rebuilding a production environment."""
    object.__setattr__(env, "_device", target)
    for name, value in list(vars(env).items()):
        if torch.is_tensor(value):
            object.__setattr__(env, name, value.to(device=target))
        elif isinstance(value, dict) and all(torch.is_tensor(item) for item in value.values()):
            object.__setattr__(
                env, name, {key: item.to(device=target) for key, item in value.items()}
            )
    env._state = env._state.replace(
        obs={name: value.to(device=target) for name, value in env._state.obs.items()},
        reward=env._state.reward.to(device=target),
        terminated=env._state.terminated.to(device=target),
        truncated=env._state.truncated.to(device=target),
    )
    env._state.info["steps"] = env._state.info["steps"].to(device=target)
    env._public_obs = {name: value.to(device=target) for name, value in env._public_obs.items()}
    env._final_observation_scratch = {
        name: value.to(device=target) for name, value in env._final_observation_scratch.items()
    }
    # Seed 7 matches every mode before construction; graph configuration's
    # acceptance replay consumes one additional CUDA generator draw in this fake
    # rig. Keep the CUDA deferred reference at that same state.
    env._torch_rng_owner = type(env.torch_rng)(device=target).manual_seed(7)
    env.torch_rng = env._torch_rng_owner
    env._torch_generator = env._torch_rng_owner.generator

    for manager_name in (
        "action_manager",
        "command_manager",
        "reward_manager",
        "termination_manager",
        "observation_manager",
        "event_manager",
    ):
        manager = getattr(env, manager_name)
        if hasattr(manager, "_device"):
            manager._device = target
        for term in getattr(manager, "_terms", {}).values():
            term._device = target
            term._torch_rng = env.torch_rng
        for name, value in list(vars(manager).items()):
            if torch.is_tensor(value):
                object.__setattr__(manager, name, value.to(device=target))
            elif isinstance(value, dict):
                object.__setattr__(manager, name, _move_mapping(value, target))
            elif isinstance(value, list):
                object.__setattr__(manager, name, [_move_tensor(item, target) for item in value])
        terms = getattr(manager, "_terms", {})
        if isinstance(terms, dict):
            for term in terms.values():
                for name, value in list(vars(term).items()):
                    if torch.is_tensor(value):
                        object.__setattr__(term, name, value.to(device=target))

    env._reset_state.declare_packed_reset_device(target)
    for name, value in list(vars(env._reset_state).items()):
        if torch.is_tensor(value):
            object.__setattr__(env._reset_state, name, value.to(device=target))
    for entity in env.scene.values():
        data = entity.data
        object.__setattr__(data, "_control_buffer", data._control_buffer.to(device=target))
        for name, value in list(vars(data).items()):
            if torch.is_tensor(value):
                object.__setattr__(data, name, value.to(device=target))

    backend = env._backend
    backend.qpos = backend.qpos.to(device=target)
    backend.qvel = backend.qvel.to(device=target)
    backend.sensors = {name: value.to(device=target) for name, value in backend.sensors.items()}


def _make_parity_env(mode: str) -> tuple[Any, Any, FakeEnvUpdateGraphRuntime]:
    cfg = _make_cfg(include_optional_managers=False)
    cfg.events = {}
    cfg.commands = {"target": _CommandCfg(resampling_time_range=(10.0, 10.0))}
    env, backend = _make_env(cfg, torch_stream_id=17)
    env.init_state()
    env.set_deferred_validation_boundaries(True)
    runtime = FakeEnvUpdateGraphRuntime()
    env.__dict__["_env_update_graph_runtime"] = runtime
    if mode in {"graph", "cuda_deferred"}:
        if not torch.cuda.is_available():
            pytest.skip("CUDA fake parity requires CUDA owner carriers")
        _move_carriers(env, torch.device("cuda", index=torch.cuda.current_device()))
    if mode == "graph":
        env.__dict__["_env_update_graph_runtime"] = runtime
        env.configure_env_update_graph("graph")
        manifest = env.get_env_update_graph_manifest()
        assert manifest["enabled"] is True
        assert manifest["capture_id"] == runtime.capture_id
        validate_env_update_graph_manifest(manifest)
    elif mode == "deferred":
        env.configure_env_update_graph("eager")
    elif mode == "cuda_deferred":
        if not torch.cuda.is_available():
            pytest.skip("CUDA deferred reference requires CUDA owner carriers")
        _move_carriers(env, torch.device("cuda", index=torch.cuda.current_device()))
        env.configure_env_update_graph("eager")
    else:
        env.set_deferred_validation_boundaries(False)
        env.configure_env_update_graph("disabled")
    return env, backend, runtime


@dataclass
class StepEvidence:
    obs: dict[str, torch.Tensor]
    reward: torch.Tensor
    reward_components: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    done: torch.Tensor
    steps: torch.Tensor
    action: torch.Tensor
    command: torch.Tensor
    command_time_left: torch.Tensor
    command_counter: torch.Tensor
    command_updates: int
    command_resamples: list[list[bool]]
    selected_rows: list[list[int]]
    reset_mask: torch.Tensor
    owner_write_mask: torch.Tensor
    rng_state: torch.Tensor
    rng_versions: tuple[int, int]
    update_receipt: tuple[str, str | None, int]
    backend_receipt: tuple[str, str | None, int]
    verdicts: dict[str, torch.Tensor]


def _capture_evidence(env: Any, backend: Any, mode: str = "eager") -> StepEvidence:
    assert env._state is not None
    term = env.command_manager.get_term("target")
    completion = env._state.info["graph_completion"]
    assert completion is not None
    return StepEvidence(
        obs={name: value.detach().cpu().clone() for name, value in env._state.obs.items()},
        reward=env._state.reward.detach().cpu().clone(),
        reward_components=env.reward_manager.step_rewards.detach().clone().cpu(),
        terminated=env._state.terminated.detach().clone().cpu(),
        truncated=env._state.truncated.detach().clone().cpu(),
        done=(env._state.terminated | env._state.truncated).detach().clone().cpu(),
        steps=env._state.info["steps"].detach().clone().cpu(),
        action=env.action_manager.action.detach().clone().cpu(),
        command=term.command.detach().clone().cpu(),
        command_time_left=term.time_left.detach().clone().cpu(),
        command_counter=term.command_counter.detach().clone().cpu(),
        command_updates=len(env.command_update_ids),
        command_resamples=[mask.detach().cpu().tolist() for mask in term.reset_masks],
        selected_rows=[rows.detach().cpu().tolist() for rows, _, _ in backend.set_state_calls],
        reset_mask=env.reset_buf.detach().clone().cpu(),
        owner_write_mask=env._reset_state._owner_tensor_write_mask.detach().clone().cpu(),
        rng_state=env.torch_rng.get_state().detach().clone().cpu(),
        rng_versions=(env.torch_rng.state_version, env.torch_rng.draw_version),
        update_receipt=(
            (completion.operation, completion.capture_id, completion.sequence)
            if mode == "graph"
            else ("", None, 0)
        ),
        backend_receipt=(
            (
                env._env_update_graph_backend_receipt.operation,
                env._env_update_graph_backend_receipt.capture_id,
                env._env_update_graph_backend_receipt.sequence,
            )
            if env._env_update_graph_backend_receipt is not None
            else (
                "step",
                backend.graph_completions[-1].capture_id,
                backend.graph_completions[-1].sequence,
            )
        ),
        verdicts={
            name: value.detach().cpu().clone()
            for name, value in env.env_update_validation_verdicts.items()
        },
    )


def _assert_step_parity(expected: StepEvidence, actual: StepEvidence) -> None:
    torch.testing.assert_close(expected.obs, actual.obs, check_device=False, check_dtype=False)
    torch.testing.assert_close(expected.reward, actual.reward)
    torch.testing.assert_close(expected.reward_components, actual.reward_components)
    torch.testing.assert_close(expected.terminated, actual.terminated)
    torch.testing.assert_close(expected.truncated, actual.truncated)
    torch.testing.assert_close(expected.done, actual.done)
    torch.testing.assert_close(expected.steps, actual.steps)
    torch.testing.assert_close(expected.action, actual.action)
    torch.testing.assert_close(expected.command, actual.command)
    torch.testing.assert_close(expected.command_time_left, actual.command_time_left)
    torch.testing.assert_close(expected.command_counter, actual.command_counter)
    assert expected.command_updates == actual.command_updates
    assert expected.command_resamples == actual.command_resamples
    assert expected.selected_rows == actual.selected_rows
    torch.testing.assert_close(expected.reset_mask, actual.reset_mask)
    torch.testing.assert_close(expected.owner_write_mask, actual.owner_write_mask)
    torch.testing.assert_close(expected.rng_state, actual.rng_state)
    assert expected.rng_versions == actual.rng_versions
    if expected.update_receipt[0]:
        assert actual.update_receipt[0] == "env_update"
        assert actual.update_receipt[1:] == expected.update_receipt[1:]
    assert expected.backend_receipt == actual.backend_receipt
    torch.testing.assert_close(expected.verdicts, actual.verdicts)


def _run_case(mode: str, actions: list[list[float]]) -> tuple[list[StepEvidence], Any, Any]:
    env, backend, runtime = _make_parity_env(mode)
    evidence: list[StepEvidence] = []
    for row in actions:
        action = torch.tensor([row], dtype=torch.float32, device=env.device).T
        env.step(action)
        evidence.append(_capture_evidence(env, backend, mode))
    assert env._env_update_graph_replay_count == (len(actions) + 1 if mode == "graph" else 0)
    assert runtime.replays == (len(actions) if mode == "graph" else 0)
    return evidence, env, backend


def _assert_case(actions: list[list[float]]) -> None:
    references: dict[str, list[StepEvidence]] = {}
    envs: dict[str, Any] = {}
    for mode in MODES:
        references[mode], envs[mode], _backend = _run_case(mode, actions)
        envs[mode].close()

    for mode in ("deferred", "graph"):
        assert len(references[mode]) == len(references["eager"])
        for index, (expected, actual) in enumerate(
            zip(references["eager"], references[mode], strict=True)
        ):
            # The graph fixture moves the deterministic test rig to CUDA; Torch
            # CPU and CUDA generators intentionally have different state layouts.
            # Compare RNG-sensitive internals with the CUDA deferred reference,
            # then require both eager and deferred semantics to remain aligned.
            reference = expected if mode != "graph" else references["cuda_deferred"][index]
            _assert_step_parity(reference, actual)
        if mode == "graph":
            for expected, actual in zip(
                references["deferred"],
                references["graph"],
                strict=True,
            ):
                # Graph publication owns a distinct update receipt, while its
                # dependent backend sequence remains aligned.
                assert actual.update_receipt[0] == "env_update"
                assert actual.update_receipt[1] == "gate-u-env-update-capture"
                assert actual.update_receipt[2] > 0
                assert actual.backend_receipt == expected.backend_receipt


@pytest.mark.parametrize(
    ("case", "actions"),
    [
        ("no_done", [[0.0, 0.0], [0.1, 0.1]]),
        ("empty_done", [[0.0, 0.0], [0.8, 0.8], [0.0, 0.0]]),
        ("partial_done", [[0.8, 0.0], [0.0, 0.8], [0.0, 0.0]]),
        ("full_done", [[0.8, 0.8], [0.0, 0.0]]),
    ],
)
def test_env_update_graph_semantic_parity_matrix(case: str, actions: list[list[float]]) -> None:
    assert case
    _assert_case(actions)
