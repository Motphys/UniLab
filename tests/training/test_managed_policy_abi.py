"""Managed compiled-plan ABI snapshots must guard sim2sim before env creation."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from omegaconf import DictConfig, OmegaConf

from unilab.base.backend import (
    BufferContract,
    BufferLayout,
    BufferLifetime,
    BufferMutability,
    BufferOwner,
    BufferPlacement,
    ControlSpec,
    ExecutionProfile,
    PhysicalUnit,
    ReferenceFrame,
    StateFieldKind,
)
from unilab.manager import (
    CompiledTaskPlan,
    EntityKind,
    EntitySelector,
    NormalizationMode,
    PolicySpec,
    QuaternionOrder,
    StateRequirement,
    TaskCompiler,
    TaskSpec,
    TensorSpec,
    TermDefinition,
    TermInvocation,
    TermPhase,
    TermRegistry,
    TermRole,
    managed_policy_abi_snapshot,
)
from unilab.training.experiment import ExperimentTracker
from unilab.training.sim2sim import (
    MANAGED_POLICY_ABI_SNAPSHOT_KEY,
    CrossBackendIncompatibleError,
    Sim2SimConfigResolver,
    extract_contract_snapshot,
    policy_load_dim_guard,
    resolve_sim2sim_config,
)


class _Resolver:
    """Independent cold-path selector fixture; it has no backend or env owner."""

    def __init__(self, body_id: int) -> None:
        self._body_id = body_id

    def resolve(self, selector: EntitySelector) -> tuple[int, ...]:
        assert selector.key == "fixture.base"
        return (self._body_id,)


def _control() -> ControlSpec:
    return ControlSpec(
        semantic_key="fixture.action",
        buffer=BufferContract(
            row_shape=(2,),
            dtype="float32",
            layout=BufferLayout.C_CONTIGUOUS,
            placement=BufferPlacement.host(),
            owner=BufferOwner.MANAGER,
            mutability=BufferMutability.READ_ONLY,
            lifetime=BufferLifetime.UNTIL_STEP_COMPLETE,
            dlpack_exportable=False,
        ),
    )


def _compiled_plan(
    *,
    body_id: int = 3,
    velocity_frame: ReferenceFrame = ReferenceFrame.BODY,
    velocity_dtype: str = "float32",
    action_scale: tuple[float, ...] = (0.25,),
    normalization: NormalizationMode = NormalizationMode.EMPIRICAL,
) -> CompiledTaskPlan:
    base = EntitySelector(
        key="fixture.base",
        entity="fixture",
        kind=EntityKind.BODY,
        expressions=("base",),
    )
    registry = TermRegistry()
    registry.register(
        TermDefinition(
            key="obs.base_velocity",
            version="1",
            phase=TermPhase.OBSERVATION,
            role=TermRole.OBSERVATION,
            state_requirements=(
                StateRequirement(
                    semantic_key="fixture.base.velocity",
                    selector=base,
                    field_kind=StateFieldKind.LINEAR_VELOCITY,
                    tensor=TensorSpec(
                        (1, 3),
                        velocity_dtype,
                        frame=velocity_frame,
                        unit=PhysicalUnit.METER_PER_SECOND,
                    ),
                ),
            ),
            output=TensorSpec(
                (3,),
                velocity_dtype,
                frame=velocity_frame,
                unit=PhysicalUnit.METER_PER_SECOND,
            ),
        )
    )
    registry.register(
        TermDefinition(
            key="obs.base_orientation",
            version="1",
            phase=TermPhase.OBSERVATION,
            role=TermRole.OBSERVATION,
            state_requirements=(
                StateRequirement(
                    semantic_key="fixture.base.orientation",
                    selector=base,
                    field_kind=StateFieldKind.ORIENTATION,
                    tensor=TensorSpec(
                        (1, 4),
                        "float32",
                        frame=ReferenceFrame.WORLD,
                        unit=PhysicalUnit.QUATERNION,
                        quaternion_order=QuaternionOrder.WXYZ,
                    ),
                ),
            ),
            output=TensorSpec(
                (4,),
                "float32",
                frame=ReferenceFrame.WORLD,
                unit=PhysicalUnit.QUATERNION,
                quaternion_order=QuaternionOrder.WXYZ,
            ),
        )
    )
    task = TaskSpec.create(
        key="managed_policy_abi_fixture",
        terms=(
            TermInvocation.create(
                key="velocity",
                definition_key="obs.base_velocity",
                observation_group="actor",
            ),
            TermInvocation.create(
                key="orientation",
                definition_key="obs.base_orientation",
                observation_group="critic",
            ),
        ),
        control=_control(),
        execution_profile=ExecutionProfile.HOST_NUMPY,
        executor_key="reference.numpy.fixture.v1",
        policy=PolicySpec(
            observation_groups=("actor", "critic"),
            action_scale=action_scale,
            normalization=normalization,
        ),
    )
    return TaskCompiler(registry).compile(
        task,
        resolver=_Resolver(body_id),
        capabilities=frozenset({"state.body.linear_velocity", "state.body.orientation"}),
    )


def _write_snapshot(run_dir: Path, snapshot: dict[str, object]) -> None:
    (run_dir / "run_config.json").write_text(
        json.dumps({"contract_snapshot": snapshot}), encoding="utf-8"
    )


def _target_cfg() -> DictConfig:
    return OmegaConf.create(
        {
            "training": {"sim_backend": "mjwarp"},
            "algo": {"obs_groups": {"actor": ["actor"]}},
            "env": {"control_config": {"action_scale": 0.25}},
        }
    )


def test_managed_policy_abi_snapshot_is_semantic_and_binding_independent() -> None:
    first = _compiled_plan(body_id=3)
    second = _compiled_plan(body_id=19)

    first_snapshot = managed_policy_abi_snapshot(first)
    second_snapshot = managed_policy_abi_snapshot(second)

    assert first.fingerprint == second.fingerprint
    assert first.selector_binding_fingerprint != second.selector_binding_fingerprint
    assert first_snapshot == second_snapshot
    assert first_snapshot["plan_fingerprint"] == first.fingerprint
    assert first_snapshot["policy_abi_fingerprint"] == first.policy_abi.fingerprint
    assert first_snapshot["observation_groups"][0]["outputs"][0]["tensor"] == {
        "shape": [3],
        "dtype": "float32",
        "frame": "body",
        "unit": "m/s",
        "quaternion_order": "none",
    }
    serialized = json.dumps(first_snapshot, sort_keys=True)
    assert "selector_binding" not in serialized
    assert "entity_ids" not in serialized


def _different_plan_fingerprint_snapshot() -> dict[str, Any]:
    snapshot = managed_policy_abi_snapshot(_compiled_plan())
    snapshot["plan_fingerprint"] = "manager-task-contract-v1:different-plan"
    return snapshot


@pytest.mark.parametrize(
    ("name", "target_snapshot"),
    [
        # plan_fingerprint is an execution-identity field stripped before comparison
        # (fix #856): a different plan_fingerprint alone must NOT trigger a denial.
        # That case is covered by test_abi_comparison_ignores_executor_key_and_execution_profile.
        (
            "observation_frame",
            lambda: managed_policy_abi_snapshot(
                _compiled_plan(velocity_frame=ReferenceFrame.WORLD)
            ),
        ),
        (
            "observation_dtype",
            lambda: managed_policy_abi_snapshot(_compiled_plan(velocity_dtype="float64")),
        ),
        (
            "action_scale",
            lambda: managed_policy_abi_snapshot(_compiled_plan(action_scale=(0.5,))),
        ),
        (
            "normalization",
            lambda: managed_policy_abi_snapshot(
                _compiled_plan(normalization=NormalizationMode.NONE)
            ),
        ),
    ],
)
def test_managed_policy_abi_mismatch_fails_closed(
    tmp_path: Path,
    name: str,
    target_snapshot: Callable[[], dict[str, Any]],
) -> None:
    source_abi = managed_policy_abi_snapshot(_compiled_plan())
    target_abi = target_snapshot()
    source_snapshot = extract_contract_snapshot(_target_cfg(), managed_policy_abi=source_abi)
    _write_snapshot(tmp_path, source_snapshot)

    env_constructed = False

    def construct_env() -> None:
        nonlocal env_constructed
        env_constructed = True

    with pytest.raises(CrossBackendIncompatibleError, match="manager.policy_abi"):
        resolve_sim2sim_config(tmp_path, _target_cfg(), managed_policy_abi=target_abi)
        construct_env()

    assert not env_constructed, name


def test_malformed_managed_policy_abi_is_never_downgraded(tmp_path: Path) -> None:
    source_abi = managed_policy_abi_snapshot(_compiled_plan())
    target_abi = copy.deepcopy(source_abi)
    del target_abi["action"]
    _write_snapshot(
        tmp_path,
        extract_contract_snapshot(_target_cfg(), managed_policy_abi=source_abi),
    )

    with pytest.raises(CrossBackendIncompatibleError, match="Invalid target managed policy ABI"):
        resolve_sim2sim_config(
            tmp_path,
            _target_cfg(),
            strict=False,
            managed_policy_abi=target_abi,
        )


def test_managed_policy_abi_presence_is_bidirectional_and_non_strict_is_explicit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    abi = managed_policy_abi_snapshot(_compiled_plan())
    _write_snapshot(tmp_path, extract_contract_snapshot(_target_cfg(), managed_policy_abi=abi))

    with pytest.raises(CrossBackendIncompatibleError, match="target=<absent>"):
        resolve_sim2sim_config(tmp_path, _target_cfg())

    assert resolve_sim2sim_config(tmp_path, _target_cfg(), strict=False) is not None
    assert "WARNING (non-strict)" in capsys.readouterr().out

    old_run = tmp_path / "old"
    old_run.mkdir()
    _write_snapshot(old_run, extract_contract_snapshot(_target_cfg()))
    with pytest.raises(CrossBackendIncompatibleError, match="source=<absent>"):
        resolve_sim2sim_config(old_run, _target_cfg(), managed_policy_abi=abi)

    no_snapshot = tmp_path / "no_snapshot"
    no_snapshot.mkdir()
    (no_snapshot / "run_config.json").write_text("{}", encoding="utf-8")
    assert resolve_sim2sim_config(no_snapshot, _target_cfg(), managed_policy_abi=abi) is not None


def test_managed_policy_abi_matching_snapshot_persists_in_experiment_tracker(
    tmp_path: Path,
) -> None:
    abi = managed_policy_abi_snapshot(_compiled_plan())
    supplied_abi = copy.deepcopy(abi)
    tracker = ExperimentTracker(
        root_dir=tmp_path,
        log_dir=tmp_path / "run",
        algo_name="ppo",
        task_name="managed_policy_abi_fixture",
        sim_backend="mujoco",
        training_cfg={"logger": "tensorboard"},
        full_cfg=_target_cfg(),
        managed_policy_abi=supplied_abi,
    )
    supplied_abi["action"]["scale"][0] = 0.5
    tracker.start()
    persisted = json.loads((tmp_path / "run" / "run_config.json").read_text(encoding="utf-8"))[
        "contract_snapshot"
    ]
    assert persisted[MANAGED_POLICY_ABI_SNAPSHOT_KEY] == abi
    assert (
        resolve_sim2sim_config(tmp_path / "run", _target_cfg(), managed_policy_abi=abi) is not None
    )


def test_experiment_tracker_accepts_cold_compiled_abi_before_start(tmp_path: Path) -> None:
    abi = managed_policy_abi_snapshot(_compiled_plan())
    tracker = ExperimentTracker(
        root_dir=tmp_path,
        log_dir=tmp_path / "run",
        algo_name="ppo",
        task_name="managed_policy_abi_fixture",
        sim_backend="mjwarp",
        training_cfg={"logger": "tensorboard"},
        full_cfg=_target_cfg(),
    )
    tracker.set_managed_policy_abi(abi)
    tracker.start()
    persisted = json.loads((tmp_path / "run" / "run_config.json").read_text(encoding="utf-8"))[
        "contract_snapshot"
    ]
    assert persisted[MANAGED_POLICY_ABI_SNAPSHOT_KEY] == abi
    with pytest.raises(RuntimeError, match="before ExperimentTracker.start"):
        tracker.set_managed_policy_abi(abi)


def test_sim2sim_resolver_facade_forwards_managed_policy_abi(tmp_path: Path) -> None:
    abi = managed_policy_abi_snapshot(_compiled_plan())
    snapshot = Sim2SimConfigResolver.extract_snapshot(_target_cfg(), managed_policy_abi=abi)
    assert snapshot[MANAGED_POLICY_ABI_SNAPSHOT_KEY] == abi
    _write_snapshot(tmp_path, snapshot)
    assert (
        Sim2SimConfigResolver.resolve(tmp_path, _target_cfg(), managed_policy_abi=abi) is not None
    )


def test_policy_load_guard_surfaces_verified_managed_policy_abi() -> None:
    abi = managed_policy_abi_snapshot(_compiled_plan())
    with pytest.raises(CrossBackendIncompatibleError) as exc_info:
        with policy_load_dim_guard(
            env_obs_dim=7,
            env_action_dim=2,
            algo_name="ppo",
            managed_policy_abi_fingerprint=abi["policy_abi_fingerprint"],
        ):
            raise RuntimeError("size mismatch for actor.0.weight")
    assert abi["policy_abi_fingerprint"] in str(exc_info.value)


# ---------------------------------------------------------------------------
# Regression tests for #856 and #857
# ---------------------------------------------------------------------------


def _variant_snapshot(
    *,
    executor_key: str = "reference.numpy.fixture.v1",
    execution_profile: str = "host_numpy",
    plan_fingerprint_suffix: str = "",
) -> dict[str, Any]:
    """Build an ABI snapshot, optionally overriding execution-identity fields."""
    snapshot = managed_policy_abi_snapshot(_compiled_plan())
    snapshot["executor_key"] = executor_key
    snapshot["execution_profile"] = execution_profile
    if plan_fingerprint_suffix:
        snapshot["plan_fingerprint"] = snapshot["plan_fingerprint"] + plan_fingerprint_suffix
    return snapshot


def test_abi_comparison_ignores_executor_key_and_execution_profile(tmp_path: Path) -> None:
    """Fix #856: same policy I/O on different GPU/executor must not be denied."""
    # Source: trained on cuda:0 with device_resident profile
    source_abi = _variant_snapshot(
        executor_key="device.cuda.0.v1",
        execution_profile="device_resident",
    )
    # Target: play on cuda:1 with host_numpy profile (e.g. mujoco reference)
    target_abi = _variant_snapshot(
        executor_key="reference.numpy.v1",
        execution_profile="host_numpy",
    )
    assert source_abi["executor_key"] != target_abi["executor_key"]
    assert source_abi["execution_profile"] != target_abi["execution_profile"]
    # Semantic policy I/O (obs groups, action, normalization) is identical.
    assert source_abi["policy_abi_fingerprint"] == target_abi["policy_abi_fingerprint"]

    _write_snapshot(
        tmp_path, extract_contract_snapshot(_target_cfg(), managed_policy_abi=source_abi)
    )

    # Must NOT raise — execution identity alone must not block cross-backend play.
    cfg = resolve_sim2sim_config(tmp_path, _target_cfg(), managed_policy_abi=target_abi)
    assert cfg is not None


def test_abi_comparison_still_denies_mismatched_plan_fingerprint(tmp_path: Path) -> None:
    """Fix #856 must not allow truly incompatible ABIs through the stripping."""
    source_abi = managed_policy_abi_snapshot(_compiled_plan())
    # Target changes action_scale — that changes policy_abi_fingerprint.
    target_abi = managed_policy_abi_snapshot(_compiled_plan(action_scale=(0.5,)))
    assert source_abi["policy_abi_fingerprint"] != target_abi["policy_abi_fingerprint"]

    _write_snapshot(
        tmp_path, extract_contract_snapshot(_target_cfg(), managed_policy_abi=source_abi)
    )

    with pytest.raises(CrossBackendIncompatibleError, match="manager.policy_abi"):
        resolve_sim2sim_config(tmp_path, _target_cfg(), managed_policy_abi=target_abi)


def test_rsl_rl_vec_env_wrapper_delegates_policy_abi_snapshot() -> None:
    """Fix #857: wrapper must forward the env's ABI snapshot instead of returning None."""
    from unilab.training.rsl_rl import RslRlVecEnvWrapper

    expected_snapshot = managed_policy_abi_snapshot(_compiled_plan())

    class _FakeEnvWithAbi:
        """Minimal env stub that exposes policy_abi_snapshot (ManagedRuntime API)."""

        policy_abi_snapshot = expected_snapshot

        cfg = OmegaConf.create(
            {
                "max_episode_seconds": 10.0,
                "ctrl_dt": 0.02,
                "max_episode_steps": 500,
            }
        )
        num_envs = 1
        observation_space = None
        action_space = type("A", (), {"shape": (1,)})()
        obs_groups_spec: dict[str, Any] = {"actor": 3}

        def reset(self):
            import numpy as np

            return {"obs": np.zeros((1, 3))}, {}

    env = _FakeEnvWithAbi()
    wrapper = RslRlVecEnvWrapper.__new__(RslRlVecEnvWrapper)
    wrapper.env = env

    assert wrapper.managed_policy_abi_snapshot == expected_snapshot


def test_rsl_rl_vec_env_wrapper_returns_none_for_plain_env() -> None:
    """Fix #857: plain envs without any ABI must still yield None (not AttributeError)."""
    from unilab.training.rsl_rl import RslRlVecEnvWrapper

    class _PlainEnv:
        pass

    wrapper = RslRlVecEnvWrapper.__new__(RslRlVecEnvWrapper)
    wrapper.env = _PlainEnv()

    assert wrapper.managed_policy_abi_snapshot is None
