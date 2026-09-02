"""Row-scoped partial-reset parity tests for MotionCommand (issue #1261).

On the partial-reset path the command manager recomputes only the reset rows;
untouched rows reuse the values produced by the per-step compute of the same
control step. These tests pin that contract:

- untouched rows keep their command/motion/relative-state/metric values
  bit-identically across a partial reset (except the sampler-stat metrics,
  which track global sampler scalars);
- reset rows match a full recompute of the same post-reset state, which is
  what the pre-#1261 full-batch implementation produced.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from unilab.base import registry
from unilab.base.config_adapter import BackendAdapter
from unilab.base.config_materialization import apply_cfg_overrides
from unilab.envs import ManagerBasedRlEnvCfg
from unilab.tasks.motion_tracking.common.manager_terms import MotionCommand
from unilab.tasks.motion_tracking.g1.manager_terms import BoxMotionCommand

_ROOT = Path(__file__).parents[2]

_CASES = (
    ("ppo", "g1_flip_tracking", "G1FlipTracking"),
    ("ppo", "g1_box_tracking", "G1BoxTracking"),
)

_SAMPLER_STAT_METRICS = ("sampling_entropy", "sampling_top1_prob", "sampling_top1_bin")

_ACCESSORS: dict[str, Callable[[MotionCommand], np.ndarray]] = {
    "command": lambda term: term.command,
    "time_steps": lambda term: term.time_steps,
    "body_pos_w": lambda term: term.body_pos_w,
    "body_pos_relative_w": lambda term: term.body_pos_relative_w,
    "body_quat_relative_w": lambda term: term.body_quat_relative_w,
    "motion_anchor_pos_b": lambda term: term.motion_anchor_pos_b,
    "motion_anchor_ori_b": lambda term: term.motion_anchor_ori_b,
    "robot_body_pos_b": lambda term: term.robot_body_pos_b,
    "robot_body_ori_b": lambda term: term.robot_body_ori_b,
    "joint_default_bias": lambda term: term.joint_default_bias,
    "robot_body_pos_w": lambda term: term._robot_body_pos_w,
    "robot_body_quat_w": lambda term: term._robot_body_quat_w,
    "robot_body_lin_vel_w": lambda term: term._robot_body_lin_vel_w,
    "robot_body_ang_vel_w": lambda term: term._robot_body_ang_vel_w,
}

_MOTION_DATA_FIELDS = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
    "object_pos_w",
    "object_quat_w",
    "object_lin_vel_w",
    "object_ang_vel_w",
)


def _make_env(config_root: str, task: str, backend: str, identity: str, num_envs: int):
    registry.ensure_registries()
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(_ROOT / "src" / "unilab" / "conf" / config_root), version_base="1.3"
    ):
        owner = compose("config", overrides=[f"task={task}/{backend}"])
    cfg = registry.materialize_env_config(identity)
    assert isinstance(cfg, ManagerBasedRlEnvCfg)
    override = BackendAdapter(
        owner, root_dir=_ROOT, algo_name=config_root
    ).build_task_env_cfg_override()
    apply_cfg_overrides(cfg, override)
    return registry.make(
        identity, num_envs=num_envs, sim_backend=backend, env_cfg_override=override
    )


def _buffers(term: MotionCommand) -> dict[str, np.ndarray]:
    buffers = {name: accessor(term) for name, accessor in _ACCESSORS.items()}
    for field_name in _MOTION_DATA_FIELDS:
        value = getattr(term._motion_data, field_name, None)
        if value is not None:
            buffers[f"motion_data.{field_name}"] = value
    if isinstance(term, BoxMotionCommand):
        buffers["object_pos_w"] = term.object_pos_w
        buffers["object_state_b"] = term.object_state_b
    return buffers


def _current(term: MotionCommand, name: str) -> np.ndarray:
    if name in _ACCESSORS:
        return _ACCESSORS[name](term)
    if name.startswith("motion_data."):
        return getattr(term._motion_data, name.split(".", 1)[1])
    if name == "object_pos_w":
        return term.object_pos_w  # type: ignore[attr-defined]
    if name == "object_state_b":
        return term.object_state_b  # type: ignore[attr-defined]
    raise KeyError(name)


@pytest.mark.parametrize(("config_root", "task", "identity"), _CASES)
def test_motion_command_partial_reset_row_parity(
    config_root: str, task: str, identity: str
) -> None:
    pytest.importorskip("mujoco")
    try:
        from mujoco_uni.batch_env import BatchEnvPool as _  # noqa: F401
    except Exception:
        pytest.skip("mujoco_uni.batch_env not available")

    num_envs = 4
    env = _make_env(config_root, task, "mujoco", identity, num_envs)
    try:
        env.init_state()
        term = env.command_manager.get_term("motion")
        assert isinstance(term, MotionCommand)
        action_dim = term.motion.num_joints
        rng = np.random.default_rng(7)
        for _ in range(5):
            env.step((0.1 * rng.standard_normal((num_envs, action_dim))).astype(np.float32))

        reset_ids = np.array([0, 2], dtype=np.int32)
        keep_ids = np.array([1, 3], dtype=np.int32)
        before = {name: value.copy() for name, value in _buffers(term).items()}
        metrics_before = {name: value.copy() for name, value in term.metrics.items()}
        reset_obs, _ = env.reset(env_ids=reset_ids)
        after = _buffers(term)

        # Untouched rows keep their per-step values bit-identically, except the
        # sampler-stat metrics which track global sampler scalars.
        for name, value in before.items():
            np.testing.assert_array_equal(
                after[name][keep_ids],
                value[keep_ids],
                err_msg=f"untouched rows changed for {name}",
            )
        for name, value in metrics_before.items():
            if name in _SAMPLER_STAT_METRICS:
                expected = np.full(
                    num_envs, getattr(term.sampler, name), dtype=term.metrics[name].dtype
                )
                np.testing.assert_array_equal(term.metrics[name], expected)
            else:
                np.testing.assert_array_equal(
                    term.metrics[name][keep_ids],
                    value[keep_ids],
                    err_msg=f"untouched rows changed for metric {name}",
                )

        # Reset observations are finite and scattered into the env state.
        assert env.state is not None
        for group, values in reset_obs.items():
            assert values.shape[0] == len(reset_ids)
            assert np.isfinite(values).all()
            np.testing.assert_array_equal(env.state.obs[group][reset_ids], values)

        # Reference: a full recompute of the post-reset state (the pre-#1261
        # behavior) must agree with the row-scoped results on every row.
        reference = {name: value.copy() for name, value in _buffers(term).items()}
        term._refresh_motion()
        term._refresh_robot_state(force=True)
        term._refresh_relative_state()
        if isinstance(term, BoxMotionCommand):
            term._refresh_object_state()
        for name, value in reference.items():
            # The default bias is intentionally resampled from RNG on reset rows.
            if name == "joint_default_bias":
                np.testing.assert_array_equal(_current(term, name), value)
                continue
            np.testing.assert_allclose(
                _current(term, name),
                value,
                rtol=1e-6,
                atol=1e-7,
                err_msg=f"row-scoped refresh disagrees with full refresh for {name}",
            )
    finally:
        env.close()
