"""Drake physics + MuJoCo rendering integration contracts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from unisim.backend.base import BackendRootStateLayout, SimBackend

from unilab.base.backend_factory import _install_drake_root_state_layout
from unilab.base.np_env import NpEnv


def test_drake_root_layout_compatibility_shim_uses_runtime_metadata() -> None:
    info = SimpleNamespace(
        joint_body_names=("base", "hip"),
        joint_qpos_adr=np.asarray([0, 7], dtype=np.int32),
        joint_qvel_adr=np.asarray([0, 6], dtype=np.int32),
        joint_qpos_dim=np.asarray([7, 1], dtype=np.int32),
        joint_qvel_dim=np.asarray([6, 1], dtype=np.int32),
    )
    backend = SimpleNamespace(
        backend_type="drake",
        _runtime=SimpleNamespace(model_info=lambda: info),
    )
    _install_drake_root_state_layout(backend)  # type: ignore[arg-type]

    layout = backend.get_root_state_layout("base")
    assert isinstance(layout, BackendRootStateLayout)
    assert layout.qpos_indices == tuple(range(7))
    assert layout.qvel_indices == tuple(range(6))


def test_drake_root_layout_compatibility_shim_rejects_non_floating_body() -> None:
    info = SimpleNamespace(
        joint_body_names=("base",),
        joint_qpos_adr=np.asarray([0], dtype=np.int32),
        joint_qvel_adr=np.asarray([0], dtype=np.int32),
        joint_qpos_dim=np.asarray([1], dtype=np.int32),
        joint_qvel_dim=np.asarray([1], dtype=np.int32),
    )
    backend = SimpleNamespace(
        backend_type="drake",
        _runtime=SimpleNamespace(model_info=lambda: info),
    )
    _install_drake_root_state_layout(backend)  # type: ignore[arg-type]

    try:
        backend.get_root_state_layout("base")
    except NotImplementedError as exc:
        assert "floating free joint" in str(exc)
    else:  # pragma: no cover - assertion style keeps the diagnostic concise.
        raise AssertionError("fixed Drake body unexpectedly exposed a root-state layout")


def test_drake_env_auto_playback_is_mujoco_record_plan() -> None:
    class _Env(NpEnv):
        @property
        def action_space(self):
            raise NotImplementedError

        def apply_action(self, actions, state):
            raise NotImplementedError

        def update_state(self, state):
            raise NotImplementedError

    env = object.__new__(_Env)
    env._backend = SimpleNamespace(  # type: ignore[attr-defined]
        backend_type="drake",
        resolve_play_render_plan=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    plan = env.resolve_play_render_plan(
        play_render_mode="auto",
        play_steps=3,
        output_video=Path("play.mp4"),
    )
    assert plan.play_render_mode == "record"


def test_sim_backend_root_layout_default_remains_fail_closed() -> None:
    try:
        SimBackend.get_root_state_layout(object(), "base")  # type: ignore[arg-type]
    except NotImplementedError as exc:
        assert "root-state layout" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("base SimBackend unexpectedly exposed root-state metadata")
