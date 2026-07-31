from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from unilab.envs.locomotion.g1 import joystick
from unilab.envs.locomotion.g1.joystick import (
    G1MjwarpManagedEnv,
    G1MjwarpManagedOnlyError,
    G1WalkFlatCfg,
    G1WalkRewardConfig,
)


class _ColdBackend:
    backend_type = "mjwarp"
    num_actuators = 29

    def __init__(self) -> None:
        self.cleanup_calls = 0
        self.forbidden_calls: list[str] = []

    def get_actuator_ctrl_range(self) -> np.ndarray:
        return np.tile(np.asarray([[-100.0, 100.0]]), (self.num_actuators, 1))

    def cleanup_scene_assets(self) -> None:
        self.cleanup_calls += 1

    def step(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        self.forbidden_calls.append("step")
        raise AssertionError("managed-only env must not call backend.step")

    def materialize(self) -> None:
        self.forbidden_calls.append("materialize")
        raise AssertionError("env construction must leave materialization to the managed runtime")


class _BadRangeBackend(_ColdBackend):
    def get_actuator_ctrl_range(self) -> np.ndarray:
        return np.zeros((self.num_actuators - 1, 2), dtype=float)


def _cfg() -> G1WalkFlatCfg:
    return G1WalkFlatCfg(
        reward_config=G1WalkRewardConfig(
            scales={"tracking_lin_vel": 2.0},
            tracking_sigma=0.25,
            gait_frequency=1.5,
            feet_phase_swing_height=0.09,
            feet_phase_tracking_sigma=0.008,
            base_height_target=0.754,
            min_base_height=0.55,
            max_tilt_deg=25.0,
            pose_weights=[1.0] * 29,
        )
    )


def _make_env(monkeypatch: pytest.MonkeyPatch) -> tuple[G1MjwarpManagedEnv, _ColdBackend]:
    backend = _ColdBackend()
    monkeypatch.setattr(joystick, "create_backend", lambda *args, **kwargs: backend)
    return G1MjwarpManagedEnv(_cfg(), num_envs=4, backend_type="mjwarp"), backend


def test_mjwarp_registry_env_is_a_cold_managed_only_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env, backend = _make_env(monkeypatch)

    assert env.num_envs == 4
    assert env.state is None
    assert env.obs_groups_spec == {"obs": 98, "critic": 101}
    assert env.observation_space.shape == (199,)
    assert env.action_space.shape == (29,)
    assert backend.forbidden_calls == []

    env.close()
    assert backend.cleanup_calls == 1


@pytest.mark.parametrize(
    ("operation", "invoke"),
    [
        ("init_state", lambda env: env.init_state()),
        ("reset", lambda env: env.reset(np.asarray([0], dtype=np.int64))),
        ("step", lambda env: env.step(np.zeros((4, 29), dtype=np.float32))),
        (
            "apply_action",
            lambda env: env.apply_action(
                np.zeros((4, 29), dtype=np.float32),
                object(),  # type: ignore[arg-type]
            ),
        ),
        (
            "update_state",
            lambda env: env.update_state(object()),  # type: ignore[arg-type]
        ),
        (
            "play",
            lambda env: env.resolve_play_render_plan(
                play_render_mode="none", play_steps=1, output_video=None
            ),
        ),
        (
            "play",
            lambda env: env.run_playback(
                initialize=lambda: None,
                step=lambda state: state,
                num_steps=1,
            ),
        ),
    ],
)
def test_retired_handwritten_lifecycle_fails_before_physics(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    invoke: Callable[[G1MjwarpManagedEnv], Any],
) -> None:
    env, backend = _make_env(monkeypatch)

    with pytest.raises(G1MjwarpManagedOnlyError) as exc_info:
        invoke(env)

    diagnostic = str(exc_info.value)
    assert f"direct {operation}" in diagnostic
    assert "retired hand-written NpEnv lifecycle" in diagnostic
    assert "task=g1_walk_flat/mjwarp" in diagnostic
    assert "training.operation=train" in diagnostic
    assert "training.operation=export" in diagnostic
    assert "task=g1_walk_flat/mujoco" in diagnostic
    assert backend.forbidden_calls == []
    env.close()


def test_managed_runtime_factory_is_the_only_runtime_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env, backend = _make_env(monkeypatch)
    captured: dict[str, object] = {}
    runtime = object()

    def fake_create_runtime(**kwargs: object) -> object:
        captured.update(kwargs)
        return runtime

    monkeypatch.setattr(
        "unilab.envs.locomotion.g1.managed_device.create_g1_managed_device_runtime",
        fake_create_runtime,
    )

    result = env.create_device_managed_runtime(
        reset_seed=17,
        record_lifecycle=True,
        enable_stability_diagnostics=True,
    )

    assert result is runtime
    assert captured == {
        "backend": backend,
        "cfg": env.cfg,
        "reset_seed": 17,
        "record_lifecycle": True,
        "enable_stability_diagnostics": True,
    }
    assert backend.forbidden_calls == []
    env.close()


def test_managed_only_env_rejects_backend_identity_before_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory_calls = 0

    def fake_create_backend(*args: object, **kwargs: object) -> object:
        nonlocal factory_calls
        del args, kwargs
        factory_calls += 1
        return _ColdBackend()

    monkeypatch.setattr(joystick, "create_backend", fake_create_backend)

    with pytest.raises(ValueError, match="owned only by backend='mjwarp'"):
        G1MjwarpManagedEnv(_cfg(), num_envs=1, backend_type="mujoco")

    assert factory_calls == 0


def test_managed_only_env_cleans_backend_when_static_space_binding_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _BadRangeBackend()
    monkeypatch.setattr(joystick, "create_backend", lambda *args, **kwargs: backend)

    with pytest.raises(ValueError, match="control range shape mismatch"):
        G1MjwarpManagedEnv(_cfg(), num_envs=1, backend_type="mjwarp")

    assert backend.cleanup_calls == 1
