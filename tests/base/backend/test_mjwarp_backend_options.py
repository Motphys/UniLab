"""MJWarp hot-path finite-check routing (#2587).

UniLab validates the final control on the device; under deferred finite
validation the verdict is published at a fixed poll boundary, so MJWarp's own
blocking per-call NaN/Inf read is redundant. ``env_backend_kwargs`` resolves
``mjwarp_validate_tensor_finite`` for UniSim's factory, which maps it to the
MJWarp adapter's ``validate_tensor_finite`` option.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from unilab.base.backend_factory import create_backend, env_backend_kwargs
from unilab.base.base import EnvCfg
from unilab.base.scene import SceneCfg
from unilab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg

_MODEL_FILE = str(
    Path(__file__).resolve().parents[2] / "fixtures" / "mjlab_cartpole" / "cartpole.xml"
)


def test_plain_env_cfg_keeps_the_mjwarp_default() -> None:
    assert env_backend_kwargs(EnvCfg())["mjwarp_validate_tensor_finite"] is None


@pytest.mark.parametrize(("defer", "expected"), [(True, False), (False, None)])
def test_deferred_finite_validation_disables_mjwarp_blocking_check(
    defer: bool, expected: bool | None
) -> None:
    cfg = ManagerBasedRlEnvCfg(defer_finite_validation=defer)
    assert env_backend_kwargs(cfg)["mjwarp_validate_tensor_finite"] is expected


@pytest.mark.parametrize("defer", [True, False])
@pytest.mark.parametrize("explicit", [True, False])
def test_explicit_mjwarp_validate_tensor_finite_wins(defer: bool, explicit: bool) -> None:
    cfg = ManagerBasedRlEnvCfg(
        defer_finite_validation=defer, mjwarp_validate_tensor_finite=explicit
    )
    assert env_backend_kwargs(cfg)["mjwarp_validate_tensor_finite"] is explicit


@pytest.mark.parametrize("value", [1, "false", 0.0])
def test_mjwarp_validate_tensor_finite_rejects_non_bool(value: object) -> None:
    with pytest.raises(ValueError, match="mjwarp_validate_tensor_finite must be bool"):
        EnvCfg(mjwarp_validate_tensor_finite=value).validate()  # type: ignore[arg-type]


def test_non_mjwarp_backend_accepts_the_routed_option() -> None:
    pytest.importorskip("mujoco")
    cfg = ManagerBasedRlEnvCfg(defer_finite_validation=True)
    backend = create_backend(
        "mujoco",
        SceneCfg(model_file=_MODEL_FILE),
        2,
        0.01,
        base_name="cart",
        **env_backend_kwargs(cfg),
    )
    backend.close()


@pytest.mark.parametrize(("defer", "raises"), [(True, False), (False, True)])
def test_mjwarp_step_tensor_finite_check_follows_env_cfg(defer: bool, raises: bool) -> None:
    pytest.importorskip("mujoco_warp")
    torch = pytest.importorskip("torch")
    warp = pytest.importorskip("warp")
    warp.init()
    if not bool(warp.get_device().is_cuda):
        pytest.skip("mjwarp runtime tests require an active CUDA Warp device")

    cfg = ManagerBasedRlEnvCfg(defer_finite_validation=defer)
    backend = create_backend(
        "mjwarp",
        SceneCfg(model_file=_MODEL_FILE),
        2,
        0.01,
        base_name="cart",
        **env_backend_kwargs(cfg),
    )
    try:
        ctrl = torch.full((2, backend.num_actuators), torch.nan, dtype=torch.float32, device="cuda")
        if raises:
            with pytest.raises(ValueError, match="ctrl must contain finite values"):
                backend.step_tensor(ctrl, 1)
        else:
            backend.step_tensor(ctrl, 1)
    finally:
        backend.close()
