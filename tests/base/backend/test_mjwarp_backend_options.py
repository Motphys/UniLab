"""MJWarp hot-path finite-check routing (#2587, #2618).

TorchEnv validates the final control for NaN/Inf on every step, eagerly or
through deferred finite validation, so MJWarp's own blocking per-call read is
a duplicate. UniLab therefore always passes ``mjwarp_validate_tensor_finite``
explicitly instead of inheriting UniSim's blocking default; UniSim's factory
maps it to the MJWarp adapter's ``validate_tensor_finite``.

Both MJWarp options are derived from the env's single validation strictness
(``finite_validation_mode``, issue #2618): strict mode keeps the backend's
per-call operand check immediate, deferred mode disables it and latches the
backend's fail-closed checks into device verdicts
(``mjwarp_defer_tensor_validation``) that the env publishes at its poll
boundary. The removed ``mjwarp_*`` config keys fail closed with a clear error.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from unilab.base.backend_factory import create_backend, env_backend_kwargs
from unilab.base.base import EnvCfg
from unilab.base.registry import apply_cfg_overrides
from unilab.base.scene import SceneCfg
from unilab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg

_MODEL_FILE = str(
    Path(__file__).resolve().parents[2] / "fixtures" / "mjlab_cartpole" / "cartpole.xml"
)


@pytest.mark.parametrize(
    "cfg",
    [
        EnvCfg(),
        ManagerBasedRlEnvCfg(),
        ManagerBasedRlEnvCfg(autoreset_cuda_graph=True),
    ],
    ids=["env_cfg", "default", "autoreset_graph"],
)
def test_mjwarp_finite_check_is_derived_from_validation_mode(cfg: EnvCfg) -> None:
    kwargs = env_backend_kwargs(cfg)
    assert kwargs["mjwarp_validate_tensor_finite"] is False
    assert kwargs["mjwarp_defer_tensor_validation"] is True


def test_mjwarp_strict_mode_keeps_the_immediate_backend_check() -> None:
    cfg = ManagerBasedRlEnvCfg(finite_validation_mode="strict")
    kwargs = env_backend_kwargs(cfg)
    assert kwargs["mjwarp_validate_tensor_finite"] is True
    assert "mjwarp_defer_tensor_validation" not in kwargs


def test_mjwarp_derived_options_reject_unknown_mode() -> None:
    cfg = ManagerBasedRlEnvCfg()
    cfg.finite_validation_mode = "sometimes"
    with pytest.raises(ValueError, match="finite_validation_mode must be 'strict' or 'deferred'"):
        env_backend_kwargs(cfg)


@pytest.mark.parametrize(
    "removed", ["mjwarp_validate_tensor_finite", "mjwarp_defer_tensor_validation"]
)
def test_mjwarp_removed_keys_fail_closed_as_env_overrides(removed: str) -> None:
    cfg = ManagerBasedRlEnvCfg()
    with pytest.raises(ValueError, match=f"has no attribute '{removed}'"):
        apply_cfg_overrides(cfg, {removed: True})


@pytest.mark.parametrize(
    "removed", ["mjwarp_validate_tensor_finite", "mjwarp_defer_tensor_validation"]
)
def test_mjwarp_removed_keys_fail_closed_as_injected_attributes(removed: str) -> None:
    cfg = ManagerBasedRlEnvCfg()
    setattr(cfg, removed, True)
    with pytest.raises(ValueError, match=f"{removed} was removed from the public config surface"):
        env_backend_kwargs(cfg)


def test_non_mjwarp_backend_accepts_the_routed_option() -> None:
    pytest.importorskip("mujoco")
    backend = create_backend(
        "mujoco",
        SceneCfg(model_file=_MODEL_FILE),
        2,
        0.01,
        base_name="cart",
        **env_backend_kwargs(ManagerBasedRlEnvCfg()),
    )
    backend.close()


@pytest.mark.parametrize(("mode", "raises"), [("deferred", False), ("strict", True)])
def test_mjwarp_step_tensor_finite_check_follows_validation_mode(mode: str, raises: bool) -> None:
    pytest.importorskip("mujoco_warp")
    torch = pytest.importorskip("torch")
    warp = pytest.importorskip("warp")
    warp.init()
    if not bool(warp.get_device().is_cuda):
        pytest.skip("mjwarp runtime tests require an active CUDA Warp device")

    # The backend check is derived from the env-level validation mode:
    # deferred disables the duplicate per-call read, strict keeps it.
    cfg = ManagerBasedRlEnvCfg(finite_validation_mode=mode)
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
