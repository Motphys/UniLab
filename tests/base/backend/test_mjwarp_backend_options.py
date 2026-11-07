"""MJWarp hot-path finite-check routing (#2587).

TorchEnv validates the final control for NaN/Inf on every step, eagerly or
through deferred finite validation, so MJWarp's own blocking per-call read is
a duplicate. UniLab therefore always passes ``mjwarp_validate_tensor_finite``
explicitly (default ``False``) instead of inheriting UniSim's blocking default;
UniSim's factory maps it to the MJWarp adapter's ``validate_tensor_finite``.
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


@pytest.mark.parametrize(
    "cfg",
    [
        EnvCfg(),
        ManagerBasedRlEnvCfg(defer_finite_validation=True),
        ManagerBasedRlEnvCfg(defer_finite_validation=False),
    ],
    ids=["env_cfg", "deferred", "eager"],
)
def test_mjwarp_finite_check_is_explicitly_disabled_by_default(cfg: EnvCfg) -> None:
    assert env_backend_kwargs(cfg)["mjwarp_validate_tensor_finite"] is False


def test_mjwarp_finite_check_can_be_reenabled_explicitly() -> None:
    cfg = ManagerBasedRlEnvCfg(mjwarp_validate_tensor_finite=True)
    assert env_backend_kwargs(cfg)["mjwarp_validate_tensor_finite"] is True


@pytest.mark.parametrize("value", [None, 1, "false", 0.0])
def test_mjwarp_validate_tensor_finite_rejects_non_bool(value: object) -> None:
    with pytest.raises(ValueError, match="mjwarp_validate_tensor_finite must be bool"):
        EnvCfg(mjwarp_validate_tensor_finite=value).validate()  # type: ignore[arg-type]


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


@pytest.mark.parametrize(("validate", "raises"), [(None, False), (True, True)])
def test_mjwarp_step_tensor_finite_check_follows_env_cfg(
    validate: bool | None, raises: bool
) -> None:
    pytest.importorskip("mujoco_warp")
    torch = pytest.importorskip("torch")
    warp = pytest.importorskip("warp")
    warp.init()
    if not bool(warp.get_device().is_cuda):
        pytest.skip("mjwarp runtime tests require an active CUDA Warp device")

    # ``None`` exercises the UniLab default; both runs use eager env-level
    # validation to show the backend default does not depend on deferral.
    cfg = ManagerBasedRlEnvCfg(defer_finite_validation=False)
    if validate is not None:
        cfg.mjwarp_validate_tensor_finite = validate
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
