"""Focused contract tests for Torch-only finite-state diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from unilab.training.run import finite_validation_env_overrides, resolve_nan_guard_cfg
from unilab.training.tensor_diagnostics import NanGuard, NanGuardCfg


def _guard(tmp_path: Path, **cfg: Any) -> NanGuard:
    return NanGuard(
        NanGuardCfg(enabled=True, output_dir=str(tmp_path / "dumps"), **cfg),
        num_envs=3,
        supports_state_playback=True,
    )


def test_detection_rejects_non_tensor_inputs() -> None:
    guard = NanGuard(NanGuardCfg(enabled=True), 3, False)

    with pytest.raises(TypeError, match="observations must be a mapping"):
        guard.check(cast(Any, [torch.zeros((3, 2))]), torch.zeros(3))
    with pytest.raises(TypeError, match="obs\\['obs'\\] must be a torch.Tensor"):
        guard.check({"obs": cast(Any, np.zeros((3, 2), dtype=np.float32))}, torch.zeros(3))
    with pytest.raises(TypeError, match="reward must be a torch.Tensor"):
        guard.check({"obs": torch.zeros((3, 2))}, cast(Any, np.zeros(3, dtype=np.float32)))
    with pytest.raises(TypeError, match="ctrl must be a torch.Tensor"):
        guard.check_ctrl(cast(Any, np.zeros((3, 2), dtype=np.float32)))
    with pytest.raises(TypeError, match="physics_state must be a torch.Tensor"):
        guard.capture(cast(Any, np.zeros((3, 2), dtype=np.float32)))
    with pytest.raises(TypeError, match="nan_env_ids must be a torch.Tensor"):
        guard.dump(cast(Any, np.array([0])), "", 0)


def test_detection_unions_bad_rows_across_groups_and_reward() -> None:
    guard = NanGuard(NanGuardCfg(enabled=True), 3, False)
    obs = {
        "obs": torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]),
        "critic": torch.tensor([[0.0], [1.0], [float("nan")]]),
    }
    reward = torch.tensor([0.0, float("inf"), 2.0])

    ids = guard.check(obs, reward)

    assert isinstance(ids, torch.Tensor)
    assert ids.dtype == torch.int64
    assert ids.tolist() == [1, 2]
    assert guard.check_ctrl(torch.zeros((3, 2))) is None
    assert guard.check({"obs": torch.zeros((3, 1))}, torch.zeros(3)) is None


def test_deferred_verdict_accumulates_mixed_rows_until_poll(tmp_path: Path) -> None:
    guard = _guard(tmp_path, defer_validation=True, check_interval=4)
    obs = {"obs": torch.tensor([[1.0, 2.0], [3.0, float("nan")], [4.0, 5.0]])}
    assert guard.check(obs, torch.zeros(3), step=1) is None
    assert guard.check_ctrl(torch.tensor([[0.0], [1.0], [float("inf")]]), step=2) is None
    assert guard.poll(step=2) is None
    assert guard.device_verdict is not None
    assert bool(guard.device_verdict)

    ids = guard.poll(step=4)
    assert isinstance(ids, torch.Tensor)
    assert ids.tolist() == [1, 2]
    assert guard.device_verdict is not None
    assert not bool(guard.device_verdict)
    assert guard.check({"obs": torch.zeros((3, 2))}, torch.zeros(3), step=5) is None


@pytest.mark.parametrize("check_interval", [0, 4])
def test_explicit_non_deferred_check_ignores_configured_poll_interval(
    tmp_path: Path, check_interval: int
) -> None:
    guard = _guard(tmp_path, defer_validation=True, check_interval=check_interval)
    obs = {"obs": torch.tensor([[0.0], [float("nan")], [0.0]])}

    obs_ids = guard.check(obs, torch.zeros(3), step=1, defer=False)
    assert isinstance(obs_ids, torch.Tensor)
    assert obs_ids.tolist() == [1]
    assert guard.device_verdict is not None
    assert not bool(guard.device_verdict)

    ctrl_ids = guard.check_ctrl(torch.tensor([[0.0], [0.0], [float("inf")]]), step=2, defer=False)
    assert isinstance(ctrl_ids, torch.Tensor)
    assert ctrl_ids.tolist() == [2]


def test_capture_interval_skips_snapshots_and_owner_state_reads(tmp_path: Path) -> None:
    guard = _guard(tmp_path, buffer_size=2, capture_interval=2)
    for step in range(5):
        if guard.capture_due(step):
            guard.capture(torch.full((3, 1), float(step)), step=step)
    path = guard.dump(torch.tensor([0]), "", 5)
    assert path is not None
    artifact = np.load(path)
    assert artifact["states"].shape == (2, 1, 1)
    torch.testing.assert_close(
        torch.from_numpy(artifact["states"]), torch.tensor([[[2.0]], [[4.0]]])
    )

    disabled = _guard(tmp_path, capture_interval=0)
    assert not disabled.capture_due(0)
    disabled.capture(torch.zeros((3, 1)))
    assert disabled._buffer == []  # noqa: SLF001 - snapshot contract


def test_finite_detection_writes_no_artifact(tmp_path: Path) -> None:
    guard = _guard(tmp_path)

    guard.capture(torch.arange(9, dtype=torch.float32).reshape(3, 3) / 8.0)
    assert guard.check({"obs": torch.zeros((3, 2))}, torch.zeros(3)) is None

    assert not (tmp_path / "dumps").exists()


def test_dump_is_the_only_host_and_filesystem_boundary(tmp_path: Path) -> None:
    guard = _guard(tmp_path, buffer_size=2, max_envs_to_dump=1)
    model = tmp_path / "model.xml"
    model.write_text("<mujoco/>", encoding="utf-8")
    state_0 = torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    state_1 = torch.tensor([[6.0, 7.0], [8.0, 9.0], [10.0, 11.0]])
    state_2 = torch.tensor([[12.0, 13.0], [14.0, 15.0], [16.0, 17.0]])
    for state in (state_0, state_1, state_2):
        guard.capture(state)

    path = guard.dump(torch.tensor([1, 2]), str(model), 7)

    assert path is not None
    dump_path = Path(path)
    artifact = np.load(dump_path)
    assert artifact["states"].shape == (2, 1, 2)
    torch.testing.assert_close(
        torch.from_numpy(artifact["states"]),
        torch.tensor([[[8.0, 9.0]], [[14.0, 15.0]]]),
    )
    assert artifact["meta_nan_env_ids"].tolist() == [1, 2]
    assert artifact["meta_dumped_env_ids"].tolist() == [1]
    assert artifact["meta_buffer_len"].item() == 2
    assert artifact["meta_detection_step"].item() == 7
    assert artifact["meta_supports_state_playback"].item() is True
    assert artifact["meta_model_file"].item() == str(model)
    assert Path(str(dump_path).replace(".npz", "_model.xml")).is_file()
    assert (dump_path.parent / "nan_dump_latest.npz").is_symlink()
    assert guard.dump(torch.tensor([1]), str(model), 8) is None


def test_configuration_and_shapes_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="positive environment count"):
        NanGuard(NanGuardCfg(enabled=True), 0, False)
    with pytest.raises(ValueError, match="buffer_size must be positive"):
        NanGuard(NanGuardCfg(enabled=True, buffer_size=0), 3, False)
    with pytest.raises(ValueError, match="max_envs_to_dump must be non-negative"):
        NanGuard(NanGuardCfg(enabled=True, max_envs_to_dump=-1), 3, False)
    for interval in (-1, 1.5, True):
        with pytest.raises(ValueError, match="capture_interval must be non-negative"):
            NanGuard(NanGuardCfg(enabled=True, capture_interval=cast(Any, interval)), 3, False)
        with pytest.raises(ValueError, match="check_interval must be non-negative"):
            NanGuard(NanGuardCfg(enabled=True, check_interval=cast(Any, interval)), 3, False)

    guard = _guard(tmp_path)
    with pytest.raises(ValueError, match="leading dimension must be 3"):
        guard.check({"obs": torch.zeros((2, 1))}, torch.zeros(3))
    with pytest.raises(TypeError, match="must be a floating-point tensor"):
        guard.check({"obs": torch.zeros((3, 2), dtype=torch.int32)}, torch.zeros(3))


def test_training_routes_guard_and_shared_finite_boundary() -> None:
    cfg = OmegaConf.create(
        {
            "training": {
                "defer_finite_validation": True,
                "finite_validation_interval": 32,
                "nan_guard": {
                    "enabled": True,
                    "capture_interval": 32,
                    "check_interval": 32,
                    "defer_validation": True,
                },
            }
        }
    )
    assert finite_validation_env_overrides(cfg) == {
        "defer_finite_validation": True,
        "finite_validation_interval": 32,
    }
    guard_cfg = resolve_nan_guard_cfg(cfg.training)
    assert guard_cfg is not None
    assert (guard_cfg.capture_interval, guard_cfg.check_interval, guard_cfg.defer_validation) == (
        32,
        32,
        True,
    )

    cfg.training.finite_validation_interval = 0
    with pytest.raises(ValueError, match="positive integer"):
        finite_validation_env_overrides(cfg)
