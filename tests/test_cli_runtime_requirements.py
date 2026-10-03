from __future__ import annotations

import pytest

from unilab import cli


def test_check_runtime_requirements_requires_mujoco_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "find_spec", lambda name: None if name == "mujoco" else object())

    with pytest.raises(SystemExit, match="sim=mujoco requires the MuJoCo extra"):
        cli._check_runtime_requirements("ppo", "mujoco")


def test_check_runtime_requirements_mujoco_needs_mjbatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Plain `mujoco` can arrive via other extras (e.g. superdex); the MuJoCo
    # physics backend is only usable with the mjbatch batch engine.
    monkeypatch.setattr(cli, "find_spec", lambda name: None if name == "mjbatch" else object())

    with pytest.raises(SystemExit, match="sim=mujoco requires the MuJoCo extra"):
        cli._check_runtime_requirements("ppo", "mujoco")


@pytest.mark.parametrize("sim", cli._SHELVED_SIMS)
def test_shelved_sims_fail_closed_before_dependency_detection(sim: str) -> None:
    with pytest.raises(SystemExit, match="temporarily outside the tensor-only Manager runtime"):
        cli._check_runtime_requirements("ppo", sim)


def test_check_runtime_requirements_requires_newton_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "find_spec", lambda name: None if name == "newton" else object())

    with pytest.raises(SystemExit, match="sim=newton requires the newton extra"):
        cli._check_runtime_requirements("sac", "newton")


def test_cli_backend_choices_are_tensor_manager_scope() -> None:
    assert cli.SUPPORTED_SIMS == ("mujoco", "mjwarp", "genesis", "newton")
