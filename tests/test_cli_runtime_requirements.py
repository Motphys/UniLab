from __future__ import annotations

import pytest

from unilab import cli


def test_check_runtime_requirements_requires_mujoco_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "find_spec", lambda name: None if name == "mujoco" else object())

    with pytest.raises(SystemExit, match="sim=mujoco requires the MuJoCo extra"):
        cli._check_runtime_requirements("ppo", "mujoco")


def test_check_runtime_requirements_requires_motrix_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "find_spec", lambda name: None if name == "motrixsim" else object())

    with pytest.raises(SystemExit, match="sim=motrix requires the Motrix extra"):
        cli._check_runtime_requirements("ppo", "motrix")


def test_check_runtime_requirements_requires_drake_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "find_spec", lambda name: None if name == "drake_uni" else object())

    with pytest.raises(SystemExit, match="sim=drake requires the Drake extra"):
        cli._check_runtime_requirements("ppo", "drake")


def test_check_runtime_requirements_requires_isolated_newton_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli,
        "find_spec",
        lambda name: None if name == "newton" else object(),
    )

    with pytest.raises(SystemExit, match=r"sim=newton.*uv sync --extra newton"):
        cli._check_runtime_requirements("ppo", "newton")


def test_newton_is_a_supported_sim() -> None:
    assert "newton" in cli.SUPPORTED_SIMS


def test_superdex_missing_runtime_reports_python_and_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    from unisim.backend.superdex import dependencies

    monkeypatch.setattr(dependencies, "superdex_dependencies_available", lambda: False)
    with pytest.raises(SystemExit, match="Python 3.12.*Physics/Robotics"):
        cli._check_runtime_requirements("ppo", "superdex")


def test_superdex_old_unisim_reports_local_link_requirement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "unisim.backend.superdex.dependencies", None)
    with pytest.raises(SystemExit, match="locally linked UniSim SuperDex"):
        cli._check_runtime_requirements("ppo", "superdex")
