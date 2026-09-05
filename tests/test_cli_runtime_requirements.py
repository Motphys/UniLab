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
