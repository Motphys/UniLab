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
