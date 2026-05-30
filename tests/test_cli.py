from __future__ import annotations

import sys
from importlib.machinery import ModuleSpec
from pathlib import Path

import pytest

from unilab import cli


def _make_minimal_checkout(
    root: Path, *, algo: str = "ppo", task: str = "go2_joystick_flat"
) -> None:
    (root / "scripts").mkdir(parents=True)
    (root / "scripts" / "train_rsl_rl.py").write_text("", encoding="utf-8")
    (root / "conf" / algo / "task" / task).mkdir(parents=True)
    (root / "conf" / algo / "task" / task / "motrix.yaml").write_text(
        "training:\n  sim_backend: motrix\n",
        encoding="utf-8",
    )


def _pretend_motrix_is_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cli,
        "find_spec",
        lambda name: ModuleSpec(name, loader=None) if name == "motrixsim" else None,
    )


def test_macos_motrix_train_uses_mxpython_when_playback_can_open_renderer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_minimal_checkout(tmp_path)
    _pretend_motrix_is_installed(monkeypatch)
    monkeypatch.setattr(cli.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        cli.shutil, "which", lambda name: "/opt/bin/mxpython" if name == "mxpython" else None
    )

    command = cli.build_command(
        mode="train",
        algo="ppo",
        task="go2_joystick_flat",
        sim="motrix",
        overrides=[],
        root=tmp_path,
    )

    assert command[0] == "/opt/bin/mxpython"
    assert command[1:] == [
        str(tmp_path / "scripts" / "train_rsl_rl.py"),
        "task=go2_joystick_flat/motrix",
    ]


def test_macos_motrix_train_no_play_uses_current_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_minimal_checkout(tmp_path)
    _pretend_motrix_is_installed(monkeypatch)
    monkeypatch.setattr(cli.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/opt/bin/mxpython")

    command = cli.build_command(
        mode="train",
        algo="ppo",
        task="go2_joystick_flat",
        sim="motrix",
        overrides=["training.no_play=true"],
        root=tmp_path,
    )

    assert command[0] == sys.executable


def test_train_profile_routes_to_owner_variant(tmp_path: Path) -> None:
    (tmp_path / "scripts").mkdir(parents=True)
    (tmp_path / "scripts" / "train_rsl_rl.py").write_text("", encoding="utf-8")
    owner_dir = tmp_path / "conf" / "ppo" / "task" / "sharpa_inhand"
    owner_dir.mkdir(parents=True)
    (owner_dir / "mujoco_hora.yaml").write_text(
        "training:\n  sim_backend: mujoco\n",
        encoding="utf-8",
    )

    command = cli.build_command(
        mode="train",
        algo="ppo",
        task="sharpa_inhand",
        sim="mujoco",
        profile="hora",
        overrides=[],
        root=tmp_path,
    )

    assert command[1:] == [
        str(tmp_path / "scripts" / "train_rsl_rl.py"),
        "task=sharpa_inhand/mujoco_hora",
    ]


def test_go2_arm_manip_loco_motrix_train_and_eval_route_to_owner_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_minimal_checkout(tmp_path, task="go2_arm_manip_loco")
    _pretend_motrix_is_installed(monkeypatch)
    monkeypatch.setattr(cli.platform, "system", lambda: "Linux")

    train_command = cli.build_command(
        mode="train",
        algo="ppo",
        task="go2_arm_manip_loco",
        sim="motrix",
        overrides=[],
        root=tmp_path,
    )
    eval_command = cli.build_command(
        mode="eval",
        algo="ppo",
        task="go2_arm_manip_loco",
        sim="motrix",
        overrides=[],
        load_run="-1",
        root=tmp_path,
    )

    assert train_command[1:] == [
        str(tmp_path / "scripts" / "train_rsl_rl.py"),
        "task=go2_arm_manip_loco/motrix",
    ]
    assert eval_command[1:3] == [
        str(tmp_path / "scripts" / "train_rsl_rl.py"),
        "task=go2_arm_manip_loco/motrix",
    ]
    assert "training.play_only=true" in eval_command
    assert "algo.load_run=-1" in eval_command


def test_macos_motrix_eval_requires_mxpython(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_minimal_checkout(tmp_path)
    _pretend_motrix_is_installed(monkeypatch)
    monkeypatch.setattr(cli.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)

    with pytest.raises(SystemExit, match="mxpython"):
        cli.build_command(
            mode="eval",
            algo="ppo",
            task="go2_joystick_flat",
            sim="motrix",
            overrides=[],
            load_run="-1",
            root=tmp_path,
        )


def test_eval_render_mode_generates_training_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_minimal_checkout(tmp_path)
    _pretend_motrix_is_installed(monkeypatch)
    monkeypatch.setattr(cli.platform, "system", lambda: "Linux")
    command = cli.build_command(
        mode="eval",
        algo="ppo",
        task="go2_joystick_flat",
        sim="motrix",
        overrides=[],
        load_run="-1",
        render_mode="record",
        root=tmp_path,
    )

    assert "training.play_render_mode=record" in command
    assert "training.play_only=true" in command
    assert "algo.load_run=-1" in command


def test_macos_motrix_render_mode_none_does_not_require_mxpython(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_minimal_checkout(tmp_path)
    _pretend_motrix_is_installed(monkeypatch)
    monkeypatch.setattr(cli.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)

    command = cli.build_command(
        mode="eval",
        algo="ppo",
        task="go2_joystick_flat",
        sim="motrix",
        overrides=[],
        load_run="-1",
        render_mode="none",
        root=tmp_path,
    )

    assert command[0] == sys.executable


def test_macos_motrix_render_mode_record_does_not_require_mxpython(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_minimal_checkout(tmp_path)
    _pretend_motrix_is_installed(monkeypatch)
    monkeypatch.setattr(cli.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)

    command = cli.build_command(
        mode="eval",
        algo="ppo",
        task="go2_joystick_flat",
        sim="motrix",
        overrides=[],
        load_run="-1",
        render_mode="record",
        root=tmp_path,
    )

    assert command[0] == sys.executable
