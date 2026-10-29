from __future__ import annotations

import sys
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import SimpleNamespace

import pytest

from unilab import cli, demo


def test_eval_returns_standard_interrupt_code_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "build_command", lambda **_kwargs: ["child"])

    def raise_interrupt(_command):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "run_process_with_graceful_interrupt", raise_interrupt)

    assert (
        cli.eval_main(["--algo", "appo", "--task", "go2_joystick_flat", "--sim", "mujoco"]) == 130
    )


def _make_minimal_checkout(
    root: Path, *, algo: str = "ppo", task: str = "go2_joystick_flat"
) -> None:
    (root / "scripts").mkdir(parents=True)
    (root / "scripts" / "train_rsl_rl.py").write_text("", encoding="utf-8")
    (root / "conf" / algo / "task" / task).mkdir(parents=True)
    (root / "conf" / algo / "task" / task / "mujoco.yaml").write_text(
        "training:\n  sim_backend: mujoco\n",
        encoding="utf-8",
    )


def test_train_profile_routes_to_owner_variant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        cli,
        "find_spec",
        lambda name: object() if name in {"mujoco", "mjbatch"} else None,
    )
    (tmp_path / "scripts").mkdir(parents=True)
    (tmp_path / "scripts" / "train_rsl_rl.py").write_text("", encoding="utf-8")
    owner_dir = tmp_path / "conf" / "ppo" / "task" / "go2_joystick_flat"
    owner_dir.mkdir(parents=True)
    (owner_dir / "mujoco_nodr.yaml").write_text(
        "training:\n  sim_backend: mujoco\n",
        encoding="utf-8",
    )

    command = cli.build_command(
        mode="train",
        algo="ppo",
        task="go2_joystick_flat",
        sim="mujoco",
        profile="nodr",
        overrides=[],
        root=tmp_path,
    )

    assert command[1:] == [
        str(tmp_path / "scripts" / "train_rsl_rl.py"),
        "task=go2_joystick_flat/mujoco_nodr",
    ]


def test_eval_mujoco_interactive_routes_to_dedicated_viewer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir(parents=True)
    (scripts_dir / "train_rsl_rl.py").write_text("", encoding="utf-8")
    (scripts_dir / "play_interactive.py").write_text("", encoding="utf-8")
    owner_dir = tmp_path / "conf" / "ppo" / "task" / "go2_joystick_flat"
    owner_dir.mkdir(parents=True)
    (owner_dir / "mujoco.yaml").write_text("training:\n  sim_backend: mujoco\n", encoding="utf-8")
    monkeypatch.setattr(
        cli,
        "find_spec",
        lambda name: ModuleSpec(name, loader=None) if name in {"mujoco", "mjbatch"} else None,
    )
    monkeypatch.setattr(cli.platform, "system", lambda: "Linux")

    command = cli.build_command(
        mode="eval",
        algo="ppo",
        task="go2_joystick_flat",
        sim="mujoco",
        overrides=[],
        load_run="-1",
        render_mode="interactive",
        root=tmp_path,
    )

    assert command == [
        sys.executable,
        str(scripts_dir / "play_interactive.py"),
        "--algo",
        "ppo",
        "--task",
        "go2_joystick_flat",
        "--sim",
        "mujoco",
        "training.play_render_mode=interactive",
        "interactive.action_mode=policy",
        "training.play_only=true",
        "algo.load_run=-1",
    ]
    assert "task=go2_joystick_flat/mujoco" not in command


def test_eval_mujoco_interactive_honors_profile_and_render_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir(parents=True)
    (scripts_dir / "train_sac.py").write_text("", encoding="utf-8")
    (scripts_dir / "play_interactive.py").write_text("", encoding="utf-8")
    owner_dir = tmp_path / "conf" / "sac" / "task" / "g1_walk_flat"
    owner_dir.mkdir(parents=True)
    (owner_dir / "mujoco_nodr.yaml").write_text(
        "training:\n  sim_backend: mujoco\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        cli,
        "find_spec",
        lambda name: ModuleSpec(name, loader=None) if name in {"mujoco", "mjbatch"} else None,
    )
    monkeypatch.setattr(cli.platform, "system", lambda: "Linux")

    command = cli.build_command(
        mode="eval",
        algo="sac",
        task="g1_walk_flat",
        sim="mujoco",
        profile="nodr",
        overrides=["training.play_render_mode=interactive", "algo.num_envs=4096"],
        load_run="run_1",
        render_mode="record",
        root=tmp_path,
    )

    assert command == [
        sys.executable,
        str(scripts_dir / "play_interactive.py"),
        "--algo",
        "sac",
        "--task",
        "g1_walk_flat",
        "--sim",
        "mujoco_nodr",
        "interactive.action_mode=policy",
        "training.play_only=true",
        "algo.load_run=run_1",
        "training.play_render_mode=interactive",
        "algo.num_envs=4096",
    ]


def _make_superdex_eval_checkout(root: Path) -> None:
    (root / "scripts").mkdir(parents=True)
    (root / "scripts" / "train_rsl_rl.py").write_text("", encoding="utf-8")
    owner_dir = root / "conf" / "ppo" / "task" / "go2_joystick_flat"
    owner_dir.mkdir(parents=True)
    (owner_dir / "superdex.yaml").write_text(
        "training:\n  sim_backend: superdex\n", encoding="utf-8"
    )


def _pretend_superdex_runtime_is_available(monkeypatch: pytest.MonkeyPatch) -> None:
    from unisim.backend.superdex import dependencies

    monkeypatch.setattr(dependencies, "superdex_dependencies_available", lambda: True)


def test_eval_superdex_interactive_forces_single_env_on_train_play_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_superdex_eval_checkout(tmp_path)
    _pretend_superdex_runtime_is_available(monkeypatch)

    command = cli.build_command(
        mode="eval",
        algo="ppo",
        task="go2_joystick_flat",
        sim="superdex",
        overrides=[],
        load_run="-1",
        render_mode="interactive",
        root=tmp_path,
    )

    # SuperDex interactive eval keeps the train-script play path (the native
    # Polyscope viewer renders it) but collapses to the single scene the
    # viewer draws; the owner layer switches the env to the serial executor.
    assert command[1] == str(tmp_path / "scripts" / "train_rsl_rl.py")
    assert "training.play_render_mode=interactive" in command
    assert "training.play_env_num=1" in command


def test_eval_superdex_interactive_preserves_explicit_play_env_num(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_superdex_eval_checkout(tmp_path)
    _pretend_superdex_runtime_is_available(monkeypatch)

    command = cli.build_command(
        mode="eval",
        algo="ppo",
        task="go2_joystick_flat",
        sim="superdex",
        overrides=["training.play_env_num=4"],
        load_run="-1",
        render_mode="interactive",
        root=tmp_path,
    )

    assert "training.play_env_num=4" in command
    assert "training.play_env_num=1" not in command


def test_eval_superdex_record_does_not_force_play_env_num(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_superdex_eval_checkout(tmp_path)
    _pretend_superdex_runtime_is_available(monkeypatch)

    command = cli.build_command(
        mode="eval",
        algo="ppo",
        task="go2_joystick_flat",
        sim="superdex",
        overrides=[],
        load_run="-1",
        render_mode="record",
        root=tmp_path,
    )

    assert "training.play_render_mode=record" in command
    assert not any(o.startswith("training.play_env_num=") for o in command)


def test_eval_mujoco_interactive_preserves_explicit_action_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir(parents=True)
    (scripts_dir / "train_rsl_rl.py").write_text("", encoding="utf-8")
    (scripts_dir / "play_interactive.py").write_text("", encoding="utf-8")
    owner_dir = tmp_path / "conf" / "ppo" / "task" / "go2_joystick_flat"
    owner_dir.mkdir(parents=True)
    (owner_dir / "mujoco.yaml").write_text("training:\n  sim_backend: mujoco\n", encoding="utf-8")
    monkeypatch.setattr(
        cli,
        "find_spec",
        lambda name: ModuleSpec(name, loader=None) if name in {"mujoco", "mjbatch"} else None,
    )
    monkeypatch.setattr(cli.platform, "system", lambda: "Linux")

    command = cli.build_command(
        mode="eval",
        algo="ppo",
        task="go2_joystick_flat",
        sim="mujoco",
        overrides=["interactive.action_mode=random"],
        load_run="-1",
        render_mode="interactive",
        root=tmp_path,
    )

    assert "interactive.action_mode=policy" not in command
    assert "interactive.action_mode=random" in command


def test_eval_mujoco_interactive_falls_back_to_sibling_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir(parents=True)
    (scripts_dir / "train_rsl_rl.py").write_text("", encoding="utf-8")
    (scripts_dir / "play_interactive.py").write_text("", encoding="utf-8")
    owner_dir = tmp_path / "conf" / "ppo" / "task" / "go2_joystick_flat"
    owner_dir.mkdir(parents=True)
    (owner_dir / "mjwarp.yaml").write_text("training:\n  sim_backend: mjwarp\n", encoding="utf-8")
    monkeypatch.setattr(
        cli,
        "find_spec",
        lambda name: ModuleSpec(name, loader=None) if name in {"mujoco", "mjbatch"} else None,
    )
    monkeypatch.setattr(cli.platform, "system", lambda: "Linux")

    command = cli.build_command(
        mode="eval",
        algo="ppo",
        task="go2_joystick_flat",
        sim="mujoco",
        overrides=[],
        load_run="-1",
        render_mode="interactive",
        root=tmp_path,
    )

    assert command[:8] == [
        sys.executable,
        str(scripts_dir / "play_interactive.py"),
        "--algo",
        "ppo",
        "--task",
        "go2_joystick_flat",
        "--sim",
        "mjwarp",
    ]
    assert "training.sim_backend=mujoco" in command
    assert "training.play_only=true" in command


def _pretend_mujoco_is_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    installed = {"mujoco", "mjbatch"}
    monkeypatch.setattr(
        cli,
        "find_spec",
        lambda name: ModuleSpec(name, loader=None) if name in installed else None,
    )


def _make_mujoco_viser_checkout(root: Path) -> Path:
    scripts_dir = root / "scripts"
    scripts_dir.mkdir(parents=True)
    (scripts_dir / "train_rsl_rl.py").write_text("", encoding="utf-8")
    (scripts_dir / "play_interactive.py").write_text("", encoding="utf-8")
    (scripts_dir / "play_viser.py").write_text("", encoding="utf-8")
    owner_dir = root / "conf" / "ppo" / "task" / "go2_joystick_flat"
    owner_dir.mkdir(parents=True)
    (owner_dir / "mujoco.yaml").write_text("training:\n  sim_backend: mujoco\n", encoding="utf-8")
    return scripts_dir


def test_eval_mujoco_viser_routes_to_viser_viewer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scripts_dir = _make_mujoco_viser_checkout(tmp_path)
    _pretend_mujoco_is_installed(monkeypatch)

    command = cli.build_command(
        mode="eval",
        algo="ppo",
        task="go2_joystick_flat",
        sim="mujoco",
        overrides=[],
        load_run="-1",
        render_mode="viser",
        root=tmp_path,
    )

    assert command == [
        sys.executable,
        str(scripts_dir / "play_viser.py"),
        "--algo",
        "ppo",
        "--task",
        "go2_joystick_flat",
        "--sim",
        "mujoco",
        "training.play_render_mode=viser",
        "interactive.action_mode=policy",
        "training.play_only=true",
        "algo.load_run=-1",
    ]
    assert "task=go2_joystick_flat/mujoco" not in command


def test_eval_mujoco_viser_via_passthrough_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scripts_dir = _make_mujoco_viser_checkout(tmp_path)
    _pretend_mujoco_is_installed(monkeypatch)

    command = cli.build_command(
        mode="eval",
        algo="ppo",
        task="go2_joystick_flat",
        sim="mujoco",
        overrides=["training.play_render_mode=viser"],
        load_run="-1",
        root=tmp_path,
    )

    assert command[1] == str(scripts_dir / "play_viser.py")
    assert "training.play_render_mode=viser" in command


def test_eval_mjwarp_viser_routes_to_viser_viewer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scripts_dir = _make_mjwarp_checkout(tmp_path)
    (scripts_dir / "play_viser.py").write_text("", encoding="utf-8")
    _pretend_mjwarp_is_installed(monkeypatch)
    monkeypatch.setattr(
        cli,
        "find_spec",
        lambda name: (
            ModuleSpec(name, loader=None)
            if name in {"mujoco", "mujoco_warp", "warp", "viser"}
            else None
        ),
    )

    command = cli.build_command(
        mode="eval",
        algo="ppo",
        task="g1_walk_flat",
        sim="mjwarp",
        overrides=[],
        load_run="-1",
        render_mode="viser",
        root=tmp_path,
    )

    assert command[1] == str(scripts_dir / "play_viser.py")
    assert command[:8] == [
        sys.executable,
        str(scripts_dir / "play_viser.py"),
        "--algo",
        "ppo",
        "--task",
        "g1_walk_flat",
        "--sim",
        "mjwarp",
    ]


def test_macos_mujoco_viser_eval_uses_current_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The viser viewer is browser-based: no GLFW/Cocoa, so no mjpython.
    _make_mujoco_viser_checkout(tmp_path)
    _pretend_mujoco_is_installed(monkeypatch)
    monkeypatch.setattr(cli.platform, "system", lambda: "Darwin")

    command = cli.build_command(
        mode="eval",
        algo="ppo",
        task="go2_joystick_flat",
        sim="mujoco",
        overrides=[],
        load_run="-1",
        render_mode="viser",
        root=tmp_path,
    )

    assert command[0] == sys.executable


def test_train_viser_render_mode_routes_to_train_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Train-mode viser playback runs inside the train script's post-training
    playback: the CLI forwards training.play_render_mode=viser unchanged."""
    scripts_dir = _make_mujoco_viser_checkout(tmp_path)
    _pretend_mujoco_is_installed(monkeypatch)

    command = cli.build_command(
        mode="train",
        algo="ppo",
        task="go2_joystick_flat",
        sim="mujoco",
        overrides=[],
        render_mode="viser",
        root=tmp_path,
    )

    assert command[1] == str(scripts_dir / "train_rsl_rl.py")
    assert "task=go2_joystick_flat/mujoco" in command
    assert "training.play_render_mode=viser" in command


def _pretend_mjwarp_is_installed(
    monkeypatch: pytest.MonkeyPatch, *, with_mujoco: bool = True
) -> None:
    installed = {"mujoco_warp", "warp"} | ({"mujoco"} if with_mujoco else set())
    monkeypatch.setattr(
        cli,
        "find_spec",
        lambda name: ModuleSpec(name, loader=None) if name in installed else None,
    )


def _make_mjwarp_checkout(root: Path) -> Path:
    scripts_dir = root / "scripts"
    scripts_dir.mkdir(parents=True)
    (scripts_dir / "train_rsl_rl.py").write_text("", encoding="utf-8")
    (scripts_dir / "play_interactive.py").write_text("", encoding="utf-8")
    owner_dir = root / "conf" / "ppo" / "task" / "g1_walk_flat"
    owner_dir.mkdir(parents=True)
    (owner_dir / "mjwarp.yaml").write_text("training:\n  sim_backend: mjwarp\n", encoding="utf-8")
    return scripts_dir


def test_eval_mjwarp_interactive_routes_to_dedicated_viewer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scripts_dir = _make_mjwarp_checkout(tmp_path)
    _pretend_mjwarp_is_installed(monkeypatch)
    monkeypatch.setattr(cli.platform, "system", lambda: "Linux")

    command = cli.build_command(
        mode="eval",
        algo="ppo",
        task="g1_walk_flat",
        sim="mjwarp",
        overrides=[],
        load_run="-1",
        render_mode="interactive",
        root=tmp_path,
    )

    assert command == [
        sys.executable,
        str(scripts_dir / "play_interactive.py"),
        "--algo",
        "ppo",
        "--task",
        "g1_walk_flat",
        "--sim",
        "mjwarp",
        "training.play_render_mode=interactive",
        "interactive.action_mode=policy",
        "training.play_only=true",
        "algo.load_run=-1",
    ]
    assert "task=g1_walk_flat/mjwarp" not in command


def test_eval_mjwarp_record_stays_on_train_script_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scripts_dir = _make_mjwarp_checkout(tmp_path)
    _pretend_mjwarp_is_installed(monkeypatch)

    command = cli.build_command(
        mode="eval",
        algo="ppo",
        task="g1_walk_flat",
        sim="mjwarp",
        overrides=[],
        load_run="-1",
        render_mode="record",
        root=tmp_path,
    )

    assert command[:2] == [sys.executable, str(scripts_dir / "train_rsl_rl.py")]
    assert "task=g1_walk_flat/mjwarp" in command
    assert "training.play_render_mode=record" in command


def test_eval_mjwarp_interactive_requires_mujoco_viewer_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_mjwarp_checkout(tmp_path)
    _pretend_mjwarp_is_installed(monkeypatch, with_mujoco=False)

    with pytest.raises(SystemExit, match="MuJoCo"):
        cli.build_command(
            mode="eval",
            algo="ppo",
            task="g1_walk_flat",
            sim="mjwarp",
            overrides=[],
            load_run="-1",
            render_mode="interactive",
            root=tmp_path,
        )


def _make_appo_mujoco_checkout(root: Path) -> Path:
    scripts_dir = root / "scripts"
    scripts_dir.mkdir(parents=True)
    (scripts_dir / "train_appo.py").write_text("", encoding="utf-8")
    (scripts_dir / "play_interactive.py").write_text("", encoding="utf-8")
    owner_dir = root / "conf" / "appo" / "task" / "go2_joystick_flat"
    owner_dir.mkdir(parents=True)
    (owner_dir / "mujoco.yaml").write_text("training:\n  sim_backend: mujoco\n", encoding="utf-8")
    return scripts_dir


def _pretend_mujoco_is_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cli,
        "find_spec",
        lambda name: ModuleSpec(name, loader=None) if name in {"mujoco", "mjbatch"} else None,
    )


def test_macos_mujoco_interactive_eval_uses_mjpython(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scripts_dir = _make_appo_mujoco_checkout(tmp_path)
    _pretend_mujoco_is_installed(monkeypatch)
    venv_bin = tmp_path / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    fake_python = venv_bin / "python"
    fake_mjpython = venv_bin / "mjpython"
    fake_python.write_text("", encoding="utf-8")
    fake_mjpython.write_text("", encoding="utf-8")
    monkeypatch.setattr(cli.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(cli.sys, "executable", str(fake_python))
    monkeypatch.setattr(cli, "_ensure_mujoco_mjpython_app", lambda: None)

    command = cli.build_command(
        mode="eval",
        algo="appo",
        task="go2_joystick_flat",
        sim="mujoco",
        overrides=[],
        load_run="-1",
        render_mode="interactive",
        root=tmp_path,
    )

    assert command[0] == str(fake_mjpython)
    assert command[1] == str(scripts_dir / "play_interactive.py")


def test_macos_mujoco_interactive_eval_requires_mjpython(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_appo_mujoco_checkout(tmp_path)
    _pretend_mujoco_is_installed(monkeypatch)
    monkeypatch.setattr(cli.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(cli.sys, "executable", str(tmp_path / "python"))
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    monkeypatch.setattr(cli, "_ensure_mujoco_mjpython_app", lambda: None)

    with pytest.raises(SystemExit, match="mjpython"):
        cli.build_command(
            mode="eval",
            algo="appo",
            task="go2_joystick_flat",
            sim="mujoco",
            overrides=[],
            load_run="-1",
            render_mode="interactive",
            root=tmp_path,
        )


def test_macos_mujoco_interactive_eval_requires_mjpython_app(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_appo_mujoco_checkout(tmp_path)
    fake_package = tmp_path / "site-packages" / "mujoco"
    fake_package.mkdir(parents=True)
    monkeypatch.setattr(
        cli,
        "find_spec",
        lambda name: (
            ModuleSpec(name, loader=None, origin=str(fake_package / "__init__.py"))
            if name in {"mujoco", "mjbatch"}
            else None
        ),
    )
    monkeypatch.setattr(cli.platform, "system", lambda: "Darwin")

    with pytest.raises(SystemExit, match=r"MuJoCo_\(mjpython\)\.app"):
        cli.build_command(
            mode="eval",
            algo="appo",
            task="go2_joystick_flat",
            sim="mujoco",
            overrides=[],
            load_run="-1",
            render_mode="interactive",
            root=tmp_path,
        )


def test_macos_mujoco_record_eval_uses_current_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_appo_mujoco_checkout(tmp_path)
    _pretend_mujoco_is_installed(monkeypatch)
    monkeypatch.setattr(cli.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)

    command = cli.build_command(
        mode="eval",
        algo="appo",
        task="go2_joystick_flat",
        sim="mujoco",
        overrides=[],
        load_run="-1",
        render_mode="record",
        root=tmp_path,
    )

    assert command[0] == sys.executable
    assert command[1] == str(tmp_path / "scripts" / "train_appo.py")


def _make_demo_checkout(root: Path, *, demo_name: str) -> None:
    spec = demo.DEMO_REGISTRY[demo_name]
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "scripts" / "train_rsl_rl.py").write_text("", encoding="utf-8")
    (root / "scripts" / "play_interactive.py").write_text("", encoding="utf-8")
    owner_dir = root / "conf" / spec.algo / "task" / spec.task
    owner_dir.mkdir(parents=True, exist_ok=True)
    (owner_dir / f"{spec.sim}.yaml").write_text(
        f"training:\n  sim_backend: {spec.sim}\n", encoding="utf-8"
    )


def _register_play_interactive_demo(monkeypatch: pytest.MonkeyPatch) -> str:
    name = "playdemo"
    monkeypatch.setitem(
        demo.DEMO_REGISTRY,
        name,
        demo.DemoSpec(algo="ppo", task="g1_walk_flat", sim="mujoco", entry="play_interactive"),
    )
    return name


def test_demo_registry_contains_expected_entries() -> None:
    assert set(demo.DEMO_REGISTRY) == {
        "dance",
        "teaser",
    }
    assert demo.DEMO_REGISTRY["teaser"].entry == "teaser"
    for name in ("dance",):
        spec = demo.DEMO_REGISTRY[name]
        assert spec.entry == "eval"
        assert spec.sim == "mujoco"
        assert spec.algo == "ppo"


def test_demo_eval_entry_passes_checkpoint_as_load_run_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_demo_checkout(tmp_path, demo_name="dance")
    monkeypatch.setattr(
        cli,
        "find_spec",
        lambda name: object() if name in {"mujoco", "mjbatch"} else None,
    )
    monkeypatch.setattr(cli.platform, "system", lambda: "Linux")

    abs_pt = str(tmp_path / "fake" / "model_0.pt")
    command = demo.build_demo_command(demo_name="dance", checkpoint_path=abs_pt, root=tmp_path)

    assert command[0] == sys.executable
    assert command[1] == str(tmp_path / "scripts" / "train_rsl_rl.py")
    assert "task=g1_motion_tracking/mujoco" in command
    assert "training.play_only=true" in command
    assert f"algo.load_run={abs_pt}" in command


def test_demo_play_interactive_sac_owner_path_uses_sac_tree(tmp_path: Path) -> None:
    (tmp_path / "scripts").mkdir(parents=True, exist_ok=True)
    (tmp_path / "scripts" / "play_interactive.py").write_text("", encoding="utf-8")
    owner_dir = tmp_path / "conf" / "sac" / "task" / "g1_walk_flat"
    owner_dir.mkdir(parents=True)
    (owner_dir / "mujoco_nodr.yaml").write_text(
        "training:\n  sim_backend: mujoco\n", encoding="utf-8"
    )
    spec = demo.DemoSpec(
        algo="sac",
        task="g1_walk_flat",
        sim="mujoco_nodr",
        entry="play_interactive",
    )

    command = demo._build_play_interactive_command(
        spec=spec,
        checkpoint_path="/tmp/model_0.pt",
        extra_overrides=[],
        root=tmp_path,
    )

    assert command[1:] == [
        str(tmp_path / "scripts" / "play_interactive.py"),
        "--algo",
        "sac",
        "--task",
        "g1_walk_flat",
        "--sim",
        "mujoco_nodr",
        "algo.load_run=/tmp/model_0.pt",
    ]


def test_demo_play_interactive_linux_does_not_materialize_mjpython_app(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demo_name = _register_play_interactive_demo(monkeypatch)
    _make_demo_checkout(tmp_path, demo_name=demo_name)
    monkeypatch.setattr(demo.platform, "system", lambda: "Linux")

    def fail_materialize() -> None:
        raise AssertionError("Linux demo path must not touch macOS mjpython setup")

    monkeypatch.setattr(cli, "_ensure_mujoco_mjpython_app", fail_materialize)

    command = demo.build_demo_command(
        demo_name=demo_name,
        checkpoint_path="/tmp/fake/model_0.pt",
        root=tmp_path,
    )

    assert command[0] == sys.executable


def test_demo_play_interactive_uses_mjpython_on_macos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demo_name = _register_play_interactive_demo(monkeypatch)
    _make_demo_checkout(tmp_path, demo_name=demo_name)
    venv_bin = tmp_path / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    fake_python = venv_bin / "python"
    fake_mjpython = venv_bin / "mjpython"
    fake_python.write_text("", encoding="utf-8")
    fake_mjpython.write_text("", encoding="utf-8")
    monkeypatch.setattr(demo.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(demo.sys, "executable", str(fake_python))
    monkeypatch.setattr(cli, "_ensure_mujoco_mjpython_app", lambda: None)

    command = demo.build_demo_command(
        demo_name=demo_name,
        checkpoint_path="/tmp/fake/model_0.pt",
        root=tmp_path,
    )

    assert command[0] == str(fake_mjpython)
    assert command[1] == str(tmp_path / "scripts" / "play_interactive.py")


def test_demo_play_interactive_checks_mujoco_mjpython_app_on_macos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    demo_name = _register_play_interactive_demo(monkeypatch)
    _make_demo_checkout(tmp_path, demo_name=demo_name)
    monkeypatch.setattr(demo.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(cli, "_ensure_mujoco_mjpython_app", lambda: calls.append("checked"))
    monkeypatch.setattr(cli, "_mjpython_executable", lambda: "/tmp/mjpython")

    command = demo.build_demo_command(
        demo_name=demo_name,
        checkpoint_path="/tmp/fake/model_0.pt",
        root=tmp_path,
    )

    assert command[0] == "/tmp/mjpython"
    assert calls == ["checked"]


def test_demo_play_interactive_requires_owner_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demo_name = _register_play_interactive_demo(monkeypatch)
    (tmp_path / "scripts").mkdir(parents=True)
    (tmp_path / "scripts" / "play_interactive.py").write_text("", encoding="utf-8")

    with pytest.raises(SystemExit, match="owner config"):
        demo.build_demo_command(
            demo_name=demo_name,
            checkpoint_path="/tmp/fake/model_0.pt",
            root=tmp_path,
        )


def test_demo_play_interactive_requires_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demo_name = _register_play_interactive_demo(monkeypatch)
    spec = demo.DEMO_REGISTRY[demo_name]
    owner_dir = tmp_path / "conf" / spec.algo / "task" / spec.task
    owner_dir.mkdir(parents=True)
    (owner_dir / f"{spec.sim}.yaml").write_text("training:\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="play_interactive.py"):
        demo.build_demo_command(
            demo_name=demo_name,
            checkpoint_path="/tmp/fake/model_0.pt",
            root=tmp_path,
        )


def test_demo_unknown_name_lists_available_demos() -> None:
    with pytest.raises(SystemExit, match="Available demos"):
        demo.get_demo_spec("not_a_real_demo")


def test_demo_main_rejects_passthrough_overrides() -> None:
    with pytest.raises(SystemExit, match="passthrough"):
        cli.demo_main(["dance", "training.device=cpu"])


def test_demo_main_unknown_name_raises_with_available_list() -> None:
    with pytest.raises(SystemExit, match="Available demos"):
        cli.demo_main(["mystery"])


def test_demo_local_only_checkpoint_missing_warns_without_hf_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(demo, "ASSETS_ROOT_PATH", tmp_path / "assets")

    def fail_resolve(_: str) -> str:
        raise AssertionError("local-only demo must not download from Hugging Face")

    monkeypatch.setattr(demo, "resolve_checkpoint_file", fail_resolve)
    monkeypatch.setitem(
        demo.DEMO_REGISTRY,
        "localonly",
        demo.DemoSpec(algo="ppo", task="g1_walk_flat", sim="mujoco", entry="play_interactive"),
    )
    monkeypatch.setattr(demo, "_LOCAL_ONLY_CHECKPOINT_DEMOS", {"localonly"})

    rc = demo.run_demo(demo_name="localonly")

    output = capsys.readouterr().out
    assert rc == 1
    assert "Checkpoint not found" in output
    assert "checkpoints/localonly/model_0.pt" in output.replace("\\", "/")


def test_demo_local_only_checkpoint_uses_existing_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = tmp_path / "checkout"
    assets = tmp_path / "assets"
    checkpoint = assets / "checkpoints" / "localonly" / "model_0.pt"
    monkeypatch.setitem(
        demo.DEMO_REGISTRY,
        "localonly",
        demo.DemoSpec(algo="ppo", task="g1_walk_flat", sim="mujoco", entry="play_interactive"),
    )
    monkeypatch.setattr(demo, "_LOCAL_ONLY_CHECKPOINT_DEMOS", {"localonly"})
    _make_demo_checkout(checkout, demo_name="localonly")
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    calls: list[list[str]] = []

    monkeypatch.setattr(demo, "ASSETS_ROOT_PATH", assets)
    monkeypatch.setattr(demo, "_package_root", lambda: checkout)
    monkeypatch.setattr(demo, "_dev_venv", lambda: checkout / ".venv")
    monkeypatch.setattr(demo.platform, "system", lambda: "Linux")

    def fail_resolve(_: str) -> str:
        raise AssertionError("local-only demo must not download from Hugging Face")

    def fake_run(command: list[str], *, check: bool, env: dict[str, str]) -> SimpleNamespace:
        assert check is False
        assert env["UV_PROJECT_ENVIRONMENT"] == str(checkout / ".venv")
        calls.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(demo, "resolve_checkpoint_file", fail_resolve)
    monkeypatch.setattr(demo.subprocess, "run", fake_run)

    rc = demo.run_demo(demo_name="localonly", device="cpu")

    assert rc == 0
    assert calls == [
        [
            sys.executable,
            str(checkout / "scripts" / "play_interactive.py"),
            "--algo",
            "ppo",
            "--task",
            "g1_walk_flat",
            "--sim",
            "mujoco",
            f"algo.load_run={checkpoint}",
            "training.device=cpu",
        ]
    ]


def test_demo_teaser_build_command_rejected() -> None:
    with pytest.raises(SystemExit, match="renderer-only"):
        demo.build_demo_command(demo_name="teaser", checkpoint_path="/unused.pt")


def test_demo_teaser_run_demo_invokes_render_teaser_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: list[str] = []
    monkeypatch.setattr(demo.platform, "system", lambda: "Linux")

    def fake_render_teaser_main() -> None:
        called.append("rendered")

    import unilab.visualization.teaser as render_teaser_module

    monkeypatch.setattr(render_teaser_module, "main", fake_render_teaser_main)

    def fail_resolve(_: str) -> str:
        raise AssertionError("teaser entry must not resolve a checkpoint")

    monkeypatch.setattr(demo, "resolve_checkpoint_file", fail_resolve)

    rc = demo.run_demo(demo_name="teaser")
    assert rc == 0
    assert called == ["rendered"]


def test_demo_teaser_uses_mxpython_subprocess_on_macos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], dict[str, str]]] = []
    resolved_scenes: list[str] = []
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.setattr(demo.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(demo.sys, "executable", "/tmp/unilab/.venv/bin/python")
    monkeypatch.setattr(demo, "_motrix_python_executable", lambda: "/tmp/unilab/.venv/bin/mxpython")
    monkeypatch.setattr(demo, "resolve_scene_dir", resolved_scenes.append)

    def fake_run(command: list[str], *, check: bool, env: dict[str, str]) -> SimpleNamespace:
        assert check is False
        assert resolved_scenes == ["scenes/teaser"]
        calls.append((command, env))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(demo.subprocess, "run", fake_run)

    def fail_render_teaser_main() -> None:
        raise AssertionError("macOS teaser must route through mxpython")

    import unilab.visualization.teaser as render_teaser_module

    monkeypatch.setattr(render_teaser_module, "main", fail_render_teaser_main)

    rc = demo.run_demo(demo_name="teaser")

    assert rc == 0
    command, env = calls[0]
    assert command == [
        "/tmp/unilab/.venv/bin/mxpython",
        str(demo._package_root() / "visualization" / "teaser.py"),
    ]
    assert env["PYTHONPATH"] == str(demo._package_root().parent)
    dev_venv = demo._dev_venv()
    if dev_venv is not None:
        assert env["UV_PROJECT_ENVIRONMENT"] == str(dev_venv)
    else:
        assert "UV_PROJECT_ENVIRONMENT" not in env


def test_demo_main_teaser_dispatches_to_render_teaser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: list[str] = []
    monkeypatch.setattr(demo.platform, "system", lambda: "Linux")

    def fake_render_teaser_main() -> None:
        called.append("rendered")

    import unilab.visualization.teaser as render_teaser_module

    monkeypatch.setattr(render_teaser_module, "main", fake_render_teaser_main)
    rc = cli.demo_main(["teaser"])
    assert rc == 0
    assert called == ["rendered"]


def _pretend_genesis_runtime(monkeypatch: pytest.MonkeyPatch, available: bool) -> None:
    from unisim.backend.genesis import dependencies as genesis_deps

    monkeypatch.setattr(genesis_deps, "genesis_dependencies_available", lambda: available)


def test_genesis_runtime_dependency_probe_runs_after_scope_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "scripts").mkdir(parents=True)
    (tmp_path / "scripts" / "train_rsl_rl.py").write_text("", encoding="utf-8")
    (tmp_path / "conf").mkdir()
    _pretend_genesis_runtime(monkeypatch, available=False)

    with pytest.raises(SystemExit, match="uv sync --extra genesis"):
        cli.build_command(
            mode="train",
            algo="ppo",
            task="g1_walk_flat",
            sim="genesis",
            overrides=[],
            root=tmp_path,
        )


def _make_custom_algo_checkout(
    root: Path,
    *,
    algo: str = "dreamer",
    task: str = "go2_joystick_flat",
    with_script: bool = True,
    with_config: bool = True,
    with_owner: bool = True,
) -> None:
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    if with_script:
        (root / "scripts" / f"train_{algo}.py").write_text("", encoding="utf-8")
    if with_config:
        (root / "conf" / algo).mkdir(parents=True, exist_ok=True)
        (root / "conf" / algo / "config.yaml").write_text("algo: {}\n", encoding="utf-8")
    if with_owner:
        owner_dir = root / "conf" / algo / "task" / task
        owner_dir.mkdir(parents=True, exist_ok=True)
        (owner_dir / "mujoco.yaml").write_text(
            "training:\n  sim_backend: mujoco\n",
            encoding="utf-8",
        )


def test_convention_routes_custom_algo_with_config_and_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_custom_algo_checkout(tmp_path)
    monkeypatch.setattr(
        cli,
        "find_spec",
        lambda name: object() if name in {"mujoco", "mjbatch"} else None,
    )
    monkeypatch.setattr(cli.platform, "system", lambda: "Linux")

    command = cli.build_command(
        mode="train",
        algo="dreamer",
        task="go2_joystick_flat",
        sim="mujoco",
        overrides=[],
        root=tmp_path,
    )

    assert command[1:] == [
        str(tmp_path / "scripts" / "train_dreamer.py"),
        "task=go2_joystick_flat/mujoco",
    ]
    assert "dreamer" in cli.available_algos(tmp_path)


def test_convention_requires_entrypoint_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_custom_algo_checkout(tmp_path, with_script=False)
    monkeypatch.setattr(
        cli,
        "find_spec",
        lambda name: object() if name in {"mujoco", "mjbatch"} else None,
    )
    monkeypatch.setattr(cli.platform, "system", lambda: "Linux")

    with pytest.raises(SystemExit, match="Unsupported algo='dreamer'"):
        cli.build_command(
            mode="train",
            algo="dreamer",
            task="go2_joystick_flat",
            sim="mujoco",
            overrides=[],
            root=tmp_path,
        )
    assert "dreamer" not in cli.available_algos(tmp_path)


def test_convention_requires_config_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _make_custom_algo_checkout(tmp_path, with_config=False)
    monkeypatch.setattr(
        cli,
        "find_spec",
        lambda name: object() if name in {"mujoco", "mjbatch"} else None,
    )
    monkeypatch.setattr(cli.platform, "system", lambda: "Linux")

    with pytest.raises(SystemExit, match="Unsupported algo='dreamer'"):
        cli.build_command(
            mode="train",
            algo="dreamer",
            task="go2_joystick_flat",
            sim="mujoco",
            overrides=[],
            root=tmp_path,
        )


def test_unknown_algo_error_lists_builtin_and_discovered_algos(tmp_path: Path) -> None:
    _make_custom_algo_checkout(tmp_path)

    with pytest.raises(SystemExit, match="choose one of") as excinfo:
        cli.build_route("dqn", "go2_joystick_flat", "mujoco", root=tmp_path)

    message = str(excinfo.value)
    assert "Unsupported algo='dqn'" in message
    for builtin in ("ppo", "appo", "sac", "flashsac"):
        assert builtin in message
    assert "dreamer" in message


def test_malformed_algo_name_is_rejected_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="Unsupported algo"):
        cli.build_route("../etc", "go2_joystick_flat", "mujoco", root=tmp_path)
