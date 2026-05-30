from __future__ import annotations

import sys
from importlib.machinery import ModuleSpec
from pathlib import Path

import pytest

from unilab import cli, demo


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


def test_demo_registry_contains_expected_entries() -> None:
    assert set(demo.DEMO_REGISTRY) == {
        "dance",
        "wallflip",
        "boxtracking",
        "locomani",
        "inhandgrasp",
        "teaser",
    }
    assert demo.DEMO_REGISTRY["locomani"].entry == "play_interactive"
    assert demo.DEMO_REGISTRY["locomani"].sim == "mujoco"
    assert demo.DEMO_REGISTRY["teaser"].entry == "teaser"
    for name in ("dance", "wallflip", "boxtracking", "inhandgrasp"):
        spec = demo.DEMO_REGISTRY[name]
        assert spec.entry == "eval"
        assert spec.sim == "motrix"
        assert spec.algo == "ppo"


def test_demo_eval_entry_passes_checkpoint_as_load_run_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_demo_checkout(tmp_path, demo_name="dance")
    _pretend_motrix_is_installed(monkeypatch)
    monkeypatch.setattr(cli.platform, "system", lambda: "Linux")

    abs_pt = str(tmp_path / "fake" / "model_0.pt")
    command = demo.build_demo_command(demo_name="dance", checkpoint_path=abs_pt, root=tmp_path)

    assert command[0] == sys.executable
    assert command[1] == str(tmp_path / "scripts" / "train_rsl_rl.py")
    assert "task=g1_motion_tracking/motrix" in command
    assert "training.play_only=true" in command
    assert f"algo.load_run={abs_pt}" in command


def test_demo_play_interactive_entry_assembles_locomani_command(
    tmp_path: Path,
) -> None:
    _make_demo_checkout(tmp_path, demo_name="locomani")
    abs_pt = str(tmp_path / "fake" / "model_0.pt")
    command = demo.build_demo_command(
        demo_name="locomani", checkpoint_path=abs_pt, device="cpu", root=tmp_path
    )

    assert command[0] == sys.executable
    assert command[1] == str(tmp_path / "scripts" / "play_interactive.py")
    assert "task=go2_arm_manip_loco/mujoco" in command
    assert f"algo.load_run={abs_pt}" in command
    assert "training.device=cpu" in command


def test_demo_play_interactive_requires_owner_yaml(tmp_path: Path) -> None:
    (tmp_path / "scripts").mkdir(parents=True)
    (tmp_path / "scripts" / "play_interactive.py").write_text("", encoding="utf-8")

    with pytest.raises(SystemExit, match="owner config"):
        demo.build_demo_command(
            demo_name="locomani",
            checkpoint_path="/tmp/fake/model_0.pt",
            root=tmp_path,
        )


def test_demo_play_interactive_requires_script(tmp_path: Path) -> None:
    spec = demo.DEMO_REGISTRY["locomani"]
    owner_dir = tmp_path / "conf" / spec.algo / "task" / spec.task
    owner_dir.mkdir(parents=True)
    (owner_dir / f"{spec.sim}.yaml").write_text("training:\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="play_interactive.py"):
        demo.build_demo_command(
            demo_name="locomani",
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


def test_demo_teaser_build_command_rejected() -> None:
    with pytest.raises(SystemExit, match="renderer-only"):
        demo.build_demo_command(demo_name="teaser", checkpoint_path="/unused.pt")


def test_demo_teaser_run_demo_invokes_render_teaser_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: list[str] = []

    def fake_render_teaser_main() -> None:
        called.append("rendered")

    import unilab.tools.render_teaser as render_teaser_module

    monkeypatch.setattr(render_teaser_module, "main", fake_render_teaser_main)

    def fail_resolve(_: str) -> str:
        raise AssertionError("teaser entry must not resolve a checkpoint")

    monkeypatch.setattr(demo, "resolve_checkpoint_file", fail_resolve)

    rc = demo.run_demo(demo_name="teaser")
    assert rc == 0
    assert called == ["rendered"]


def test_demo_main_teaser_dispatches_to_render_teaser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: list[str] = []

    def fake_render_teaser_main() -> None:
        called.append("rendered")

    import unilab.tools.render_teaser as render_teaser_module

    monkeypatch.setattr(render_teaser_module, "main", fake_render_teaser_main)
    rc = cli.demo_main(["teaser"])
    assert rc == 0
    assert called == ["rendered"]
