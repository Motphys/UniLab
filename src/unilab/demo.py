"""Demo entrypoint: fetch checkpoint from HF, then launch interactive playback."""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from unilab.assets import ASSETS_ROOT_PATH
from unilab.assets.hub import resolve_checkpoint_file

_OFFICIAL_MUJOCO_PACKAGE = "mujoco==3.8.0"


@dataclass(frozen=True)
class DemoSpec:
    algo: str
    task: str
    sim: str
    entry: str  # "eval", "play_interactive", or "teaser"


DEMO_REGISTRY: dict[str, DemoSpec] = {
    "dance": DemoSpec(algo="ppo", task="g1_motion_tracking", sim="motrix", entry="eval"),
    "wallflip": DemoSpec(algo="ppo", task="g1_wall_flip_tracking", sim="motrix", entry="eval"),
    "boxtracking": DemoSpec(algo="ppo", task="g1_box_tracking", sim="motrix", entry="eval"),
    "locomani": DemoSpec(
        algo="ppo", task="go2_arm_manip_loco", sim="mujoco", entry="play_interactive"
    ),
    "inhandgrasp": DemoSpec(
        algo="hora_distill",
        task="sharpa_inhand",
        sim="mujoco_nodr",
        entry="play_interactive",
    ),
    "sharpa_appo_student": DemoSpec(
        algo="hora_distill",
        task="sharpa_inhand",
        sim="mujoco_nodr",
        entry="play_interactive",
    ),
    "teaser": DemoSpec(algo="", task="", sim="", entry="teaser"),
}

_LOCAL_ONLY_CHECKPOINT_DEMOS = {"sharpa_appo_student"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_demo_spec(demo_name: str) -> DemoSpec:
    try:
        return DEMO_REGISTRY[demo_name]
    except KeyError as exc:
        available = ", ".join(sorted(DEMO_REGISTRY))
        raise SystemExit(f"Unknown demo {demo_name!r}. Available demos: {available}") from exc


def _checkpoint_relative_path(demo_name: str) -> str:
    return f"checkpoints/{demo_name}/model_0.pt"


def _local_checkpoint_path(demo_name: str) -> Path:
    return ASSETS_ROOT_PATH / _checkpoint_relative_path(demo_name)


def _refresh_local_checkpoint(demo_name: str) -> None:
    local = _local_checkpoint_path(demo_name)
    if local.exists():
        local.unlink()


def _resolve_demo_checkpoint(demo_name: str) -> str | None:
    local = _local_checkpoint_path(demo_name)
    if demo_name not in _LOCAL_ONLY_CHECKPOINT_DEMOS:
        resolved = resolve_checkpoint_file(_checkpoint_relative_path(demo_name))
        assert isinstance(resolved, str)
        return resolved

    if local.exists():
        return str(local)

    print(
        f"Local checkpoint not found for demo {demo_name!r}: {local}\n"
        "Checkpoint not found; no download source is used for this demo.\n"
        "Place the stage-2 PyTorch checkpoint at this path and run the demo again."
    )
    return None


def _build_play_interactive_command(
    *,
    spec: DemoSpec,
    checkpoint_path: str,
    extra_overrides: Sequence[str],
    root: Path | None = None,
) -> list[str]:
    selected_root = root or _repo_root()
    script = selected_root / "scripts" / "play_interactive.py"
    if not script.is_file():
        raise SystemExit(f"Entrypoint script not found: {script}")
    if spec.algo == "sac":
        owner_yaml = (
            selected_root / "conf" / "offpolicy" / "task" / "sac" / spec.task / f"{spec.sim}.yaml"
        )
    elif spec.algo == "hora_distill":
        owner_yaml = (
            selected_root / "conf" / "hora_distill" / "task" / spec.task / f"{spec.sim}.yaml"
        )
    else:
        owner_yaml = selected_root / "conf" / spec.algo / "task" / spec.task / f"{spec.sim}.yaml"
    if not owner_yaml.is_file():
        raise SystemExit(
            f"No owner config exists for algo={spec.algo}, task={spec.task}, sim={spec.sim}: "
            f"{owner_yaml}"
        )
    command = [
        *_play_interactive_command_prefix(selected_root),
        str(script),
        "--algo",
        spec.algo,
    ]
    command.extend(
        [
            f"task={spec.task}/{spec.sim}",
            f"algo.load_run={checkpoint_path}",
            *extra_overrides,
        ]
    )
    return command


def _play_interactive_command_prefix(root: Path) -> list[str]:
    if platform.system() != "Darwin":
        return [sys.executable]

    _ensure_mujoco_uni_mjpython_app(root)
    return [_current_env_mjpython()]


def _official_mujoco_env(root: Path) -> Path:
    return root / ".tmp" / "mjpython-demo"


def _official_mujoco_app_mjpython(env_root: Path) -> Path:
    site_packages = (
        env_root
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    return site_packages / "mujoco" / "MuJoCo_(mjpython).app" / "Contents" / "MacOS" / "mjpython"


def _official_mujoco_app(env_root: Path) -> Path:
    return _official_mujoco_app_mjpython(env_root).parents[2]


def _mujoco_package_dir() -> Path:
    spec = importlib.util.find_spec("mujoco")
    if spec is None or spec.origin is None:
        raise SystemExit("macOS MuJoCo demos require the project mujoco-uni package.")
    return Path(spec.origin).resolve().parent


def _mujoco_uni_mjpython_app() -> Path:
    return _mujoco_package_dir() / "MuJoCo_(mjpython).app"


def _ensure_official_mujoco_env(root: Path) -> Path:
    env_root = _official_mujoco_env(root)
    env_mjpython = env_root / "bin" / "mjpython"
    if env_mjpython.is_file() and _official_mujoco_app_mjpython(env_root).is_file():
        return env_root

    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit(
            "macOS MuJoCo demos require `uv` to create an isolated official MuJoCo "
            "mjpython environment."
        )

    env_root.parent.mkdir(parents=True, exist_ok=True)
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    subprocess.run([uv, "venv", str(env_root), "--python", python_version], check=True)
    subprocess.run(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(env_root / "bin" / "python"),
            _OFFICIAL_MUJOCO_PACKAGE,
        ],
        check=True,
    )
    if not env_mjpython.is_file() or not _official_mujoco_app_mjpython(env_root).is_file():
        raise SystemExit(
            "Failed to create the isolated official MuJoCo mjpython environment for "
            "macOS demo playback."
        )
    return env_root


def _ensure_mujoco_uni_mjpython_app(root: Path) -> None:
    dest = _mujoco_uni_mjpython_app()
    if (dest / "Contents" / "MacOS" / "mjpython").is_file():
        return

    official_env = _ensure_official_mujoco_env(root)
    shutil.copytree(_official_mujoco_app(official_env), dest, dirs_exist_ok=True)
    if not (dest / "Contents" / "MacOS" / "mjpython").is_file():
        raise SystemExit(f"Failed to materialize MuJoCo mjpython app at {dest}")


def _current_env_mjpython() -> str:
    if Path(sys.executable).name == "mjpython":
        return sys.executable

    venv_mjpython = Path(sys.executable).with_name("mjpython")
    if venv_mjpython.is_file():
        return str(venv_mjpython)

    mjpython = shutil.which("mjpython")
    if mjpython is not None:
        return mjpython

    raise SystemExit("macOS MuJoCo demos require `mjpython` in the active environment.")


def build_demo_command(
    *,
    demo_name: str,
    checkpoint_path: str,
    device: str | None = None,
    root: Path | None = None,
) -> list[str]:
    """Assemble the subprocess command for a given demo + resolved checkpoint."""
    from unilab.cli import build_command

    spec = get_demo_spec(demo_name)
    extra: list[str] = []
    if device is not None:
        extra.append(f"training.device={device}")

    if spec.entry == "eval":
        return build_command(
            mode="eval",
            algo=spec.algo,
            task=spec.task,
            sim=spec.sim,
            overrides=[f"algo.load_run={checkpoint_path}", *extra],
            root=root,
        )
    if spec.entry == "play_interactive":
        return _build_play_interactive_command(
            spec=spec, checkpoint_path=checkpoint_path, extra_overrides=extra, root=root
        )
    if spec.entry == "teaser":
        raise SystemExit(
            f"Demo {demo_name!r} is a renderer-only entry and has no subprocess command."
        )
    raise SystemExit(f"Unknown demo entry kind: {spec.entry!r}")


def _mxpython_executable() -> str:
    from unilab.cli import _mxpython_executable as resolve_mxpython_executable

    return resolve_mxpython_executable()


def _run_teaser_demo() -> int:
    if platform.system() == "Darwin" and Path(sys.executable).name != "mxpython":
        command = [
            _mxpython_executable(),
            str(_repo_root() / "src" / "unilab" / "tools" / "render_teaser.py"),
        ]
        env = os.environ.copy()
        env.setdefault("UV_PROJECT_ENVIRONMENT", str(_repo_root() / ".venv"))
        return subprocess.run(command, check=False, env=env).returncode

    from unilab.tools.render_teaser import main as render_teaser_main

    render_teaser_main()
    return 0


def run_demo(*, demo_name: str, refresh: bool = False, device: str | None = None) -> int:
    spec = get_demo_spec(demo_name)
    if spec.entry == "teaser":
        return _run_teaser_demo()
    if refresh and demo_name not in _LOCAL_ONLY_CHECKPOINT_DEMOS:
        _refresh_local_checkpoint(demo_name)
    elif refresh:
        print(f"Refresh ignored for local-only demo {demo_name!r}; no download source is used.")
    checkpoint_path = _resolve_demo_checkpoint(demo_name)
    if checkpoint_path is None:
        return 1

    command = build_demo_command(
        demo_name=demo_name, checkpoint_path=checkpoint_path, device=device
    )
    env = os.environ.copy()
    env.setdefault("UV_PROJECT_ENVIRONMENT", str(_repo_root() / ".venv"))
    returncode = subprocess.run(command, check=False, env=env).returncode
    if returncode == 0:
        print(f"Demo finished: {demo_name} (algo={spec.algo}, task={spec.task}, sim={spec.sim})")
    return returncode
