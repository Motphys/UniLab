"""Demo entrypoint: fetch checkpoint from HF, then launch interactive playback."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from unilab.assets import ASSETS_ROOT_PATH
from unilab.assets.hub import resolve_checkpoint_file


@dataclass(frozen=True)
class DemoSpec:
    algo: str
    task: str
    sim: str
    entry: str  # "eval" or "play_interactive"


DEMO_REGISTRY: dict[str, DemoSpec] = {
    "dance": DemoSpec(algo="ppo", task="g1_motion_tracking", sim="motrix", entry="eval"),
    "wallflip": DemoSpec(algo="ppo", task="g1_wall_flip_tracking", sim="motrix", entry="eval"),
    "boxtracking": DemoSpec(algo="ppo", task="g1_box_tracking", sim="motrix", entry="eval"),
    "locomani": DemoSpec(
        algo="ppo", task="go2_arm_manip_loco", sim="mujoco", entry="play_interactive"
    ),
    "inhandgrasp": DemoSpec(algo="ppo", task="sharpa_inhand", sim="motrix", entry="eval"),
}


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


def _refresh_local_checkpoint(demo_name: str) -> None:
    local = ASSETS_ROOT_PATH / _checkpoint_relative_path(demo_name)
    if local.exists():
        local.unlink()


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
    owner_yaml = selected_root / "conf" / spec.algo / "task" / spec.task / f"{spec.sim}.yaml"
    if not owner_yaml.is_file():
        raise SystemExit(
            f"No owner config exists for algo={spec.algo}, task={spec.task}, sim={spec.sim}: "
            f"{owner_yaml}"
        )
    return [
        sys.executable,
        str(script),
        f"task={spec.task}/{spec.sim}",
        f"algo.load_run={checkpoint_path}",
        *extra_overrides,
    ]


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
    raise SystemExit(f"Unknown demo entry kind: {spec.entry!r}")


def run_demo(*, demo_name: str, refresh: bool = False, device: str | None = None) -> int:
    spec = get_demo_spec(demo_name)
    if refresh:
        _refresh_local_checkpoint(demo_name)
    checkpoint_path = resolve_checkpoint_file(_checkpoint_relative_path(demo_name))
    assert isinstance(checkpoint_path, str)

    command = build_demo_command(
        demo_name=demo_name, checkpoint_path=checkpoint_path, device=device
    )
    env = os.environ.copy()
    env.setdefault("UV_PROJECT_ENVIRONMENT", str(_repo_root() / ".venv"))
    returncode = subprocess.run(command, check=False, env=env).returncode
    if returncode == 0:
        print(f"Demo finished: {demo_name} (algo={spec.algo}, task={spec.task}, sim={spec.sim})")
    return returncode
