"""Thin package CLI for routing to existing UniLab training entrypoints."""

from __future__ import annotations

import argparse
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Sequence

from unilab.demo import run_demo

SUPPORTED_ALGOS = ("ppo", "mlx_ppo", "appo", "sac", "td3", "flashsac")
SUPPORTED_SIMS = ("mujoco", "motrix")
OFFPOLICY_ALGOS = {"sac", "td3", "flashsac"}
RESERVED_OVERRIDE_KEYS = {
    "algo",
    "task",
    "training.sim_backend",
    "training.play_only",
}
TASK_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_-]*$")
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class Route:
    script_name: str
    config_group: str
    owner_task: str
    generated_overrides: tuple[str, ...]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _script_path(route: Route, root: Path) -> Path:
    return root / "scripts" / route.script_name


def _owner_yaml_path(route: Route, root: Path) -> Path:
    return root / "conf" / route.config_group / "task" / route.owner_task


def _check_private_checkout(root: Path) -> None:
    if not (root / "conf").is_dir() or not (root / "scripts").is_dir():
        raise SystemExit(
            "The current UniLab CLI expects a UniLab source checkout. "
            "Run it from the uv-managed editable environment created by this repo."
        )


def _check_reserved_overrides(overrides: Sequence[str]) -> None:
    reserved = [
        override for override in overrides if _override_key(override) in RESERVED_OVERRIDE_KEYS
    ]
    if reserved:
        joined = ", ".join(reserved)
        raise SystemExit(
            "Route-defining Hydra overrides must be provided through CLI flags, "
            f"not passthrough: {joined}"
        )


def _override_key(override: str) -> str:
    key = override.split("=", 1)[0].strip()
    return key.lstrip("+~")


def _check_task_name(task: str) -> None:
    if TASK_NAME_PATTERN.fullmatch(task) is None:
        raise SystemExit(
            "--task must be a registry task name such as `go1_joystick`; "
            "do not include slashes, dots, or path separators."
        )


def _check_profile(profile: str | None) -> None:
    if profile is None:
        return
    if TASK_NAME_PATTERN.fullmatch(profile) is None:
        raise SystemExit(
            "--profile must be a task owner variant such as `hora`; "
            "do not include slashes, dots, or path separators."
        )


def _check_load_run(load_run: str) -> None:
    if load_run == "-1":
        return
    if RUN_ID_PATTERN.fullmatch(load_run) is None or load_run in {".", ".."}:
        raise SystemExit("--load-run must be `-1` or a run directory name, not a path.")


def _check_runtime_requirements(algo: str, sim: str) -> None:
    if algo == "mlx_ppo" and platform.system() != "Darwin":
        raise SystemExit("mlx_ppo is only supported on macOS; use --algo ppo for torch PPO.")
    if sim == "motrix" and find_spec("motrixsim") is None:
        raise SystemExit(
            "sim=motrix requires the Motrix extra. Install it with `uv sync --extra motrix`."
        )


def _override_bool(overrides: Sequence[str], key: str) -> bool | None:
    selected: bool | None = None
    for override in overrides:
        if _override_key(override) != key or "=" not in override:
            continue
        value = override.split("=", 1)[1].strip().lower()
        if value in {"true", "1", "yes", "on"}:
            selected = True
        elif value in {"false", "0", "no", "off"}:
            selected = False
    return selected


def _needs_motrix_renderer(mode: str, sim: str, overrides: Sequence[str]) -> bool:
    if sim != "motrix":
        return False
    if mode == "eval":
        return True
    if mode == "train":
        return _override_bool(overrides, "training.no_play") is not True
    return False


def _python_executable_for_route(mode: str, sim: str, overrides: Sequence[str]) -> str:
    if platform.system() != "Darwin" or not _needs_motrix_renderer(mode, sim, overrides):
        return sys.executable

    if Path(sys.executable).name == "mxpython":
        return sys.executable

    mxpython = shutil.which("mxpython")
    if mxpython is None:
        raise SystemExit(
            "macOS Motrix playback uses the native renderer and must be launched with "
            "`mxpython`. Install the Motrix extra so `mxpython` is on PATH, or use "
            "`training.no_play=true` for non-rendering training."
        )
    return mxpython


def build_route(algo: str, task: str, sim: str, profile: str | None = None) -> Route:
    task_choice: str
    owner = f"{sim}_{profile}" if profile is not None else sim
    if algo in OFFPOLICY_ALGOS:
        task_choice = f"{algo}/{task}/{owner}"
        return Route(
            script_name="train_offpolicy.py",
            config_group="offpolicy",
            owner_task=f"{algo}/{task}/{owner}.yaml",
            generated_overrides=(f"algo={algo}", f"task={task_choice}"),
        )
    task_choice = f"{task}/{owner}"
    if algo == "ppo":
        return Route(
            script_name="train_rsl_rl.py",
            config_group="ppo",
            owner_task=f"{task}/{owner}.yaml",
            generated_overrides=(f"task={task_choice}",),
        )
    if algo == "mlx_ppo":
        return Route(
            script_name="train_mlx_ppo.py",
            config_group="ppo",
            owner_task=f"{task}/{owner}.yaml",
            generated_overrides=(f"task={task_choice}",),
        )
    if algo == "appo":
        return Route(
            script_name="train_appo.py",
            config_group="appo",
            owner_task=f"{task}/{owner}.yaml",
            generated_overrides=(f"task={task_choice}",),
        )
    raise SystemExit(f"Unsupported algo={algo!r}; choose one of: {', '.join(SUPPORTED_ALGOS)}")


def build_command(
    *,
    mode: str,
    algo: str,
    task: str,
    sim: str,
    overrides: Sequence[str],
    profile: str | None = None,
    load_run: str | None = None,
    root: Path | None = None,
) -> list[str]:
    selected_root = root or repo_root()
    _check_private_checkout(selected_root)
    _check_task_name(task)
    _check_profile(profile)
    _check_reserved_overrides(overrides)
    _check_runtime_requirements(algo, sim)

    route = build_route(algo, task, sim, profile)
    script = _script_path(route, selected_root)
    if not script.is_file():
        raise SystemExit(f"Entrypoint script not found: {script}")

    owner_yaml = _owner_yaml_path(route, selected_root)
    if not owner_yaml.is_file():
        raise SystemExit(
            f"No owner config exists for algo={algo}, task={task}, sim={sim}: {owner_yaml}"
        )

    generated = list(route.generated_overrides)
    if mode == "eval":
        generated.append("training.play_only=true")
        if load_run is not None:
            _check_load_run(load_run)
            if any(_override_key(o) == "algo.load_run" for o in overrides):
                raise SystemExit("Use either --load-run or algo.load_run=..., not both.")
            generated.append(f"algo.load_run={load_run}")

    executable = _python_executable_for_route(mode, sim, overrides)
    return [executable, str(script), *generated, *overrides]


def _train_eval_parser(*, mode: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=mode)
    parser.add_argument("--algo", required=True, choices=SUPPORTED_ALGOS)
    parser.add_argument("--task", required=True)
    parser.add_argument("--sim", required=True, choices=SUPPORTED_SIMS)
    parser.add_argument("--profile", default=None)
    if mode == "eval":
        parser.add_argument("--load-run", default=None)
    return parser


def _demo_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="demo")
    parser.add_argument("--preset", default="go2_joystick_mujoco_ppo")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--device", default=None)
    return parser


def _run_train_eval(mode: str, argv: Sequence[str] | None = None) -> int:
    parser = _train_eval_parser(mode=mode)
    args, overrides = parser.parse_known_args(argv)

    command = build_command(
        mode=mode,
        algo=args.algo,
        task=args.task,
        sim=args.sim,
        profile=args.profile,
        overrides=overrides,
        load_run=getattr(args, "load_run", None),
    )
    return subprocess.run(command, check=False).returncode


def train_main(argv: Sequence[str] | None = None) -> int:
    return _run_train_eval("train", argv)


def eval_main(argv: Sequence[str] | None = None) -> int:
    return _run_train_eval("eval", argv)


def demo_main(argv: Sequence[str] | None = None) -> int:
    parser = _demo_parser()
    args, overrides = parser.parse_known_args(argv)
    if overrides:
        raise SystemExit(
            f"demo does not accept passthrough Hydra overrides: {', '.join(overrides)}"
        )
    return run_demo(
        preset_name=args.preset,
        refresh=args.refresh,
        device=args.device,
    )


if __name__ == "__main__":
    raise SystemExit(train_main())
