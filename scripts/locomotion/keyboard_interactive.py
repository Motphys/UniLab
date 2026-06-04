"""MuJoCo-only keyboard interactive playback for locomotion policies.

Usage:
    uv run scripts/locomotion/keyboard_interactive.py --algo appo --task go2_joystick_flat --sim mujoco
    uv run scripts/locomotion/keyboard_interactive.py --algo flashsac --task g1_walk_flat --sim mujoco
    uv run scripts/locomotion/keyboard_interactive.py --algo ppo --task go2_joystick_flat --sim motrix \
      algo.load_run=2024-02-04_12-00-00

``--sim`` selects the task owner YAML to compose. Playback still runs through a
single MuJoCo environment and ``mujoco.viewer`` so the policy can be visualized
and controlled from the keyboard.
"""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
SCRIPTS_DIR = ROOT_DIR / "scripts"
LOCOMOTION_ENV_DIR = SRC_DIR / "unilab" / "envs" / "locomotion"
for candidate in (SRC_DIR, ROOT_DIR, SCRIPTS_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from play_interactive import (  # noqa: E402
    SUPPORTED_INTERACTIVE_ALGOS,
    _build_play_args,
    _compose_interactive_config,
    _override_key,
    play_interactive,
)

from unilab.base import registry  # noqa: E402
from unilab.training import (  # noqa: E402
    get_log_root,
    resolve_checkpoint_path,
    resolve_task_checkpoint_path,
)

_FORCED_OVERRIDE_KEYS = {
    "task",
    "training.sim_backend",
    "interactive.action_mode",
    "interactive.keyboard",
}
_VELOCITY_ARROW_HEIGHT = 0.6
_VELOCITY_ARROW_SCALE = 0.45
_VELOCITY_ARROW_WIDTH = 0.025


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Open mujoco.viewer for a locomotion policy with keyboard velocity commands."),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--algo",
        default="ppo",
        choices=SUPPORTED_INTERACTIVE_ALGOS,
        help="Algorithm config root to compose.",
    )
    parser.add_argument(
        "--task",
        required=True,
        help="Task owner name, for example go2_joystick_flat.",
    )
    parser.add_argument(
        "--sim",
        required=True,
        help="Owner YAML variant to read, for example mujoco or motrix.",
    )
    parser.add_argument(
        "--viz-arrow",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Show velocity arrows in the MuJoCo viewer.",
    )
    parser.add_argument(
        "--check-task-type",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Require a locomotion task whose policy obs contains velocity commands.",
    )
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Additional Hydra overrides, for example algo.load_run=<run>.",
    )
    args = parser.parse_args(argv)
    args.overrides = [str(item) for item in args.overrides if str(item) != "--"]
    return args


def _validate_owner_selector(task: str, sim: str) -> tuple[str, str]:
    task_name = str(task).strip()
    sim_name = str(sim).strip()
    if not task_name:
        raise SystemExit("--task must be non-empty.")
    if not sim_name:
        raise SystemExit("--sim must be non-empty.")
    if "/" in task_name:
        raise SystemExit("--task should not include '/<sim>'; pass the backend with --sim.")
    return task_name, sim_name


def _keyboard_overrides(*, task: str, sim: str, extra: list[str]) -> list[str]:
    task_name, sim_name = _validate_owner_selector(task, sim)
    normalized: list[str] = [f"task={task_name}/{sim_name}"]
    for override in extra:
        key = _override_key(override)
        if key in _FORCED_OVERRIDE_KEYS:
            raise SystemExit(
                f"{key} is controlled by keyboard_interactive.py; use --task/--sim instead."
            )
        normalized.append(override)
    normalized.extend(
        [
            "interactive.action_mode=policy",
            "interactive.keyboard=true",
        ]
    )
    return normalized


def _compose_keyboard_config(
    *,
    algo: str,
    task: str,
    sim: str,
    extra_overrides: list[str] | None = None,
) -> DictConfig:
    return _compose_interactive_config(
        algo,
        _keyboard_overrides(task=task, sim=sim, extra=list(extra_overrides or [])),
    )


def _registered_task_classes(task_name: str) -> list[type[Any]]:
    envs = getattr(registry, "_envs", {})
    meta = envs.get(task_name)
    if meta is None:
        known = ", ".join(sorted(envs.keys())[:20])
        suffix = f" Known tasks include: {known}" if known else ""
        raise SystemExit(f"Task is not registered: {task_name}.{suffix}")

    classes: list[type[Any]] = [meta.env_cfg_cls]
    classes.extend(meta.env_cls_dict.values())
    return classes


def _class_source_path(cls: type[Any]) -> Path | None:
    try:
        source = inspect.getsourcefile(cls)
    except TypeError:
        return None
    return Path(source).resolve() if source is not None else None


def _is_locomotion_task(task_name: str) -> bool:
    root = LOCOMOTION_ENV_DIR.resolve()
    for cls in _registered_task_classes(task_name):
        source_path = _class_source_path(cls)
        if source_path is None:
            continue
        if source_path == root or root in source_path.parents:
            return True
    return False


def _assert_locomotion_task(task_name: str) -> None:
    if _is_locomotion_task(task_name):
        return
    sources = [
        str(path.relative_to(ROOT_DIR))
        for cls in _registered_task_classes(task_name)
        if (path := _class_source_path(cls)) is not None and path.is_relative_to(ROOT_DIR)
    ]
    detail = f" Sources: {', '.join(sources)}" if sources else ""
    raise SystemExit(
        f"Task {task_name!r} is not implemented under src/unilab/envs/locomotion.{detail}"
    )


def _selected_checkpoint_value(cfg: DictConfig) -> str | None:
    value = OmegaConf.select(cfg, "algo.checkpoint", default=-1)
    if value in (None, "", -1, "-1", "None", "null"):
        return None
    return str(value)


def _resolve_model_checkpoint(cfg: DictConfig) -> tuple[Path | None, Path | None]:
    return resolve_task_checkpoint_path(
        ROOT_DIR,
        task_name=str(cfg.training.task_name),
        load_run=str(OmegaConf.select(cfg, "algo.load_run", default="-1")),
        algo_log_name=str(cfg.algo.algo_log_name),
        checkpoint=_selected_checkpoint_value(cfg),
        log_root=OmegaConf.select(cfg, "training.log_root", default=None),
    )


def _resolve_appo_checkpoint(cfg: DictConfig) -> tuple[Path | None, Path | None]:
    if _selected_checkpoint_value(cfg) is not None:
        return _resolve_model_checkpoint(cfg)
    return resolve_checkpoint_path(
        get_log_root(ROOT_DIR, cfg) / str(cfg.training.task_name),
        str(OmegaConf.select(cfg, "algo.load_run", default="-1")),
        suffix=".pt",
    )


def _resolve_sac_checkpoint(cfg: DictConfig) -> tuple[Path | None, Path | None]:
    return resolve_checkpoint_path(
        ROOT_DIR / "logs" / str(cfg.algo.algo_log_name) / str(cfg.training.task_name),
        str(OmegaConf.select(cfg, "algo.load_run", default="-1")),
        suffix=".pt",
    )


def _resolve_hora_distill_checkpoint(cfg: DictConfig) -> tuple[Path | None, Path | None]:
    from train_hora_distill import _resolve_stage2_checkpoint_path

    return _resolve_stage2_checkpoint_path(cfg)


def _resolve_required_checkpoint(cfg: DictConfig, *, algo: str) -> Path:
    if algo == "ppo":
        checkpoint_path, run_dir = _resolve_model_checkpoint(cfg)
    elif algo == "appo":
        checkpoint_path, run_dir = _resolve_appo_checkpoint(cfg)
    elif algo in {"sac", "flashsac"}:
        checkpoint_path, run_dir = _resolve_sac_checkpoint(cfg)
    elif algo == "hora_distill":
        checkpoint_path, run_dir = _resolve_hora_distill_checkpoint(cfg)
    else:
        raise SystemExit(f"Unsupported --algo={algo!r}")

    if checkpoint_path is not None:
        return Path(checkpoint_path)

    load_run = str(OmegaConf.select(cfg, "algo.load_run", default="-1"))
    if run_dir is None:
        raise SystemExit(
            "[keyboard_interactive] No run found for "
            f"algo={algo}, task={cfg.training.task_name}, algo.load_run={load_run!r}. "
            "Train a policy first or pass algo.load_run=<run-dir>."
        )
    raise SystemExit(
        "[keyboard_interactive] No checkpoint found under "
        f"{run_dir} for algo={algo}, task={cfg.training.task_name}. "
        "Train until a checkpoint is saved or pass algo.checkpoint=<iteration-or-filename>."
    )


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    cfg = _compose_keyboard_config(
        algo=str(args.algo),
        task=str(args.task),
        sim=str(args.sim),
        extra_overrides=list(args.overrides),
    )
    task_name = str(OmegaConf.select(cfg, "training.task_name"))
    if bool(args.check_task_type):
        _assert_locomotion_task(task_name)
    checkpoint_path = _resolve_required_checkpoint(cfg, algo=str(args.algo))

    print(
        "[keyboard_interactive] "
        f"Composed task={args.task}/{args.sim} "
        f"(training.task_name={task_name}, training.sim_backend={cfg.training.sim_backend})."
    )
    print(f"[keyboard_interactive] Checkpoint: {checkpoint_path}")
    print("[keyboard_interactive] Viewer backend: mujoco.viewer; controls: policy + keyboard.")
    play_args = _build_play_args(cfg, algo=str(args.algo))
    play_args.require_keyboard_command_obs = bool(args.check_task_type)
    play_args.show_velocity_arrows = bool(args.viz_arrow)
    play_args.velocity_arrow_height = _VELOCITY_ARROW_HEIGHT
    play_args.velocity_arrow_scale = _VELOCITY_ARROW_SCALE
    play_args.velocity_arrow_width = _VELOCITY_ARROW_WIDTH
    play_args.velocity_arrow_lateral_offset = 0.0
    play_interactive(play_args, cfg, algo=str(args.algo))


if __name__ == "__main__":
    main()
