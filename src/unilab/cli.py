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

SUPPORTED_ALGOS = ("ppo", "appo", "sac", "td3", "flashsac")
SUPPORTED_SIMS = ("mujoco", "mjwarp", "motrix", "drake", "isaacgym", "genesis", "isaacsim")
SUPPORTED_RENDER_MODES = ("auto", "interactive", "record", "none")
OFFPOLICY_ALGOS = {"sac", "td3", "flashsac"}
# Built-in algos whose entrypoint script does not follow the train_<algo>.py
# naming convention.
SPECIAL_SCRIPT_NAMES = {"ppo": "train_rsl_rl.py", "appo": "train_appo.py"}
INTERACTIVE_PLAY_ALGOS = {"ppo", "appo", "sac", "td3", "flashsac"}
# Physics backends whose interactive eval runs through the dedicated MuJoCo
# viewer script: the selected backend owns the rollout while MuJoCo renders.
MUJOCO_VIEWER_PHYSICS_SIMS = frozenset({"mujoco", "mjwarp"})
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


def package_root() -> Path:
    return Path(__file__).resolve().parent


def _script_path(route: Route, root: Path) -> Path:
    return root / "scripts" / route.script_name


def _owner_yaml_path(route: Route, root: Path) -> Path:
    return root / "conf" / route.config_group / "task" / route.owner_task


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
    if sim == "mujoco" and find_spec("mujoco") is None:
        raise SystemExit(
            "sim=mujoco requires the MuJoCo extra. Install it with "
            "`pip install unilab[mujoco]` (or `uv sync --extra mujoco` in a source checkout)."
        )
    if sim == "mjwarp" and (find_spec("mujoco_warp") is None or find_spec("warp") is None):
        raise SystemExit(
            "sim=mjwarp requires the mjwarp extra. Install it with "
            "`pip install unilab[mjwarp]` (or `uv sync --extra mjwarp` in a source checkout)."
        )
    if sim == "motrix" and find_spec("motrixsim") is None:
        raise SystemExit(
            "sim=motrix requires the Motrix extra. Install it with "
            "`pip install unilab[motrix]` (or `uv sync --extra motrix` in a source checkout)."
        )
    if sim == "drake":
        if find_spec("drake_uni") is None:
            raise SystemExit(
                "sim=drake requires the Drake extra and a built DrakeUni batch extension. "
                "Run `make setup-drake` in a source checkout (or use the setup script directly)."
            )
        try:
            from drake_uni.runtime import batch_diagnostics

            diagnostics = batch_diagnostics()
        except Exception as exc:
            raise SystemExit(
                "sim=drake could not load the Drake batch extension. "
                "Set DRAKE_HOME and the platform library path (LD_LIBRARY_PATH on Linux, "
                "DYLD_LIBRARY_PATH on macOS), then rerun `make setup-drake`; details: "
                f"{exc}"
            ) from exc
        if not diagnostics.batch_available:
            detail = diagnostics.batch_import_error or "unknown import error"
            raise SystemExit(
                f"sim=drake requires a working Drake batch extension; diagnostic reported: {detail}"
            )
    if sim == "isaacgym":
        from unisim.backend.isaacgym.dependencies import isaacgym_runtime_available

        if not isaacgym_runtime_available():
            raise SystemExit(
                "sim=isaacgym requires the external Python 3.8 worker runtime. "
                "Install it with `scripts/tools/setup_isaacgym_env.sh` (see the "
                "IsaacGym backend docs page)."
            )
    if sim == "genesis":
        from unisim.backend.genesis.dependencies import genesis_dependencies_available

        if not genesis_dependencies_available():
            raise SystemExit(
                "sim=genesis requires the genesis-world extra (pinned 1.3.3, torch>=2.8). "
                "Install it with `pip install unilab[genesis]` (or `uv sync --extra genesis` "
                "in a source checkout; see the Genesis backend docs page)."
            )
    if sim == "isaacsim":
        from unisim.backend.isaacsim.dependencies import isaacsim_runtime_available

        if not isaacsim_runtime_available():
            raise SystemExit(
                "sim=isaacsim requires the external Python 3.11 IsaacSim/IsaacLab worker "
                "runtime. Install it with `scripts/tools/setup_isaacsim_env.sh` (see the "
                "IsaacSim backend docs page)."
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


def _override_value(overrides: Sequence[str], key: str) -> str | None:
    selected: str | None = None
    for override in overrides:
        if _override_key(override) != key or "=" not in override:
            continue
        selected = override.split("=", 1)[1].strip()
    return selected


def _needs_motrix_renderer(mode: str, sim: str, overrides: Sequence[str]) -> bool:
    if sim != "motrix":
        return False
    play_render_mode = _override_value(overrides, "training.play_render_mode")
    if play_render_mode is not None and play_render_mode.strip().lower() in {"none", "record"}:
        return False
    if mode == "eval":
        return True
    if mode == "train":
        return _override_bool(overrides, "training.no_play") is not True
    return False


def _python_executable_for_route(mode: str, sim: str, overrides: Sequence[str]) -> str:
    if platform.system() != "Darwin" or not _needs_motrix_renderer(mode, sim, overrides):
        return sys.executable

    return _mxpython_executable()


def _mxpython_executable() -> str:
    if Path(sys.executable).name == "mxpython":
        return sys.executable

    mxpython = shutil.which("mxpython")
    if mxpython is not None:
        return mxpython

    venv_mxpython = Path(sys.executable).with_name("mxpython")
    if venv_mxpython.is_file():
        return str(venv_mxpython)

    raise SystemExit(
        "macOS Motrix playback uses the native renderer and must be launched with "
        "`mxpython`. Install the Motrix extra so `mxpython` is on PATH, or use "
        "`training.no_play=true` for non-rendering training."
    )


def available_algos(root: Path | None = None) -> tuple[str, ...]:
    """Return routable algo names: built-ins plus convention-discovered ones.

    A custom algo ``X`` is routable when both ``conf/X/config.yaml`` and
    ``scripts/train_X.py`` exist under the package root. Config trees without
    an entrypoint script (e.g. ``hora_distill``) are not routable.
    """
    selected_root = root or package_root()
    discovered: list[str] = []
    conf_root = selected_root / "conf"
    if conf_root.is_dir():
        for child in sorted(conf_root.iterdir()):
            if not child.is_dir() or child.name in SUPPORTED_ALGOS:
                continue
            if not (child / "config.yaml").is_file():
                continue
            if (selected_root / "scripts" / f"train_{child.name}.py").is_file():
                discovered.append(child.name)
    return (*SUPPORTED_ALGOS, *discovered)


def build_route(
    algo: str, task: str, sim: str, profile: str | None = None, *, root: Path | None = None
) -> Route:
    owner = f"{sim}_{profile}" if profile is not None else sim
    task_choice = f"{task}/{owner}"
    if algo in OFFPOLICY_ALGOS:
        script_name = f"train_{algo}.py"
    elif algo in SPECIAL_SCRIPT_NAMES:
        script_name = SPECIAL_SCRIPT_NAMES[algo]
    else:
        selected_root = root or package_root()
        script_name = f"train_{algo}.py"
        routable = (
            TASK_NAME_PATTERN.fullmatch(algo) is not None
            and (selected_root / "conf" / algo / "config.yaml").is_file()
            and (selected_root / "scripts" / script_name).is_file()
        )
        if not routable:
            raise SystemExit(
                f"Unsupported algo={algo!r}; choose one of: "
                f"{', '.join(available_algos(selected_root))}"
            )
    return Route(
        script_name=script_name,
        config_group=algo,
        owner_task=f"{task}/{owner}.yaml",
        generated_overrides=(f"task={task_choice}",),
    )


def _eval_fallback_owner(route: Route, root: Path, *, sim: str, profile: str | None) -> str | None:
    """Pick a sibling backend owner for eval when the requested sim has no owner YAML.

    Eval replays a trained checkpoint, so any sibling owner of the same task (and
    profile shape) supplies the task/algo contract; the requested backend is
    re-applied through the sim2sim-allowlisted ``training.sim_backend`` override
    and validated by the runtime sim2sim preflight against the source run.
    """
    task_dir = _owner_yaml_path(route, root).parent
    for candidate_sim in SUPPORTED_SIMS:
        if candidate_sim == sim:
            continue
        owner = f"{candidate_sim}_{profile}" if profile is not None else candidate_sim
        if (task_dir / f"{owner}.yaml").is_file():
            return owner
    return None


def _uses_mujoco_interactive_play(
    *,
    mode: str,
    algo: str,
    sim: str,
    render_mode: str | None,
    overrides: Sequence[str],
) -> bool:
    """Return whether eval should use the dedicated MuJoCo viewer script."""
    if (
        mode != "eval"
        or sim not in MUJOCO_VIEWER_PHYSICS_SIMS
        or algo not in INTERACTIVE_PLAY_ALGOS
    ):
        return False
    selected_mode = _override_value(overrides, "training.play_render_mode") or render_mode
    return selected_mode is not None and selected_mode.strip().lower() == "interactive"


def build_command(
    *,
    mode: str,
    algo: str,
    task: str,
    sim: str,
    overrides: Sequence[str],
    profile: str | None = None,
    load_run: str | None = None,
    render_mode: str | None = None,
    root: Path | None = None,
) -> list[str]:
    selected_root = root or package_root()
    _check_task_name(task)
    _check_profile(profile)
    _check_reserved_overrides(overrides)
    _check_runtime_requirements(algo, sim)

    route = build_route(algo, task, sim, profile, root=selected_root)
    use_interactive_play = _uses_mujoco_interactive_play(
        mode=mode,
        algo=algo,
        sim=sim,
        render_mode=render_mode,
        overrides=overrides,
    )
    if use_interactive_play and find_spec("mujoco") is None:
        raise SystemExit(
            "interactive eval renders through the MuJoCo viewer and requires the MuJoCo "
            "extra. Install it with `pip install unilab[mujoco]` (or `uv sync --extra "
            "mujoco` in a source checkout)."
        )
    script = (
        selected_root / "scripts" / "play_interactive.py"
        if use_interactive_play
        else _script_path(route, selected_root)
    )
    if not script.is_file():
        raise SystemExit(f"Entrypoint script not found: {script}")

    owner = f"{sim}_{profile}" if profile is not None else sim
    sim_backend_override: str | None = None
    owner_yaml = _owner_yaml_path(route, selected_root)
    if not owner_yaml.is_file():
        if mode != "eval":
            raise SystemExit(
                f"No owner config exists for algo={algo}, task={task}, sim={sim}: {owner_yaml}"
            )
        fallback_owner = _eval_fallback_owner(route, selected_root, sim=sim, profile=profile)
        if fallback_owner is None:
            raise SystemExit(
                f"No owner config exists for algo={algo}, task={task}, sim={sim}: {owner_yaml}; "
                "eval fallback found no sibling backend owner config for this task either"
            )
        owner = fallback_owner
        route = Route(
            script_name=route.script_name,
            config_group=route.config_group,
            owner_task=f"{task}/{fallback_owner}.yaml",
            generated_overrides=(f"task={task}/{fallback_owner}",),
        )
        sim_backend_override = sim
        print(
            f"[eval] no owner config for sim={sim}; reusing sibling owner {fallback_owner!r} "
            f"with training.sim_backend={sim} (sim2sim contract check still applies)",
            file=sys.stderr,
        )

    generated = [] if use_interactive_play else list(route.generated_overrides)
    if sim_backend_override is not None:
        generated.append(f"training.sim_backend={sim_backend_override}")
    if render_mode is not None and _override_value(overrides, "training.play_render_mode") is None:
        generated.append(f"training.play_render_mode={render_mode}")
    if use_interactive_play and _override_value(overrides, "interactive.action_mode") is None:
        # The low-level viewer defaults to zero actions for debugging, while
        # eval must preserve the policy-control behavior of the train scripts.
        generated.append("interactive.action_mode=policy")
    if mode == "eval":
        generated.append("training.play_only=true")
        if load_run is not None:
            _check_load_run(load_run)
            if any(_override_key(o) == "algo.load_run" for o in overrides):
                raise SystemExit("Use either --load-run or algo.load_run=..., not both.")
            generated.append(f"algo.load_run={load_run}")

    executable = _python_executable_for_route(mode, sim, (*generated, *overrides))
    if use_interactive_play:
        return [
            executable,
            str(script),
            "--algo",
            algo,
            "--task",
            task,
            "--sim",
            owner,
            *generated,
            *overrides,
        ]
    return [executable, str(script), *generated, *overrides]


def _train_eval_parser(*, mode: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=mode)
    parser.add_argument(
        "--algo",
        required=True,
        metavar="ALGO",
        help=(
            "algorithm config tree under conf/; built-ins: "
            f"{', '.join(SUPPORTED_ALGOS)}. Custom algos are routable when "
            "conf/<algo>/config.yaml and scripts/train_<algo>.py both exist."
        ),
    )
    parser.add_argument("--task", required=True)
    parser.add_argument("--sim", required=True, choices=SUPPORTED_SIMS)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--render-mode", choices=SUPPORTED_RENDER_MODES, default=None)
    if mode == "eval":
        parser.add_argument("--load-run", default=None)
    return parser


def _demo_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="demo")
    parser.add_argument("demo_name")
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
        render_mode=args.render_mode,
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
        demo_name=args.demo_name,
        refresh=args.refresh,
        device=args.device,
    )


if __name__ == "__main__":
    raise SystemExit(train_main())
