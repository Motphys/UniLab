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

SUPPORTED_ALGOS = ("ppo", "appo", "sac", "flashsac", "warpsac")
SUPPORTED_SIMS = (
    "mujoco",
    "mjwarp",
    "genesis",
    "newton",
)
# Adapters retained by UniSim but temporarily outside the tensor-only Manager
# runtime during issue #1811. They must remain unroutable through public CLI.
_SHELVED_SIMS = (
    "motrix",
    "drake",
    "isaacgym",
    "isaacsim",
    "superdex",
)
SUPPORTED_RENDER_MODES = ("auto", "interactive", "record", "viser", "none")
OFFPOLICY_ALGOS = {"sac", "flashsac", "warpsac"}
# Built-in algos whose entrypoint script does not follow the train_<algo>.py
# naming convention.
SPECIAL_SCRIPT_NAMES = {"ppo": "train_rsl_rl.py", "appo": "train_appo.py"}
INTERACTIVE_PLAY_ALGOS = {"ppo", "appo", "sac", "flashsac", "warpsac"}
# Physics backends whose interactive eval runs through the dedicated MuJoCo
# viewer script: the selected backend owns the rollout while MuJoCo renders.
MUJOCO_VIEWER_PHYSICS_SIMS = frozenset({"mujoco", "mjwarp"})
# Backends whose upstream unisim adapter does not implement the physics-state
# playback contract the viser viewer renders from (they expose native
# renderers instead). Supporting viser there requires upstream unisim
# capability work, so these sims fail closed with an actionable message.
VISER_UNSUPPORTED_SIMS = frozenset({"genesis", "isaacgym", "isaacsim"})
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
            "--task must be a registry task name such as `go2_joystick`; "
            "do not include slashes, dots, or path separators."
        )


def _check_profile(profile: str | None) -> None:
    if profile is None:
        return
    if TASK_NAME_PATTERN.fullmatch(profile) is None:
        raise SystemExit(
            "--profile must be a task owner variant such as `nodr`; "
            "do not include slashes, dots, or path separators."
        )


def _check_load_run(load_run: str) -> None:
    if load_run == "-1":
        return
    if RUN_ID_PATTERN.fullmatch(load_run) is None or load_run in {".", ".."}:
        raise SystemExit("--load-run must be `-1` or a run directory name, not a path.")


def _check_runtime_requirements(algo: str, sim: str) -> None:
    if sim not in SUPPORTED_SIMS:
        raise SystemExit(
            f"sim={sim} is temporarily outside the tensor-only Manager runtime "
            f"({', '.join(SUPPORTED_SIMS)}). Re-enabling it requires new UniSim "
            "capability, parity, and support-matrix evidence; see issue #1811."
        )
    # The MuJoCo physics backend (unisim.backend.mujoco.backend) needs the
    # mjbatch native batch engine; plain `mujoco` can also arrive via
    # other extras (e.g. superdex), so gate on `mjbatch` here.
    if sim == "mujoco" and (find_spec("mujoco") is None or find_spec("mjbatch") is None):
        raise SystemExit(
            "sim=mujoco requires the MuJoCo extra. Install it with "
            "`pip install unilab[mujoco]` (or `uv sync --extra mujoco` in a source checkout)."
        )
    if sim == "mjwarp" and (find_spec("mujoco_warp") is None or find_spec("warp") is None):
        raise SystemExit(
            "sim=mjwarp requires the mjwarp extra. Install it with "
            "`pip install unilab[mjwarp]` (or `uv sync --extra mjwarp` in a source checkout)."
        )
    del algo
    if sim == "genesis":
        from unisim.backend.genesis.dependencies import genesis_dependencies_available

        if not genesis_dependencies_available():
            raise SystemExit(
                "sim=genesis requires the genesis-world extra (pinned 1.3.3, torch>=2.8). "
                "Install it with `pip install unilab[genesis]` (or `uv sync --extra genesis` "
                "in a source checkout; see the Genesis backend docs page)."
            )
    if sim == "newton" and find_spec("newton") is None:
        raise SystemExit(
            "sim=newton requires the newton extra (pinned 1.5.1). Install it with "
            "`pip install unilab[newton]` (or `uv sync --extra newton` in a source "
            "checkout)."
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


def _python_executable_for_route(mode: str, sim: str, overrides: Sequence[str]) -> str:
    del mode, sim, overrides
    if platform.system() != "Darwin":
        return sys.executable
    return sys.executable


def _mujoco_package_dir() -> Path:
    spec = find_spec("mujoco")
    if spec is None or spec.origin is None:
        raise SystemExit("macOS MuJoCo interactive rendering requires the official mujoco package.")
    return Path(spec.origin).resolve().parent


def _mujoco_mjpython_app() -> Path:
    return _mujoco_package_dir() / "MuJoCo_(mjpython).app"


def _ensure_mujoco_mjpython_app() -> None:
    app = _mujoco_mjpython_app()
    if (app / "Contents" / "MacOS" / "mjpython").is_file():
        return
    raise SystemExit(
        "macOS MuJoCo interactive rendering requires the MuJoCo_(mjpython).app "
        f"bundled with the official mujoco wheel, but it is missing at {app}."
    )


def _mjpython_executable() -> str:
    if Path(sys.executable).name == "mjpython":
        return sys.executable

    venv_mjpython = Path(sys.executable).with_name("mjpython")
    if venv_mjpython.is_file():
        return str(venv_mjpython)

    mjpython = shutil.which("mjpython")
    if mjpython is not None:
        return mjpython

    raise SystemExit(
        "macOS MuJoCo interactive rendering must run under `mjpython` (the Cocoa "
        "launcher bundled with the mujoco wheel). Install the MuJoCo extra so "
        "`mjpython` is on PATH, or use `--render-mode record` for offscreen rendering."
    )


def _interactive_mujoco_executable() -> str:
    """Resolve the macOS interpreter for the MuJoCo viewer route.

    The glfw-based MuJoCo viewer requires `mjpython` so Cocoa owns the main
    thread; fail closed with an install hint when it is unavailable.
    """
    _ensure_mujoco_mjpython_app()
    return _mjpython_executable()


def available_algos(root: Path | None = None) -> tuple[str, ...]:
    """Return routable algo names: built-ins plus convention-discovered ones.

    A custom algo ``X`` is routable when both ``conf/X/config.yaml`` and
    ``scripts/train_X.py`` exist under the package root. Config trees without
    an entrypoint script are not routable.
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


def _uses_viser_play(
    *,
    mode: str,
    algo: str,
    sim: str,
    render_mode: str | None,
    overrides: Sequence[str],
) -> bool:
    """Return whether eval should use the browser-based viser viewer script.

    Any physics owner whose backend declares physics-state playback is
    routable; :data:`VISER_UNSUPPORTED_SIMS` sims are rejected earlier with a
    targeted message, and the viewer itself fails closed on the runtime
    capability check.
    """
    del sim  # routing is sim-agnostic; capability gating happens at runtime
    if mode != "eval" or algo not in INTERACTIVE_PLAY_ALGOS:
        return False
    selected_mode = _override_value(overrides, "training.play_render_mode") or render_mode
    return selected_mode is not None and selected_mode.strip().lower() == "viser"


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
    use_viser_play = _uses_viser_play(
        mode=mode,
        algo=algo,
        sim=sim,
        render_mode=render_mode,
        overrides=overrides,
    )
    selected_mode = _override_value(overrides, "training.play_render_mode") or render_mode
    viser_mode_requested = selected_mode is not None and selected_mode.strip().lower() == "viser"
    if viser_mode_requested and sim in VISER_UNSUPPORTED_SIMS:
        raise SystemExit(
            f"render mode 'viser' renders through the physics-state playback contract, "
            f"which the {sim} backend does not implement in upstream unisim; "
            "use --render-mode interactive or record for its native renderer."
        )
    if viser_mode_requested and mode == "eval" and not use_viser_play:
        raise SystemExit(
            f"render mode 'viser' eval requires one of the interactive play algos "
            f"({', '.join(sorted(INTERACTIVE_PLAY_ALGOS))}); got algo={algo}."
        )
    if use_interactive_play and find_spec("mujoco") is None:
        raise SystemExit(
            "interactive eval renders through the MuJoCo viewer and requires the MuJoCo "
            "extra. Install it with `pip install unilab[mujoco]` (or `uv sync --extra "
            "mujoco` in a source checkout)."
        )
    if viser_mode_requested and find_spec("mujoco") is None:
        raise SystemExit(
            "viser playback renders MuJoCo playback models and requires the MuJoCo "
            "package. Install it with `pip install unilab[mujoco]` (or `uv sync "
            "--extra mujoco` in a source checkout)."
        )
    viewer_script_name: str | None = None
    if use_interactive_play:
        viewer_script_name = "play_interactive.py"
    elif use_viser_play:
        viewer_script_name = "play_viser.py"
    script = (
        selected_root / "scripts" / viewer_script_name
        if viewer_script_name is not None
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

    use_viewer_play = use_interactive_play or use_viser_play
    generated = [] if use_viewer_play else list(route.generated_overrides)
    if sim_backend_override is not None:
        generated.append(f"training.sim_backend={sim_backend_override}")
    if render_mode is not None and _override_value(overrides, "training.play_render_mode") is None:
        generated.append(f"training.play_render_mode={render_mode}")
    if (
        mode == "eval"
        and sim == "superdex"
        and selected_mode is not None
        and selected_mode.strip().lower() == "interactive"
        and _override_value(overrides, "training.play_env_num") is None
    ):
        # Native superdex interactive rendering draws exactly one scene; the
        # owner layer also switches the env to the serial executor.
        generated.append("training.play_env_num=1")
    if use_viewer_play and _override_value(overrides, "interactive.action_mode") is None:
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
    if use_interactive_play and platform.system() == "Darwin":
        executable = _interactive_mujoco_executable()
    if use_viewer_play:
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
    try:
        return subprocess.run(command, check=False).returncode
    except KeyboardInterrupt:
        # Ctrl+C is delivered to both this routing process and the child
        # playback/training process.  The child owns its renderer cleanup;
        # keep the wrapper quiet and use the conventional shell exit code.
        return 130


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
