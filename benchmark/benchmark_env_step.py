"""Benchmark env.step performance for a given task and backend.

Usage:
    # Run all combinations (go1/go2/g1/g1_motion_tracking x mujoco/motrix):
    uv run python benchmark/benchmark_env_step.py

    # Single task + backend:
    uv run python benchmark/benchmark_env_step.py task=g1_joystick/motrix

    # Override bench params:
    uv run python benchmark/benchmark_env_step.py task=go1_joystick/mujoco num_envs=4096 num_steps=500
"""

import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).parent.parent

TASK_CONFIGS = {
    "go1": "task=go1_joystick",
    "go2": "task=go2_joystick",
    "g1": "task=g1_joystick",
    "g1_mt": "task=g1_motion_tracking",
}

# Default benchmark parameters
DEFAULT_NUM_ENVS = 2048
DEFAULT_NUM_STEPS = 200
DEFAULT_WARMUP_STEPS = 10

BACKENDS = ["mujoco", "motrix"]


def _is_matrix_mode(argv: list[str]) -> bool:
    """Return True when user didn't specify task or backend explicitly."""
    for arg in argv:
        if arg.startswith("task=") or arg.startswith("training.sim_backend="):
            return False
    return True


def _parse_bench_args(args: list[str]) -> tuple[dict[str, str], list[str]]:
    """Parse benchmark-specific args and return (bench_kwargs, hydra_overrides).

    Benchmark args: num_envs=XXX, num_steps=XXX, warmup_steps=XXX
    These are not part of Hydra config, so we extract them separately.
    """
    bench_kwargs: dict[str, str] = {}
    hydra_overrides: list[str] = []

    for arg in args:
        if arg.startswith("num_envs="):
            bench_kwargs["num_envs"] = arg.split("=", 1)[1]
        elif arg.startswith("num_steps="):
            bench_kwargs["num_steps"] = arg.split("=", 1)[1]
        elif arg.startswith("warmup_steps="):
            bench_kwargs["warmup_steps"] = arg.split("=", 1)[1]
        else:
            hydra_overrides.append(arg)

    return bench_kwargs, hydra_overrides


def _compose_cfg(extra_args: list[str]):
    """Compose a Hydra config, handling GlobalHydra lifecycle."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    config_dir = str(ROOT_DIR / "conf" / "ppo")

    overrides = list(extra_args) + [
        "hydra.run.dir=.",
        "hydra.output_subdir=null",
        "hydra/job_logging=disabled",
        "hydra/hydra_logging=disabled",
    ]

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        return compose(config_name="config", overrides=overrides)


def _run_single(extra_args: list[str]) -> dict:
    """Run a single bench in-process via Hydra and return timing records."""
    from unilab.training import BackendAdapter, create_env, ensure_registries

    bench_kwargs, hydra_overrides = _parse_bench_args(extra_args)
    cfg = _compose_cfg(hydra_overrides)

    ensure_registries()

    num_envs = int(bench_kwargs.get("num_envs", DEFAULT_NUM_ENVS))
    num_steps = int(bench_kwargs.get("num_steps", DEFAULT_NUM_STEPS))
    warmup_steps = int(bench_kwargs.get("warmup_steps", DEFAULT_WARMUP_STEPS))

    task_name = cfg.training.task_name
    sim_backend = cfg.training.sim_backend

    adapter = BackendAdapter(cfg, root_dir=ROOT_DIR, algo_name="ppo")
    env_cfg_override = adapter.build_task_env_cfg_override()

    env = create_env(
        cfg,
        num_envs=num_envs,
        env_cfg_override=env_cfg_override,
        sim_backend=sim_backend,
        task_name=task_name,
    )

    nu = env._backend.num_actuators  # type: ignore[reportAttributeAccessIssue]
    env.init_state()

    for _ in range(warmup_steps):
        actions = np.random.uniform(-1, 1, size=(num_envs, nu)).astype(np.float32)
        env.step(actions)

    timing_records: dict[str, list[float]] = {}
    for _ in range(num_steps):
        actions = np.random.uniform(-1, 1, size=(num_envs, nu)).astype(np.float32)
        state = env.step(actions)
        timing = state.info.get("timing", {})
        for k, v in timing.items():
            timing_records.setdefault(k, []).append(v)

    env.close()

    return {
        "task_name": task_name,
        "sim_backend": sim_backend,
        "num_envs": num_envs,
        "num_steps": num_steps,
        "warmup_steps": warmup_steps,
        "timing_records": timing_records,
    }


def _print_single_report(result: dict) -> None:
    tr = result["timing_records"]
    total_arr = np.array(tr.get("env_step_total_ms", []))
    total_s = total_arr.sum() / 1000.0
    steps_per_env = result["num_steps"] * result["num_envs"]

    print(f"\n{'=' * 60}")
    print(f"  Task:       {result['task_name']}")
    print(f"  Backend:    {result['sim_backend']}")
    print(f"  Num envs:   {result['num_envs']}")
    print(f"  Steps:      {result['num_steps']} (warmup: {result['warmup_steps']})")
    print(f"{'=' * 60}")
    print(f"  Total time:       {total_s:.3f}s")
    print(f"  Mean step time:   {total_arr.mean():.3f}ms")
    print(f"  Median step time: {np.median(total_arr):.3f}ms")
    print(f"  Std step time:    {total_arr.std():.3f}ms")
    print(f"  Min step time:    {total_arr.min():.3f}ms")
    print(f"  Max step time:    {total_arr.max():.3f}ms")
    print(f"  Throughput:       {steps_per_env / total_s:.0f} env-steps/s")
    if tr:
        print(f"{'- ' * 30}")
        print("  Breakdown:")
        for k, v in tr.items():
            if k == "env_step_total_ms":
                continue
            arr = np.array(v)
            print(f"    {k:25s}  mean={arr.mean():.3f}ms  median={np.median(arr):.3f}ms")
    print(f"{'=' * 60}")


def _short_task_label(task_name: str) -> str:
    """Shorten 'Go1JoystickFlatTerrain' → 'go1'."""
    name = task_name.lower()
    if "motiontracking" in name:
        return "g1_mt"
    for prefix in ("go1", "go2", "g1"):
        if name.startswith(prefix):
            return prefix
    return task_name[:8]


def _print_comparison_table(results: list[dict]) -> None:
    rows_spec: list[tuple[str, str]] = [
        # (display_label, timing_key_or_special)
        ("total", "env_step_total_ms"),
        ("  apply_action", "apply_action_ms"),
        ("  step_core", "step_core_ms"),
        ("    set_ctrl", "backend_set_ctrl_ms"),
        ("    physics", "backend_physics_ms"),
        ("    refresh", "backend_refresh_cache_ms"),
        ("  update_state", "update_state_ms"),
        ("  reset_done", "reset_done_ms"),
        ("throughput", "__throughput__"),
    ]

    # Build short column labels
    col_labels = [
        f"{r.get('task_key', _short_task_label(r['task_name']))}/{r['sim_backend']}"
        for r in results
    ]

    metric_w = 16
    col_w = max(12, max(len(c) for c in col_labels) + 2)

    def hline(left: str, mid: str, right: str, fill: str = "─") -> str:
        return left + fill * metric_w + mid + mid.join(fill * col_w for _ in results) + right

    # Header
    print()
    print(hline("┌", "┬", "┐"))
    header = "│" + "metric".center(metric_w) + "│"
    header += "│".join(c.center(col_w) for c in col_labels) + "│"
    print(header)

    unit_row = "│" + "(median ms)".center(metric_w) + "│"
    unit_row += "│".join(" " * col_w for _ in results) + "│"
    print(unit_row)
    print(hline("├", "┼", "┤"))

    # Data rows
    for label, key in rows_spec:
        cells: list[str] = []
        if key == "__throughput__":
            for r in results:
                total_arr = np.array(r["timing_records"].get("env_step_total_ms", []))
                total_s = total_arr.sum() / 1000.0
                steps_per_env = r["num_steps"] * r["num_envs"]
                cells.append(f"{steps_per_env / total_s:,.0f}" if total_s > 0 else "-")
        else:
            for r in results:
                arr = r["timing_records"].get(key)
                cells.append(f"{np.median(arr):.3f}" if arr else "-")

        row = "│" + label.ljust(metric_w) + "│"
        row += "│".join(v.rjust(col_w - 1) + " " for v in cells) + "│"

        # Separator before throughput
        if key == "__throughput__":
            print(hline("├", "┼", "┤"))

        print(row)

    print(hline("└", "┴", "┘"))
    if results:
        r0 = results[0]
        print(f"  ({r0['num_envs']} envs, {r0['num_steps']} steps, throughput = env-steps/s)")


def _run_matrix(extra_args: list[str]) -> None:
    """Run all task x backend combinations and print comparison."""
    from unilab.base.backend.motrix_backend import MOTRIX_AVAILABLE

    backends = BACKENDS if MOTRIX_AVAILABLE else ["mujoco"]
    if not MOTRIX_AVAILABLE:
        print("Note: motrixsim not available, running mujoco only\n")

    results: list[dict] = []
    for task_key, task_override in TASK_CONFIGS.items():
        for backend in backends:
            label = f"{task_key}/{backend}"
            print(f"Running {label} ...", flush=True)
            try:
                # Task config format: task=go1_joystick/mujoco
                args = [f"{task_override}/{backend}"] + extra_args
                result = _run_single(args)
                result["task_key"] = task_key
                results.append(result)
                _print_single_report(result)
            except Exception as e:
                print(f"  FAILED: {e}\n")

    if len(results) > 1:
        _print_comparison_table(results)


def main() -> None:
    argv = sys.argv[1:]

    if _is_matrix_mode(argv):
        _run_matrix(argv)
    else:
        result = _run_single(argv)
        _print_single_report(result)


if __name__ == "__main__":
    main()
