from __future__ import annotations

import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import hydra
import mujoco
import numpy as np
from omegaconf import DictConfig, OmegaConf

ROOT_DIR = Path(__file__).parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from unilab.base.backend.xml import materialize_scene_visual_override

from unilab.training import BackendAdapter, create_env, ensure_registries


def _coerce_str_list(value: Any, *, name: str) -> list[str]:
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a non-empty list.")
    out = [str(v) for v in value]
    if len(out) == 0:
        raise ValueError(f"{name} must be a non-empty list.")
    return out


def _coerce_int_list(value: Any, *, name: str) -> list[int]:
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a non-empty list.")
    out = [int(v) for v in value]
    if len(out) == 0:
        raise ValueError(f"{name} must be a non-empty list.")
    return out


def _env_cfg_override(cfg: DictConfig) -> dict[str, Any]:
    return BackendAdapter(
        cfg,
        root_dir=ROOT_DIR,
        algo_name="ppo",
        scene_materializer=materialize_scene_visual_override,
    ).build_task_env_cfg_override()


def _bench_ms(fn, *, warmup: int, repeat: int) -> list[float]:
    for _ in range(warmup):
        fn()
    out: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1000.0)
    return out


def _pct(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _serial_site_jacobian(
    backend: Any, site_id: int, dof_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    dof_indices = np.asarray(dof_indices, dtype=np.int32).reshape(-1)
    num_envs = int(backend._num_envs)
    jacp_out = np.zeros((num_envs, 3, len(dof_indices)), dtype=np.float64)
    jacr_out = np.zeros((num_envs, 3, len(dof_indices)), dtype=np.float64)
    state_snapshot = np.asarray(backend._physics_state, dtype=np.float64).copy()

    for env_idx in range(num_envs):
        variant_idx = int(backend._model_assignments[env_idx])
        model = backend._model_variants[variant_idx]
        data = mujoco.MjData(model)
        mujoco.mj_setState(
            model,
            data,
            state_snapshot[env_idx],
            int(mujoco.mjtState.mjSTATE_FULLPHYSICS),
        )
        mujoco.mj_forward(model, data)
        jacp_full = np.zeros((3, model.nv), dtype=np.float64)
        jacr_full = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jacSite(model, data, jacp_full, jacr_full, int(site_id))
        jacp_out[env_idx] = jacp_full[:, dof_indices]
        jacr_out[env_idx] = jacr_full[:, dof_indices]

    return jacp_out, jacr_out


def _print_stats(tag: str, ms_values: Sequence[float]) -> None:
    values = list(ms_values)
    print(
        f"{tag:>18}: "
        f"mean={statistics.mean(values):.3f} ms, "
        f"median={statistics.median(values):.3f} ms, "
        f"p95={_pct(values, 95):.3f} ms"
    )


@dataclass
class BenchResult:
    num_envs: int
    backend_threads: int
    correctness_ok: bool
    parallel_median_ms: float
    parallel_p95_ms: float
    serial_median_ms: float
    serial_p95_ms: float
    speedup: float
    step_median_ms: float | None
    jacobian_step_ratio_pct: float | None


def _format_markdown_table(results: Sequence[BenchResult]) -> str:
    lines = [
        "| num_envs | backend_threads | correctness | par_median_ms | par_p95_ms | ser_median_ms | ser_p95_ms | speedup(serial/par) | step_median_ms | jacobian_step_ratio_pct |",
        "|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        step_med = "-" if r.step_median_ms is None else f"{r.step_median_ms:.3f}"
        jac_ratio = "-" if r.jacobian_step_ratio_pct is None else f"{r.jacobian_step_ratio_pct:.2f}"
        lines.append(
            "| "
            f"{r.num_envs} | {r.backend_threads} | {'PASS' if r.correctness_ok else 'FAIL'} | "
            f"{r.parallel_median_ms:.3f} | {r.parallel_p95_ms:.3f} | "
            f"{r.serial_median_ms:.3f} | {r.serial_p95_ms:.3f} | "
            f"{r.speedup:.3f}x | {step_med} | {jac_ratio} |"
        )
    return "\n".join(lines)


def _run_one_case(
    cfg: DictConfig,
    *,
    num_envs: int,
    site_name: str,
    joint_names: list[str],
    jac_warmup: int,
    jac_repeat: int,
    step_warmup: int,
    step_repeat: int,
    atol: float,
    run_step_bench: bool,
) -> BenchResult:
    env = create_env(
        cfg,
        num_envs=num_envs,
        env_cfg_override=_env_cfg_override(cfg),
    )
    try:
        backend = getattr(env, "_backend", None)
        if backend is None:
            raise RuntimeError("Cannot access env backend from benchmark script.")
        if not hasattr(backend, "get_site_jacobian_w"):
            raise RuntimeError("Backend does not expose get_site_jacobian_w.")

        site_id = int(backend.get_site_ids([site_name])[0])
        dof_indices = backend.get_joint_dof_indices(joint_names)

        jacp_par, jacr_par = backend.get_site_jacobian_w(site_id, dof_indices)
        jacp_ser, jacr_ser = _serial_site_jacobian(backend, site_id, dof_indices)
        np.testing.assert_allclose(jacp_par, jacp_ser, atol=atol)
        np.testing.assert_allclose(jacr_par, jacr_ser, atol=atol)

        par_ms = _bench_ms(
            lambda: backend.get_site_jacobian_w(site_id, dof_indices),
            warmup=jac_warmup,
            repeat=jac_repeat,
        )
        ser_ms = _bench_ms(
            lambda: _serial_site_jacobian(backend, site_id, dof_indices),
            warmup=jac_warmup,
            repeat=jac_repeat,
        )

        par_median = statistics.median(par_ms)
        ser_median = statistics.median(ser_ms)
        speedup = ser_median / max(par_median, 1.0e-12)

        step_median_ms: float | None = None
        jacobian_step_ratio_pct: float | None = None
        if run_step_bench:
            action_space = getattr(env, "action_space")
            action_dim = int(action_space.shape[0])
            actions = np.random.uniform(-1.0, 1.0, size=(int(env.num_envs), action_dim)).astype(
                np.float64
            )

            env.init_state()
            for _ in range(step_warmup):
                env.step(actions)

            step_total_ms: list[float] = []
            for _ in range(step_repeat):
                state = env.step(actions)
                timing = state.info.get("timing", {})
                value = float(timing.get("env_step_total_ms", np.nan))
                if np.isfinite(value):
                    step_total_ms.append(value)

            if step_total_ms:
                step_median_ms = statistics.median(step_total_ms)
                jacobian_step_ratio_pct = 100.0 * par_median / max(step_median_ms, 1.0e-12)

        return BenchResult(
            num_envs=num_envs,
            backend_threads=int(getattr(backend, "_n_threads", -1)),
            correctness_ok=True,
            parallel_median_ms=par_median,
            parallel_p95_ms=_pct(par_ms, 95),
            serial_median_ms=ser_median,
            serial_p95_ms=_pct(ser_ms, 95),
            speedup=speedup,
            step_median_ms=step_median_ms,
            jacobian_step_ratio_pct=jacobian_step_ratio_pct,
        )
    finally:
        env.close()


@hydra.main(version_base="1.3", config_path="../conf/ppo", config_name="config")
def main(cfg: DictConfig) -> None:
    ensure_registries()
    if str(cfg.training.sim_backend) != "mujoco":
        raise ValueError("This benchmark requires task=.../mujoco.")

    # Probe once to infer default site/joint names.
    probe_env = create_env(
        cfg,
        num_envs=1,
        env_cfg_override=_env_cfg_override(cfg),
    )
    try:
        env_cfg = getattr(probe_env, "_cfg", None)
        asset_cfg = getattr(env_cfg, "asset", None)
        default_site_name = getattr(asset_cfg, "ee_site_name", None)
        default_joint_names = getattr(asset_cfg, "arm_joint_names", None)
    finally:
        probe_env.close()

    if default_site_name is None or default_joint_names is None:
        raise ValueError(
            "Cannot infer ee_site_name/arm_joint_names from env cfg. "
            "Please provide +bench.site_name and +bench.joint_names."
        )

    site_name = str(OmegaConf.select(cfg, "bench.site_name", default=default_site_name))
    joint_names = _coerce_str_list(
        OmegaConf.select(cfg, "bench.joint_names", default=list(default_joint_names)),
        name="bench.joint_names",
    )

    jac_warmup = int(OmegaConf.select(cfg, "bench.jac_warmup", default=20))
    jac_repeat = int(OmegaConf.select(cfg, "bench.jac_repeat", default=100))
    step_warmup = int(OmegaConf.select(cfg, "bench.step_warmup", default=10))
    step_repeat = int(OmegaConf.select(cfg, "bench.step_repeat", default=100))
    atol = float(OmegaConf.select(cfg, "bench.atol", default=1.0e-6))
    run_step_bench = bool(OmegaConf.select(cfg, "bench.run_step_bench", default=True))

    env_sweep_list = _coerce_int_list(
        OmegaConf.select(cfg, "bench.env_sweep", default=[4, 32, 128]),
        name="bench.env_sweep",
    )

    markdown_path_raw = OmegaConf.select(cfg, "bench.markdown_path", default="")
    markdown_path = str(markdown_path_raw).strip() if markdown_path_raw is not None else ""

    print("=== MuJoCo Site Jacobian Benchmark Sweep ===")
    print(f"task={cfg.training.task_name} backend={cfg.training.sim_backend}")
    print(f"site={site_name} joints={joint_names}")
    print(
        f"env_sweep={env_sweep_list} jac_warmup={jac_warmup} jac_repeat={jac_repeat} "
        f"step_warmup={step_warmup} step_repeat={step_repeat} atol={atol}"
    )

    results: list[BenchResult] = []
    for nenv in env_sweep_list:
        print(f"\n--- Running num_envs={nenv} ---")
        result = _run_one_case(
            cfg,
            num_envs=nenv,
            site_name=site_name,
            joint_names=joint_names,
            jac_warmup=jac_warmup,
            jac_repeat=jac_repeat,
            step_warmup=step_warmup,
            step_repeat=step_repeat,
            atol=atol,
            run_step_bench=run_step_bench,
        )
        results.append(result)
        print(
            "result: "
            f"threads={result.backend_threads}, "
            f"par_median={result.parallel_median_ms:.3f} ms, "
            f"ser_median={result.serial_median_ms:.3f} ms, "
            f"speedup={result.speedup:.3f}x"
        )
        if result.step_median_ms is not None and result.jacobian_step_ratio_pct is not None:
            print(
                f"        step_median={result.step_median_ms:.3f} ms, "
                f"jacobian/step={result.jacobian_step_ratio_pct:.2f}%"
            )

    markdown = _format_markdown_table(results)
    print("\n=== Markdown Table ===")
    print(markdown)

    if markdown_path:
        path = Path(markdown_path)
        if not path.is_absolute():
            path = ROOT_DIR / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown + "\n", encoding="utf-8")
        print(f"\nSaved markdown table to: {path}")


if __name__ == "__main__":
    main()
