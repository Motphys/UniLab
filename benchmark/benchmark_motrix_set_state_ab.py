"""Local A/B microbenchmark for Motrix ``set_state`` (issue #679).

Two variants run against the same ``MotrixBackend`` instance and the same
sequence of reset requests:

* ``baseline`` — the pre-#679 body: fresh ``np.zeros(num_envs, bool)`` mask each
  call, full ``np.array(qpos, copy=True)`` conversion, and helpers receive raw
  ``env_indices`` so they redo ``np.asarray(env_indices, dtype=np.intp)``
  internally, plus an unconditional ``np.ascontiguousarray`` on actuator ctrl.
* ``optimized`` — the current backend method: reusable scratch buffers, single
  hoisted ``env_ids_intp``, and a c-contiguous fast path for actuator ctrl.

Both variants instrument the same 16-key schema documented in
:mod:`unilab.base.np_env`. Only the internals differ.

The script does NOT touch the collector loop / physics — it only drives the
``set_state`` method — so it isolates the change under study.

Usage::

    uv run python benchmark/benchmark_motrix_set_state_ab.py \
        --num-envs 1024 --iters 30 --warmup 5
    uv run python benchmark/benchmark_motrix_set_state_ab.py --out benchmark/outputs/set_state_ab.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from unilab.assets import ASSETS_ROOT_PATH  # noqa: E402
from unilab.base.scene import SceneCfg  # noqa: E402
from unilab.dr.types import ResetRandomizationPayload  # noqa: E402

_G1_MODEL_FILE = str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml")

# Schema mirrors BACKEND_SET_STATE_DETAIL_TIMING_KEYS in np_env.py; keys that
# don't apply to motrix stay at 0.0.
_TIMING_KEYS = (
    "set_state_mask_ms",
    "set_state_data_slice_ms",
    "set_state_data_reset_ms",
    "set_state_clear_forces_ms",
    "set_state_geom_overrides_ms",
    "set_state_reset_rand_ms",
    "set_state_set_dof_vel_ms",
    "set_state_set_dof_pos_ms",
    "set_state_actuator_ctrl_ms",
    "set_state_forward_kinematic_ms",
    "set_state_refresh_pose_cache_ms",
    "set_state_invalidate_velocity_ms",
    "set_state_qpos_convert_ms",
    "set_state_pool_reset_ms",
    "set_state_state_scatter_ms",
    "set_state_internal_gap_ms",
)


def _baseline_set_state(
    self: Any,
    env_indices: np.ndarray,
    qpos: np.ndarray,
    qvel: np.ndarray,
    randomization: ResetRandomizationPayload | None = None,
) -> dict:
    """Inline copy of the pre-#679 body, monkey-patched onto the backend.

    Kept in the benchmark rather than the backend so the shipping code stays
    single-path. Uses only the helper kwargs that existed before the refactor
    (which are still accepted with defaults), so it stays a faithful baseline.
    """
    timing: dict[str, float] = {k: 0.0 for k in _TIMING_KEYS}
    outer_t0 = time.perf_counter()

    t0 = time.perf_counter()
    # NOTE: allocates a fresh (num_envs, qpos_dim) array every call.
    qpos_motrix = self._mujoco_qpos_to_motrix(qpos)
    timing["set_state_qpos_convert_ms"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    # NOTE: allocates a fresh bool mask every call.
    mask = np.zeros(self._num_envs, dtype=bool)
    mask[env_indices] = True
    timing["set_state_mask_ms"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    data_slice = self._data[mask]
    timing["set_state_data_slice_ms"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    data_slice.reset(self._model)
    timing["set_state_data_reset_ms"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    # NOTE: helper redoes np.asarray(env_indices, dtype=np.intp) internally
    # because we do NOT pass env_ids_intp.
    self._clear_applied_body_forces(env_indices)
    timing["set_state_clear_forces_ms"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    self._apply_init_geom_size_overrides(data_slice, env_indices)
    timing["set_state_geom_overrides_ms"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    self._apply_reset_randomization(data_slice, env_indices, randomization)
    timing["set_state_reset_rand_ms"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    data_slice.set_dof_vel(qvel)
    timing["set_state_set_dof_vel_ms"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    data_slice.set_dof_pos(qpos_motrix, self._model)
    timing["set_state_set_dof_pos_ms"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    if self._supports_position_actuator_gains and len(self._joint_dof_pos_indices) == int(
        self.num_actuators
    ):
        if self._joint_dof_pos_slice is not None:
            ctrl = qpos_motrix[:, self._joint_dof_pos_slice]
        else:
            ctrl = qpos_motrix[:, self._joint_dof_pos_indices]
    elif self._actuator_joint_pos_indices is not None:
        if self._actuator_joint_pos_slice is not None:
            ctrl = qpos_motrix[:, self._actuator_joint_pos_slice]
        else:
            ctrl = qpos_motrix[:, self._actuator_joint_pos_indices]
    else:
        ctrl = np.zeros((len(env_indices), self.num_actuators), dtype=self._np_dtype)
    # NOTE: unconditional ascontiguousarray copy.
    data_slice.actuator_ctrls = np.ascontiguousarray(ctrl)
    timing["set_state_actuator_ctrl_ms"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    self._model.forward_kinematic(data_slice)
    timing["set_state_forward_kinematic_ms"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    self._refresh_link_pose_cache(env_indices, data_slice=data_slice)
    timing["set_state_refresh_pose_cache_ms"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    self._invalidate_link_velocity_cache()
    timing["set_state_invalidate_velocity_ms"] = (time.perf_counter() - t0) * 1000.0

    outer_total_ms = (time.perf_counter() - outer_t0) * 1000.0
    measured_ms = sum(
        v
        for k, v in timing.items()
        if k
        not in (
            "set_state_pool_reset_ms",
            "set_state_state_scatter_ms",
            "set_state_internal_gap_ms",
        )
    )
    timing["set_state_internal_gap_ms"] = outer_total_ms - measured_ms
    return {"timing": timing}


def _make_backend(num_envs: int):
    from unilab.base.backend.motrix.backend import MotrixBackend

    return MotrixBackend(
        SceneCfg(model_file=_G1_MODEL_FILE),
        num_envs,
        sim_dt=0.005,
        base_name="pelvis",
    )


def _identity_qpos(nq: int, num_envs: int, xyz=(0.0, 0.0, 0.8)) -> np.ndarray:
    q = np.zeros((num_envs, nq), dtype=np.float32)
    q[:, :3] = xyz
    q[:, 3] = 1.0
    return q


def _run_variant(
    backend,
    *,
    variant: str,
    num_envs: int,
    reset_ratio: float,
    warmup: int,
    iters: int,
    rng: np.random.Generator,
    randomization: ResetRandomizationPayload | None,
) -> dict[str, Any]:
    """Drive set_state ``warmup+iters`` times; report per-key mean/median ms."""
    nq = backend.get_dof_pos().shape[-1] + 7
    nv = backend.get_dof_vel().shape[-1] + 6
    num_reset = max(1, int(num_envs * reset_ratio))

    samples: dict[str, list[float]] = {k: [] for k in _TIMING_KEYS}
    outer_samples: list[float] = []

    # New env_indices per call (mimics real reset_done pattern).
    def _sample_indices() -> np.ndarray:
        return rng.choice(num_envs, size=num_reset, replace=False).astype(np.int32)

    fn = (
        backend.set_state
        if variant == "optimized"
        else lambda *a, **kw: _baseline_set_state(backend, *a, **kw)
    )

    for step in range(warmup + iters):
        env_indices = _sample_indices()
        qpos = _identity_qpos(nq, num_reset)
        qvel = np.zeros((num_reset, nv), dtype=np.float32)
        t0 = time.perf_counter()
        result = fn(env_indices, qpos, qvel, randomization=randomization)
        outer_ms = (time.perf_counter() - t0) * 1000.0
        if step < warmup:
            continue
        outer_samples.append(outer_ms)
        assert isinstance(result, dict) and "timing" in result, "variant must return timing dict"
        for k in _TIMING_KEYS:
            samples[k].append(float(result["timing"].get(k, 0.0)))

    summary = {
        "variant": variant,
        "num_envs": num_envs,
        "num_reset": num_reset,
        "reset_ratio": reset_ratio,
        "warmup": warmup,
        "iters": iters,
        "outer_ms": {
            "mean": statistics.mean(outer_samples),
            "median": statistics.median(outer_samples),
            "min": min(outer_samples),
            "max": max(outer_samples),
        },
        "sub_ms": {
            k: {
                "mean": statistics.mean(v),
                "median": statistics.median(v),
            }
            for k, v in samples.items()
        },
    }
    return summary


def _format_ab_table(baseline: dict[str, Any], optimized: dict[str, Any]) -> str:
    """Render an ASCII A/B table with per-key mean ms and delta."""
    header = f"{'sub-timing':40s} {'baseline ms':>12s} {'optimized ms':>13s} {'delta ms':>10s} {'speedup':>9s}"
    rows = [header, "-" * len(header)]
    total_saved = 0.0
    for k in _TIMING_KEYS:
        b = baseline["sub_ms"][k]["mean"]
        o = optimized["sub_ms"][k]["mean"]
        delta = b - o
        total_saved += delta
        speedup = (b / o) if o > 1e-9 else float("inf")
        rows.append(f"{k:40s} {b:12.4f} {o:13.4f} {delta:10.4f} {speedup:9.2f}x")
    rows.append("-" * len(header))
    b_outer = baseline["outer_ms"]["mean"]
    o_outer = optimized["outer_ms"]["mean"]
    rows.append(
        f"{'outer set_state wall-clock':40s} {b_outer:12.4f} {o_outer:13.4f}"
        f" {b_outer - o_outer:10.4f} {(b_outer / o_outer) if o_outer > 1e-9 else float('inf'):9.2f}x"
    )
    rows.append(f"{'sum of sub-key deltas':40s} {'':12s} {'':13s} {total_saved:10.4f}")
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument(
        "--reset-ratios",
        type=str,
        default="0.05,0.25,0.50",
        help="Comma-separated reset ratios (fraction of envs reset per call).",
    )
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path for JSON output.",
    )
    parser.add_argument(
        "--randomization",
        action="store_true",
        help="Include a small DR payload (gravity) to exercise set_state_reset_rand_ms.",
    )
    args = parser.parse_args(argv)

    ratios = tuple(float(x) for x in args.reset_ratios.split(","))
    backend = _make_backend(args.num_envs)
    print(f"MotrixBackend ready: num_envs={backend.num_envs}, dt=0.005")

    output: dict[str, Any] = {
        "num_envs": args.num_envs,
        "iters": args.iters,
        "warmup": args.warmup,
        "results": [],
    }

    for ratio in ratios:
        num_reset = max(1, int(args.num_envs * ratio))
        payload = (
            ResetRandomizationPayload(
                gravity=np.tile(np.asarray([0.0, 0.0, -9.81], dtype=np.float32), (num_reset, 1)),
            )
            if args.randomization
            else None
        )
        rng_a = np.random.default_rng(args.seed)
        rng_b = np.random.default_rng(args.seed)
        baseline = _run_variant(
            backend,
            variant="baseline",
            num_envs=args.num_envs,
            reset_ratio=ratio,
            warmup=args.warmup,
            iters=args.iters,
            rng=rng_a,
            randomization=payload,
        )
        optimized = _run_variant(
            backend,
            variant="optimized",
            num_envs=args.num_envs,
            reset_ratio=ratio,
            warmup=args.warmup,
            iters=args.iters,
            rng=rng_b,
            randomization=payload,
        )
        print(f"\n=== reset_ratio={ratio:.2f} ({num_reset}/{args.num_envs} envs per set_state) ===")
        print(_format_ab_table(baseline, optimized))
        output["results"].append(
            {
                "reset_ratio": ratio,
                "num_reset": num_reset,
                "baseline": baseline,
                "optimized": optimized,
            }
        )

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"\nWrote JSON: {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
