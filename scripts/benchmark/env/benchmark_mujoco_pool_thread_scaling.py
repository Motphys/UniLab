"""MuJoCo BatchEnvPool thread-count scaling probe (issue #1328).

Steps a raw ``BatchEnvPool`` (no env semantics, no learner) on the G1 flat
scene with several ``nthread`` / ``cpu_ids`` configurations and reports, per
configuration, wall time per ``pool.step`` and the average number of cores the
process kept busy (process CPU time / wall time via ``os.times``).

Used to separate two effects of the default
``nthread = min(num_envs, 2 * cpu_count)`` pool sizing:

- thread count vs. pinning (``cpu_ids``): on the reference 16C/32T host the
  32-thread unpinned and pinned rows match, so the 2x-oversubscription loss
  comes from the thread count itself;
- the physics scaling ceiling: throughput saturates near the physical core
  count (memory-bandwidth bound), so extra threads mostly cost wall time.

Run:
    uv run scripts/benchmark/env/benchmark_mujoco_pool_thread_scaling.py

    # subset + tuning:
    uv run scripts/benchmark/env/benchmark_mujoco_pool_thread_scaling.py \
        --num-envs 4096 --nstep 3 --chunk-size 6 \
        --configs 64:unpinned,32:unpinned,32:pinned,16:pinned
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Sequence

import numpy as np

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
DEFAULT_MODEL = os.path.join(REPO_ROOT, "src/unilab/assets/robots/g1/scene_flat.xml")


def _cpu_time() -> float:
    t = os.times()
    return t.user + t.system


def build_state(model, nenvs: int) -> np.ndarray:
    """Tile the ``stand`` keyframe (or a plain forward) into a full-batch state."""
    import mujoco

    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    spec = int(mujoco.mjtState.mjSTATE_FULLPHYSICS)
    row = np.empty(mujoco.mj_stateSize(model, spec), dtype=np.float64)
    mujoco.mj_getState(model, data, row, spec)
    return np.tile(row, (nenvs, 1)).copy()


def bench_config(
    model,
    state0: np.ndarray,
    *,
    nthread: int,
    pinned: bool,
    nstep: int,
    chunk_size: int | None,
    warmup: int,
    iters: int,
) -> tuple[float, float]:
    """Return (wall ms/step, busy cores) for one pool configuration."""
    from mujoco_uni.batch_env import BatchEnvPool

    cpu_ids = list(range(nthread)) if pinned else None
    pool = BatchEnvPool(model, nbatch=state0.shape[0], nthread=nthread, cpu_ids=cpu_ids)
    nenvs = state0.shape[0]
    ctrl = np.zeros((nenvs, nstep, model.nu), dtype=np.float64)
    st = state0.copy()
    try:
        for _ in range(warmup):
            st = pool.step(st, nstep=nstep, control=ctrl, chunk_size=chunk_size)
        t0 = time.perf_counter()
        c0 = _cpu_time()
        for _ in range(iters):
            st = pool.step(st, nstep=nstep, control=ctrl, chunk_size=chunk_size)
        wall_ms = (time.perf_counter() - t0) / iters * 1000.0
        cores = (_cpu_time() - c0) / iters * 1000.0 / wall_ms
        return wall_ms, cores
    finally:
        pool.close()


def _parse_configs(spec: str) -> list[tuple[int, bool]]:
    out = []
    for item in spec.split(","):
        nthread_s, mode = item.strip().split(":")
        if mode not in ("pinned", "unpinned"):
            raise ValueError(f"unknown config mode {mode!r} in {item!r}")
        out.append((int(nthread_s), mode == "pinned"))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL, help="MuJoCo XML scene path")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--nstep", type=int, default=3, help="sim substeps per pool.step")
    parser.add_argument("--chunk-size", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument(
        "--configs",
        default="64:unpinned,32:unpinned,32:pinned,24:pinned,16:pinned,8:pinned",
        help="Comma-separated nthread:pinned|unpinned entries",
    )
    args = parser.parse_args(argv)

    import mujoco

    model = mujoco.MjModel.from_xml_path(args.model)
    state0 = build_state(model, args.num_envs)
    print(
        f"model={os.path.basename(args.model)} nu={model.nu} nv={model.nv} "
        f"nstate={state0.shape[1]} num_envs={args.num_envs} host_cpus={os.cpu_count()} "
        f"nstep={args.nstep} chunk_size={args.chunk_size}"
    )
    print(f"{'config':>18s} | {'ms/step':>8s} | {'cores':>6s}")
    for nthread, pinned in _parse_configs(args.configs):
        wall_ms, cores = bench_config(
            model,
            state0,
            nthread=nthread,
            pinned=pinned,
            nstep=args.nstep,
            chunk_size=args.chunk_size,
            warmup=args.warmup,
            iters=args.iters,
        )
        label = f"{nthread}t {'pinned' if pinned else 'unpinned'}"
        print(f"{label:>18s} | {wall_ms:8.2f} | {cores:6.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
