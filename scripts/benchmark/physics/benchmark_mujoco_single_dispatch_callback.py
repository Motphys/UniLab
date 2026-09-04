#!/usr/bin/env python3
"""One-off microbenchmark for issue #1262.

Question: can one ``BatchEnvPool.step(nstep=N)`` dispatch reproduce the MBA
per-substep control semantics that today require N ``pool.step(nstep=1)``
dispatches (``MuJoCoBackend._step_with_pre_step_control``)?

Measured on the g1_walk_flat MuJoCo contract (scene_flat.xml + injected body
tracking sensors, sim_dt=1/150, sim_substeps=3):

- ``multi``:   current MBA path — N dispatches of nstep=1, with a Python-side
               per-substep control recompute (JointPositionAction-style:
               ``target = processed - encoder_bias``, constant within one
               control step) between dispatches.
- ``traj``:    single dispatch, nstep=N, control baked as an (nbatch, N, nu)
               trajectory with identical rows.
- ``const``:   single dispatch, nstep=N, control passed as one (nbatch, nu)
               array (native ``control_is_constant`` fast path).

All paths start from the same contact-rich state (standing keyframe settled
with foot contact) and use the same pool, model, sensors, and chunk size.
The numerical check compares final full-physics state and sensordata between
``multi`` and the single-dispatch paths.
"""

from __future__ import annotations

import argparse
import os
import time

import mujoco
import numpy as np
from mujoco_uni import BatchEnvPool
from unisim.backend.mujoco.xml import (
    create_discardvisual_xml,
    inject_mujoco_tracking_sensors,
)

SCENE_XML = "src/unilab/assets/robots/g1/scene_flat.xml"
BASE_BODY = "pelvis"
KEYFRAME = "stand"
SIM_DT = 1.0 / 150.0
SUBSTEPS = 3  # ctrl_dt 0.02 / sim_dt 0.006667, matches g1_walk_flat
CTRL_SPEC = int(mujoco.mjtState.mjSTATE_CTRL)
FULLPHYSICS = mujoco.mjtState.mjSTATE_FULLPHYSICS


def build_model() -> mujoco.MjModel:
    path = create_discardvisual_xml(SCENE_XML)
    path, _, _ = inject_mujoco_tracking_sensors(path, baselink_name=BASE_BODY)
    model = mujoco.MjModel.from_xml_path(path)
    model.opt.timestep = SIM_DT
    return model


def keyframe_state(model: mujoco.MjModel) -> np.ndarray:
    data = mujoco.MjData(model)
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, KEYFRAME)
    if kid < 0:
        raise ValueError(f"keyframe '{KEYFRAME}' not found in {SCENE_XML}")
    mujoco.mj_resetDataKeyframe(model, data, kid)
    mujoco.mj_forward(model, data)
    state = np.zeros(mujoco.mj_stateSize(model, FULLPHYSICS), dtype=np.float64)
    mujoco.mj_getState(model, data, state, FULLPHYSICS)
    return state


def make_ctrl(model: mujoco.MjModel, state0: np.ndarray, nbatch: int) -> np.ndarray:
    """Stand-pose position targets plus per-env jitter (deterministic)."""
    nq = model.nq
    qpos = state0[1 : 1 + nq]  # FULLPHYSICS: time, qpos, qvel, act, ...
    stand = qpos[-model.nu :]  # free root (7) precedes actuated joints
    rng = np.random.default_rng(0)
    ctrl = stand[None, :] + rng.uniform(-0.05, 0.05, size=(nbatch, model.nu))
    return np.ascontiguousarray(ctrl, dtype=np.float64)


def settle(pool: BatchEnvPool, state0: np.ndarray, ctrl: np.ndarray, chunk_size: int) -> np.ndarray:
    """Roll out a few control steps so feet are in steady contact."""
    state = state0
    for _ in range(40):
        state = pool.step(
            state,
            nstep=SUBSTEPS,
            control=ctrl,
            control_spec=CTRL_SPEC,
            chunk_size=chunk_size,
        )
    return state


def run_multi(pool, state, ctrl, bias, chunk_size):
    """Current MBA path: SUBSTEPS dispatches of nstep=1 with callback between."""
    pool_ms = 0.0
    callback_ms = 0.0
    for _ in range(SUBSTEPS):
        t0 = time.perf_counter()
        native_ctrl = np.subtract(ctrl, bias)  # JointPositionAction-style recompute
        callback_ms += (time.perf_counter() - t0) * 1e3
        t0 = time.perf_counter()
        state, _sensor = pool.step(
            state,
            nstep=1,
            control=native_ctrl[:, None, :],
            control_spec=CTRL_SPEC,
            chunk_size=chunk_size,
            return_sensor=True,
        )
        pool_ms += (time.perf_counter() - t0) * 1e3
    return state, pool_ms, callback_ms


def run_traj(pool, state, ctrl, bias, chunk_size):
    """Single dispatch, baked (nbatch, N, nu) trajectory with identical rows."""
    t0 = time.perf_counter()
    native_ctrl = np.subtract(ctrl, bias)
    callback_ms = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    traj = np.broadcast_to(native_ctrl[:, None, :], (ctrl.shape[0], SUBSTEPS, ctrl.shape[1]))
    state, _sensor = pool.step(
        state,
        nstep=SUBSTEPS,
        control=traj,
        control_spec=CTRL_SPEC,
        chunk_size=chunk_size,
        return_sensor=True,
    )
    return state, (time.perf_counter() - t0) * 1e3, callback_ms


def run_const(pool, state, ctrl, bias, chunk_size):
    """Single dispatch, baked (nbatch, nu) constant control (native fast path)."""
    t0 = time.perf_counter()
    native_ctrl = np.subtract(ctrl, bias)
    callback_ms = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    state, _sensor = pool.step(
        state,
        nstep=SUBSTEPS,
        control=native_ctrl,
        control_spec=CTRL_SPEC,
        chunk_size=chunk_size,
        return_sensor=True,
    )
    return state, (time.perf_counter() - t0) * 1e3, callback_ms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=8192)
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=[13, 41])
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    model = build_model()
    nthread = min(args.num_envs, (os.cpu_count() or 1) * 2)
    print(
        f"model: nu={model.nu} nq={model.nq} nv={model.nv} nsensor={model.nsensor} "
        f"nsensordata={model.nsensordata}; nbatch={args.num_envs} nthread={nthread}"
    )
    pool = BatchEnvPool(model, nbatch=args.num_envs, nthread=nthread)
    try:
        state0 = np.broadcast_to(keyframe_state(model)[None, :], (args.num_envs, pool.nstate))
        state0 = np.ascontiguousarray(state0, dtype=np.float64)
        ctrl = make_ctrl(model, state0[0], args.num_envs)
        bias = np.zeros_like(ctrl)  # encoder_bias placeholder (zeros on g1_walk_flat)
        contact_state = settle(pool, state0, ctrl, chunk_size=args.chunk_sizes[0])

        paths = {"multi": run_multi, "traj": run_traj, "const": run_const}

        # Numerical comparison from the same contact-rich state.
        finals = {}
        for name, fn in paths.items():
            out = fn(pool, contact_state, ctrl, bias, args.chunk_sizes[0])
            finals[name] = out[0]
        for name in ("traj", "const"):
            diff = np.abs(finals[name] - finals["multi"])
            bitwise = float(np.mean(finals[name] == finals["multi"]))
            print(
                f"numerics {name} vs multi: max_abs_diff={diff.max():.3e} "
                f"bitwise_equal={bitwise:.4f}"
            )
        print(
            f"numerics traj vs const: bitwise={bool(np.array_equal(finals['traj'], finals['const']))}"
        )

        # Timing.
        for chunk_size in args.chunk_sizes:
            print(
                f"--- chunk_size={chunk_size} (ms per control step, median of {args.repeats}) ---"
            )
            for name, fn in paths.items():
                for _ in range(args.warmup):
                    fn(pool, contact_state, ctrl, bias, chunk_size)
                pool_ts, cb_ts = [], []
                for _ in range(args.repeats):
                    _s, pool_ms, cb_ms = fn(pool, contact_state, ctrl, bias, chunk_size)
                    pool_ts.append(pool_ms)
                    cb_ts.append(cb_ms)
                print(
                    f"  {name:>5}: pool={np.median(pool_ts):7.2f}  "
                    f"callback={np.median(cb_ts):5.2f}  "
                    f"total={np.median(pool_ts) + np.median(cb_ts):7.2f}"
                )
    finally:
        pool.close()


if __name__ == "__main__":
    main()
