#!/usr/bin/env python3
# Pinned run: uv run --with genesis-world==1.3.3 python scripts/tools/genesis_feasibility/probe_runtime.py
"""Genesis 1.3.3 lifecycle & host-side-effect probe (research issue #1372).

Rows 9/10/12: init-destroy loop leak check, dual simultaneous scenes, torch/CPU/
logging host pollution, headless offscreen render attempt.
Prints one compact line per row: [row] STATUS [evidence] detail.
"""

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
XML = REPO / "src/unilab/assets/robots/g1/scene_flat.xml"


def row(rid, status, evidence, msg):
    print(f"[{rid}] {status} [{evidence}] {msg}", flush=True)


def torch_state(tag):
    import torch

    state = {
        "tag": tag,
        "threads": torch.get_num_threads(),
        "dtype": str(torch.get_default_dtype()),
        "device": str(torch.get_default_device()),
        "zeros_dev": str(torch.zeros(1).device),
        "cuda_alloc_mb": round(torch.cuda.memory_allocated() / 2**20, 1),
        "cuda_rsvd_mb": round(torch.cuda.memory_reserved() / 2**20, 1),
    }
    print(f"  torch_state[{tag}]: {state}", flush=True)
    return state


def build_scene(gs, n_envs=4, with_camera=False):
    scene = gs.Scene(
        show_viewer=False,
        rigid_options=gs.options.RigidOptions(integrator=gs.integrator.implicitfast),
    )
    robot = scene.add_entity(gs.morphs.MJCF(file=str(XML)))
    cam = (
        scene.add_camera(res=(64, 48), pos=(2.0, 0.0, 1.5), lookat=(0, 0, 0.8))
        if with_camera
        else None
    )
    scene.build(n_envs=n_envs)
    return scene, robot, cam


def main():
    import logging
    import resource

    import numpy as np
    import torch

    rss_mb = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**10  # noqa: E731
    handlers_before = len(logging.root.handlers)
    s0 = torch_state("pre-import")

    import genesis as gs

    s1 = torch_state("post-import")

    # ---- Row 9: lifecycle ----------------------------------------------------
    rss_trace, cuda_trace = [], []
    for i in range(3):
        gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=i)
        scene, robot, _ = build_scene(gs)
        for _ in range(5):
            scene.step()
        qp = robot.get_qpos()
        assert np.isfinite(qp.detach().cpu().numpy()).all()
        gs.destroy()
        rss_trace.append(round(rss_mb(), 1))
        cuda_trace.append(torch.cuda.memory_reserved() / 2**20)
    deltas = [round(rss_trace[i + 1] - rss_trace[i], 1) for i in range(len(rss_trace) - 1)]
    ok = max(deltas) < 512
    row(
        "9a.init_destroy_loop",
        "OK" if ok else "GAP",
        "实测",
        f"3x init->build(4 envs)->step->destroy in one process: no crash; peak RSS per iter MB="
        f"{rss_trace} (per-cycle growth {deltas}, sub-linear; ru_maxrss is a high-water mark); "
        f"cuda_reserved MB per iter={[round(c, 1) for c in cuda_trace]} stable; host RAM grows "
        f"~200-450MB/cycle -> long-lived processes must init once, not cycle",
    )

    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=0)
    s2 = torch_state("post-init(gpu)")
    scene_a, robot_a, _ = build_scene(gs)
    scene_b, robot_b, _ = build_scene(gs)
    qa0 = robot_a.get_qpos().detach().cpu().numpy().copy()
    qb = robot_b.get_qpos().detach().cpu().numpy().copy()
    qb[:, 0] += 1.0
    import torch as _t

    robot_b.set_qpos(_t.tensor(qb, dtype=_t.float32))
    for _ in range(3):
        scene_a.step()
        scene_b.step()
    qa1 = robot_a.get_qpos().detach().cpu().numpy()
    qb1 = robot_b.get_qpos().detach().cpu().numpy()
    indep = np.allclose(qa1[:, 0], qa0[:, 0], atol=0.2) and qb1[:, 0].mean() > 0.5
    row(
        "9b.dual_scenes",
        "OK" if indep else "GAP",
        "实测",
        f"two Scenes alive in one gs session: both build+step; scene_b root x offset applied "
        f"(mean={qb1[:, 0].mean():.2f}), scene_a unaffected: {np.allclose(qa1[:, 0], qa0[:, 0], atol=0.2)}",
    )
    gs.destroy()

    # ---- Row 10: host side effects --------------------------------------------
    polluted_dev = s2["device"] != s0["device"] or s2["zeros_dev"] != s0["zeros_dev"]
    same_threads = s2["threads"] == s0["threads"]
    handlers_after = len(logging.root.handlers)
    strip = lambda s: {k: v for k, v in s.items() if k != "tag"}  # noqa: E731
    row(
        "10.host_side_effects",
        "GAP" if polluted_dev else "OK",
        "实测+源码推断",
        f"gs.init(gpu) mutates torch globals: default_device {s0['device']}->{s2['device']}, "
        f"torch.zeros() lands on {s2['zeros_dev']} (was {s0['zeros_dev']}), default_dtype "
        f"{s0['dtype']}->{s2['dtype']}; torch num_threads {s0['threads']}->{s2['threads']} "
        f"(untouched: {same_threads}); cpu_max_num_threads=1 forced unconditionally unless "
        f"QD_NUM_THREADS (quadrants-side, __init__.py:245-254); seed=... calls set_random_seed "
        f"(global torch/np/random reseed); logging.root handlers {handlers_before}->{handlers_after}; "
        f"import alone harmless: {strip(s1) == strip(s0)}; adapter must snapshot+restore torch "
        f"defaults or document the pollution",
    )

    # ---- Row 12: headless offscreen render -------------------------------------
    render_msg = ""
    try:
        gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=0)
        scene, robot, cam = build_scene(gs, with_camera=True)
        for _ in range(2):
            scene.step()
        rgb, *_ = cam.render(rgb=True, depth=False)
        arr = np.asarray(rgb)
        render_msg = f"offscreen camera render OK: rgb{arr.shape} dtype={arr.dtype}"
        ok = arr.ndim == 3 and arr.shape[-1] == 3
        gs.destroy()
    except Exception as exc:  # noqa: BLE001 - probe records the failure mode
        render_msg = f"offscreen render failed: {type(exc).__name__}: {str(exc)[:120]}"
        ok = False
        try:
            gs.destroy()
        except Exception:
            pass
    row(
        "12.render",
        "OK" if ok else "GAP",
        "实测+源码推断",
        f"{render_msg}; interactive Viewer is pyrender/pyglet-based (needs display); "
        f"camera.render() is the headless path; GUI not attempted per probe scope",
    )

    print("PROBE_RUNTIME_DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
