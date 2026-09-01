#!/usr/bin/env python3
# Pinned run: uv run --with genesis-world==1.3.3 python scripts/tools/genesis_feasibility/probe_perf.py
"""Genesis 1.3.3 NumPy boundary cost probe (research issue #1372), row 11.

At n_envs in {256, 2048, 4096}: (a) scene.step() alone, (b) step + D2H pull of
the state slices a SimBackend needs, (c) step + H2D ctrl push each step.
Sizes the mjwarp-style host-cache strategy for the Genesis adapter.
"""

import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
XML = REPO / "src/unilab/assets/robots/g1/scene_flat.xml"
ENVS = (256, 2048, 4096)
STEPS, WARMUP = 30, 5


def row(rid, status, evidence, msg):
    print(f"[{rid}] {status} [{evidence}] {msg}", flush=True)


def main():
    import genesis as gs
    import mujoco
    import torch

    m = mujoco.MjModel.from_xml_path(str(XML))
    key_qpos = m.key_qpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "stand")].copy()

    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=0)

    def timed(fn, n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n * 1000.0  # ms per call

    for n_envs in ENVS:
        scene = gs.Scene(
            show_viewer=False,
            rigid_options=gs.options.RigidOptions(integrator=gs.integrator.implicitfast),
        )
        robot = scene.add_entity(gs.morphs.MJCF(file=str(XML)))
        scene.build(n_envs=n_envs)
        act_joints = [j for j in robot.joints if j.n_dofs == 1]
        act_dofs = [j.dofs_idx_local[0] for j in act_joints]
        stand = np.array([key_qpos[j.qs_idx_local[0]] for j in act_joints], dtype=np.float32)
        ctrl_np = np.tile(stand, (n_envs, 1))
        link_ids = [
            robot.get_link(n).idx_local
            for n in (
                "pelvis",
                "torso_link",
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_knee_link",
                "right_knee_link",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            )
        ]

        robot.set_qpos(np.tile(key_qpos, (n_envs, 1)))
        robot.zero_all_dofs_velocity()
        robot.control_dofs_position(torch.from_numpy(ctrl_np).cuda(), dofs_idx_local=act_dofs)

        def pull_state():
            slices = (
                robot.get_qpos(),
                robot.get_dofs_velocity(),
                robot.get_pos(),
                robot.get_quat(),
                robot.get_vel(),
                robot.get_ang(),
                robot.get_links_pos(link_ids),
                robot.get_links_quat(link_ids),
            )
            return [s.detach().cpu().numpy() for s in slices]

        def step_a():
            scene.step()

        def step_b():
            scene.step()
            pull_state()

        def step_c():
            robot.control_dofs_position(torch.from_numpy(ctrl_np).cuda(), dofs_idx_local=act_dofs)
            scene.step()

        for fn in (step_a, step_b, step_c):
            timed(fn, WARMUP)
        # interleave a/b/c across rounds to cancel clock/warmup ordering bias
        samples = {"a": [], "b": [], "c": []}
        fns = {"a": step_a, "b": step_b, "c": step_c}
        order = ["a", "b", "c"]
        for _ in range(STEPS):
            for k in order:
                samples[k].append(timed(fns[k], 1))
            order = order[1:] + order[:1]
        ta, tb, tc = (float(np.median(samples[k])) for k in "abc")
        d2h, h2d = tb - ta, tc - ta
        row(
            f"11.boundary[{n_envs}]",
            "OK",
            "实测",
            f"step={ta:.3f}ms; +D2H(qpos/qvel/root/8 links pos+quat)={tb:.3f}ms "
            f"(+{d2h:.3f}ms, +{100 * d2h / ta:.1f}%); +H2D ctrl push={tc:.3f}ms "
            f"(+{h2d:.3f}ms, +{100 * h2d / ta:.1f}%); SPS(a)={n_envs / ta * 1000:.3g}",
        )
        scene_a_state = robot.get_qpos().detach().cpu().numpy()
        assert np.isfinite(scene_a_state).all()

    gs.destroy()
    print("PROBE_PERF_DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
