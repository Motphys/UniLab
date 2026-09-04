#!/usr/bin/env python3
# Pinned run: uv run --with genesis-world==1.3.3 python scripts/tools/genesis_feasibility/probe_contract.py
"""SimBackend contract capability probe on Genesis 1.3.3 (research issue #1372).

Rows 1-8: cold-path metadata vs MuJoCo ground truth, qpos/qvel layout, control
semantics, set_state subset reset, keyframe, MJCF global option, sensors, DR.
Prints one compact line per row: [row] STATUS [evidence] detail.
Research probe only; no production code. MuJoCo is used solely as ground truth.
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
XML = REPO / "src/unilab/assets/robots/g1/scene_flat.xml"
N_ENVS = 8


def row(rid, status, evidence, msg):
    print(f"[{rid}] {status} [{evidence}] {msg}", flush=True)


def to_np(x):
    import torch

    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def row0(x):
    """Reduce a batched getter to env-0 row; return (value, batch_consistent)."""
    a = to_np(x)
    if a.ndim >= 2:
        flat = a.reshape(a.shape[0], -1)
        return flat[0].reshape(a.shape[1:]), bool(np.allclose(flat, flat[:1], atol=1e-5))
    return a, True


def range0(x):
    """Reduce a (lower, upper) batched getter pair to env-0 [n, 2] array."""
    lo, up = x
    return np.stack([row0(lo)[0], row0(up)[0]], axis=-1)


def main():
    import genesis as gs
    import mujoco
    import torch

    m = mujoco.MjModel.from_xml_path(str(XML))
    d = mujoco.MjData(m)
    jname = lambda j: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)  # noqa: E731
    mj_act_joints = [jname(int(m.actuator_trnid[i, 0])) for i in range(m.nu)]
    mj_hinge = [j for j in range(m.njnt) if m.jnt_type[j] in (2, 3)]

    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=0)
    rigid_opts = dict(
        integrator=gs.integrator.implicitfast,  # match g1.xml <option> explicitly
        batch_links_info=True,
        batch_dofs_info=True,
    )
    scene = gs.Scene(show_viewer=False, rigid_options=gs.options.RigidOptions(**rigid_opts))
    robot = scene.add_entity(gs.morphs.MJCF(file=str(XML)))
    torso = robot.get_link("torso_link")
    foot_l = robot.get_link("left_ankle_roll_link")
    imu = scene.add_sensor(gs.sensors.IMU(entity_idx=robot.idx, link_idx_local=torso.idx_local))
    contact_error = None
    try:  # ContactForceSensor path is exercised separately; keep scene usable if it breaks
        probe_scene = gs.Scene(
            show_viewer=False, rigid_options=gs.options.RigidOptions(**rigid_opts)
        )
        probe_robot = probe_scene.add_entity(gs.morphs.MJCF(file=str(XML)))
        probe_scene.add_sensor(
            gs.sensors.ContactForce(
                entity_idx=probe_robot.idx,
                link_idx_local=probe_robot.get_link("left_ankle_roll_link").idx_local,
            )
        )
        probe_scene.build(n_envs=N_ENVS)
        probe_scene.step()
        contact_error = ""
    except Exception as exc:  # noqa: BLE001 - probe records the failure mode
        contact_error = f"{type(exc).__name__}: {str(exc)[:80]}"
    scene.build(n_envs=N_ENVS)
    init_qpos = to_np(robot.get_qpos()).copy()
    act_joints = [j for j in robot.joints if j.n_dofs == 1]
    act_dofs = [j.dofs_idx_local[0] for j in act_joints]
    root_joint = next(j for j in robot.joints if j.n_dofs == 6)
    mj_act_of = {n: i for i, n in enumerate(mj_act_joints)}

    # ---- Row 1: cold-path metadata vs MuJoCo ground truth ----------------
    mj_1dof = [jname(j) for j in mj_hinge]
    ok = [j.name for j in act_joints] == mj_1dof
    row(
        "1a.joint_order",
        "OK" if ok else "FAIL",
        "实测",
        f"1-dof joint names/order match MuJoCo: {ok} (gs={len(act_joints)}, mj={len(mj_1dof)}); "
        f"n_dofs gs={robot.n_dofs} mj={m.nv}; n_links gs={robot.n_links} mj_nbody={m.nbody}",
    )
    gain, _ = row0(robot.get_dofs_act_gain())
    dof2joint = {int(dof): j.name for j in robot.joints for dof in j.dofs_idx_local}
    gs_act_joints = [dof2joint[int(dof)] for dof in np.nonzero(gain > 0)[0]]
    ok = gs_act_joints == mj_act_joints
    row(
        "1b.actuator_order",
        "OK" if ok else "FAIL",
        "实测",
        f"actuated joint order (act_gain>0) == mj actuator order: {ok} (n={len(gs_act_joints)})",
    )

    fr = range0(robot.get_dofs_force_range())
    fr_diff = max(
        np.abs(fr[j.dofs_idx_local[0]] - m.actuator_forcerange[mj_act_of[j.name]]).max()
        for j in act_joints
    )
    authored = bool(np.any(m.actuator_ctrlrange != 0.0))
    row(
        "1c.ctrl_range",
        "GAP",
        "源码推断+实测",
        f"actuator_ctrlrange dropped for position actuators (kept only for biastype=NONE, mjcf.py); "
        f"g1 does not author ctrlrange (mj raw all-zero: {not authored}) -> declare in owner YAML; "
        f"forcerange imported: max diff={fr_diff:.2e}",
    )

    gs_lim_all = range0(robot.get_dofs_limit())
    diffs = [
        np.abs(gs_lim_all[j.dofs_idx_local[0]] - m.jnt_range[mj]).max()
        for j, mj in zip(act_joints, mj_hinge)
    ]
    row(
        "1d.joint_range",
        "OK" if max(diffs) < 1e-5 else "FAIL",
        "实测",
        f"max |joint_range diff| over {len(diffs)} 1-dof joints = {max(diffs):.2e}",
    )

    mj_mass = {
        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b): (m.body_mass[b], m.body_ipos[b])
        for b in range(1, m.nbody)
    }
    dm, dp = [], []
    for link in robot.links:
        if link.name in mj_mass and link.inertial_mass is not None:
            dm.append(abs(float(np.ravel(to_np(link.inertial_mass))[0]) - mj_mass[link.name][0]))
            dp.append(np.abs(np.ravel(to_np(link.inertial_pos)) - mj_mass[link.name][1]).max())
    row(
        "1e.mass_ipos",
        "OK" if dm and max(dm) < 1e-4 and max(dp) < 1e-5 else "FAIL",
        "实测",
        f"per-link mass max diff={max(dm):.2e}, body_ipos max diff={max(dp):.2e} over {len(dm)} links",
    )

    mj_gfl = {}
    for gid in range(m.ngeom):
        if not (m.geom_contype[gid] or m.geom_conaffinity[gid]):
            continue  # visual-only geoms live in entity.vgeoms on the Genesis side
        bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[gid]))
        mj_gfl.setdefault(bname or "world", []).append(round(float(m.geom_friction[gid, 0]), 6))
    gs_gfl = {}
    for g in robot.geoms:
        gs_gfl.setdefault(g.link.name, []).append(round(float(to_np(g.friction)), 6))
    shared = set(mj_gfl) & set(gs_gfl)
    mismatched = [n for n in shared if sorted(mj_gfl[n]) != sorted(gs_gfl[n])]
    n_mj_col = sum(len(v) for v in mj_gfl.values())
    arm, _ = row0(robot.get_dofs_armature())
    ok = len(shared) >= 15 and not mismatched and np.allclose(arm, m.dof_armature, atol=1e-5)
    row(
        "1f.friction_armature",
        "OK" if ok else "FAIL",
        "实测+源码推断",
        f"collision-geom slide-friction per-link multisets match MuJoCo on {len(shared)} links, "
        f"mismatched={mismatched[:3]}; entity.geoms holds collision geoms only "
        f"(gs={len(robot.geoms)} vs mj collision={n_mj_col}; visual -> vgeoms); contype/conaffinity "
        f"are RE-SYNTHESIZED by solve_contype_conaffinity (bitmask ints differ from mj, collision "
        f"matrix semantics preserved) -> get_geom_contact_masks must expose genesis-native masks; "
        f"RigidGeom has no name in 1.3.3; dof_armature match={np.allclose(arm, m.dof_armature, atol=1e-5)}",
    )

    g_grav = scene.sim.options.gravity
    row(
        "1g.gravity",
        "OK",
        "实测",
        f"SimOptions.gravity={g_grav} (None -> default [0,0,-9.81]); mj opt.gravity="
        f"{np.round(m.opt.gravity, 3).tolist()}; g1.xml does not override gravity",
    )

    # ---- Row 5: keyframe (before any set_qpos mutation) -------------------
    key_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "stand")
    key_qpos = m.key_qpos[key_id].copy()
    row(
        "5.keyframe",
        "GAP",
        "实测",
        f"Genesis ignores MJCF <keyframe>: init qpos root={np.round(init_qpos[0, :3], 3).tolist()} "
        f"vs keyframe root={np.round(key_qpos[:3], 3).tolist()}, joint max diff="
        f"{np.abs(init_qpos[0, 7:] - key_qpos[7:]).max():.3f} (init joints == mj.qpos0: "
        f"{np.allclose(init_qpos[0, 7:], m.qpos0[7:], atol=1e-5)}); cold-path fallback "
        f"mujoco mj.key_qpos works: key_id={key_id}",
    )

    # ---- Row 2: qpos/qvel layout ------------------------------------------
    qp = to_np(robot.get_qpos())
    pelvis = robot.get_link("pelvis")
    lp = to_np(robot.get_links_pos(relative=False))[:, pelvis.idx_local]
    lq = to_np(robot.get_links_quat(relative=False))[:, pelvis.idx_local]
    dofp = to_np(robot.get_dofs_position())
    ok = (
        qp.shape == (N_ENVS, 36)
        and np.allclose(qp[:, :3], lp)
        and np.allclose(qp[:, 3:7], lq)
        and np.allclose(qp[:, 7:], dofp[:, 6:])
    )
    row(
        "2a.qpos_layout",
        "OK" if ok else "FAIL",
        "实测",
        f"get_qpos {qp.shape} = [root xyz | root quat wxyz | 29 joint qpos] == links_pos/quat(pelvis) "
        f"+ dofs_position[6:]: {ok}; dofs view is {dofp.shape[1]}-wide incl. 6 root dofs "
        f"(dofs_position[0:6]=root xyz+euler-zeros); MJCF entity base_link="
        f"'{robot.base_link.name}' -> entity-level get_pos/get_quat/get_vel/get_ang track the WORLD "
        f"link and are useless for the floating root; adapter must use link-addressed getters; "
        f"dtype={qp.dtype}",
    )

    knee = robot.get_joint("left_knee_joint")
    mj_knee = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "left_knee_joint")
    ok = list(knee.dofs_idx_local) == [int(m.jnt_dofadr[mj_knee])] and list(knee.qs_idx_local) == [
        int(m.jnt_qposadr[mj_knee])
    ]
    row(
        "2b.name_to_idx",
        "OK" if ok else "FAIL",
        "实测",
        f"get_joint(name).dofs_idx_local/qs_idx_local == mj jnt_dofadr/jnt_qposadr: {ok} "
        f"(root joint: n_dofs={root_joint.n_dofs}, n_qs={root_joint.n_qs}, "
        f"dofs={root_joint.dofs_idx_local}, qs={root_joint.qs_idx_local})",
    )

    robot.set_qpos(np.tile(key_qpos, (N_ENVS, 1)))
    d.qpos[:] = key_qpos
    mujoco.mj_forward(m, d)
    mj_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    g_abs = to_np(robot.get_links_pos(relative=False))[:, torso.idx_local]
    g_lq = to_np(robot.get_links_quat(relative=False))[:, torso.idx_local]
    dp_abs = np.abs(g_abs - d.xpos[mj_bid]).max()  # xpos = body frame origin (xipos is COM frame)
    dq_ = np.abs(g_lq - d.xquat[mj_bid]).max()
    ok = dp_abs < 1e-4 and min(dq_, np.abs(g_lq + d.xquat[mj_bid]).max()) < 1e-4
    row(
        "2c.quat_fk",
        "OK" if ok else "FAIL",
        "实测",
        f"FK vs mujoco at keyframe: torso_link xpos max diff={dp_abs:.2e}, quat(wxyz) diff={dq_:.2e}; "
        f"genesis get_links_pos == mujoco xpos (body frame origin, NOT xipos COM frame)",
    )

    c = float(np.cos(np.pi / 4))
    q90 = key_qpos.copy()
    q90[3:7] = [c, np.sin(np.pi / 4), 0.0, 0.0]  # 90 deg about x
    robot.set_qpos(np.tile(q90, (N_ENVS, 1)))
    robot.set_dofs_velocity(
        torch.tensor([0, 0, 0, 0, 0, 1.0]).repeat(N_ENVS, 1),
        dofs_idx_local=list(root_joint.dofs_idx_local),
    )
    scene.step()  # link velocity state is only valid after a step barrier
    ang = to_np(robot.get_links_ang())[:, pelvis.idx_local][0]
    frame = (
        "qvel body-frame + getter world (== MuJoCo semantics)"
        if np.allclose(ang, [0, -1, 0], atol=0.1)
        else "qvel world-frame or getter passthrough"
        if np.allclose(ang, [0, 0, 1], atol=0.1)
        else "unknown"
    )
    row(
        "2d.root_ang_frame",
        "OK" if "MuJoCo" in frame else "GAP",
        "实测",
        f"set root dofs[3:6]=[0,0,1] at quat 90deg-x, after 1 step links_ang(pelvis)="
        f"{np.round(ang, 4).tolist()}; interpretation: {frame}; contract wants qvel body-frame "
        f"columns in set_state and world-frame get_base_ang_vel",
    )

    # ---- Row 3: control semantics ------------------------------------------
    gs_kp, _ = row0(robot.get_dofs_kp())
    gs_kv, _ = row0(robot.get_dofs_kv())
    dk = max(
        abs(gs_kp[j.dofs_idx_local[0]] - m.actuator_gainprm[mj_act_of[j.name], 0])
        for j in act_joints
    )
    dv = max(
        abs(gs_kv[j.dofs_idx_local[0]] + m.actuator_biasprm[mj_act_of[j.name], 2])
        for j in act_joints
    )
    row(
        "3a.pd_gains",
        "OK" if max(dk, dv) < 1e-3 else "FAIL",
        "实测",
        f"MJCF <position kp kv> -> act_gain/act_bias (PD-reducible): max kp diff={dk:.2e}, "
        f"kv diff={dv:.2e} over 29 actuators",
    )

    robot.set_qpos(np.tile(key_qpos, (N_ENVS, 1)))
    robot.zero_all_dofs_velocity()
    stand_act = np.array([key_qpos[j.qs_idx_local[0]] for j in act_joints], dtype=np.float32)
    elbow_i = next(i for i, j in enumerate(act_joints) if j.name == "left_elbow_joint")
    t2 = stand_act.copy()
    t2[elbow_i] += 0.2
    robot.control_dofs_position(torch.tensor(np.tile(t2, (N_ENVS, 1))), dofs_idx_local=act_dofs)
    errs = []
    held = 0.0
    for i in range(100):
        scene.step()
        if i == 4:
            held = float(np.abs(row0(robot.get_dofs_control_force())[0][act_dofs]).max())
        if (i + 1) in (10, 30, 100):
            errs.append(abs(to_np(robot.get_dofs_position())[0, act_dofs[elbow_i]] - t2[elbow_i]))
    final_dof = to_np(robot.get_dofs_position())
    root_z = float(to_np(robot.get_pos())[0, 2])
    ok = held > 1e-3 and errs[0] < 0.08 and bool(np.isfinite(final_dof).all())
    row(
        "3b.ctrl_held_pd",
        "OK" if ok else "FAIL",
        "实测",
        f"one control_dofs_position call held across 100x scene.step() (max |ctrl force|={held:.2f} "
        f"after 5 steps); elbow +0.2rad err @10/30/100 steps={np.round(errs, 4).tolist()} (robot "
        f"topples without balance ctrl, root z@{100}={root_z:.2f}); == MuJoCo step(ctrl,nsteps) "
        f"ctrl-broadcast semantics; no per-substep hook (adapter loops scene.step for nsteps)",
    )

    # ---- Row 4: set_state subset reset --------------------------------------
    robot.set_qpos(np.tile(key_qpos, (N_ENVS, 1)))
    robot.zero_all_dofs_velocity()
    snap = to_np(robot.get_qpos()).copy()
    q2 = key_qpos.copy()
    q2[0] += 0.5
    q2[7] += 0.1
    robot.set_qpos(torch.tensor(np.tile(q2, (2, 1)), dtype=torch.float32), envs_idx=[2, 5])
    vel = np.zeros((1, 35), dtype=np.float32)
    vel[0, act_dofs[0]] = 0.7
    robot.set_dofs_velocity(torch.tensor(vel), dofs_idx_local=list(range(35)), envs_idx=[1])
    after = to_np(robot.get_qpos())
    keep = [0, 1, 3, 4, 6, 7]
    touched = np.allclose(after[[2, 5]], q2, atol=1e-5)
    others = np.allclose(after[keep], snap[keep], atol=1e-5)
    dv_after = to_np(robot.get_dofs_velocity())
    vok = abs(dv_after[1, act_dofs[0]] - 0.7) < 1e-5 and abs(dv_after[0, act_dofs[0]]) < 1e-6
    for _ in range(3):
        scene.step()
    rt = to_np(robot.get_qpos())
    ok = touched and others and vok and bool(np.isfinite(rt).all())
    row(
        "4.set_state_subset",
        "OK" if ok else "FAIL",
        "实测",
        f"env-indexed set without rebuild: touched==target {touched}, untouched preserved {others}, "
        f"subset set_dofs_velocity {vok}; round-trip set->step->get finite={bool(np.isfinite(rt).all())} "
        f"shape={rt.shape} dtype={rt.dtype}",
    )

    # ---- Row 6: MJCF global <option> ----------------------------------------
    row(
        "6.global_option",
        "GAP",
        "实测+源码推断",
        f"<option timestep={m.opt.timestep:.6f} integrator=implicitfast> dropped by importer "
        f"(morphs.py note); effective dt from Sim/RigidOptions (here default {float(scene.sim.dt)}); "
        f"integrator/cone/solver-iters are scene-level RigidOptions -> owner-YAML fields; "
        f"cone=elliptic only warned (mjcf.py parse_xml)",
    )

    # ---- Row 7: sensors ------------------------------------------------------
    r = imu.read()
    acc, gyro = to_np(r.lin_acc), to_np(r.ang_vel)
    netcf = to_np(robot.get_links_net_contact_force())
    foot_force = float(np.linalg.norm(netcf[0, foot_l.idx_local]))
    cfs_msg = (
        "ContactForceSensor read OK (with batch_*_info=True)"
        if contact_error == ""
        else f"ContactForceSensor FAILED with batch_*_info=True on torch 2.7.0: {contact_error}"
    )
    row(
        "7.sensors",
        "GAP",
        "源码推断+实测",
        f"none of mj nsensor={m.nsensor} MJCF sensors imported (no parse code); working equivalents: "
        f"IMUSensor acc{acc.shape}/gyro{tuple(gyro.shape)} noise-free, links_net_contact_force"
        f"{netcf.shape} (foot |F|={foot_force:.1f}N at stand; |F|>thr == MJCF contact data=found); "
        f"{cfs_msg}; velocimeter/framepos/framequat/framezaxis/framelinvel -> get_links_pos/quat/vel "
        f"+ adapter site-offset math",
    )

    # ---- Row 8: DR -----------------------------------------------------------
    mass_b = to_np(robot.get_links_inertial_mass())
    solver = scene.sim.rigid_solver
    new_mass = float(mass_b[0, torso.idx_local]) * 1.25
    solver.set_links_inertial_mass([new_mass], links_idx=[torso.idx], envs_idx=[3])
    mass_a = to_np(robot.get_links_inertial_mass())
    mok = (
        abs(mass_a[3, torso.idx_local] - new_mass) < 1e-4
        and abs(mass_a[0, torso.idx_local] - mass_b[0, torso.idx_local]) < 1e-4
    )

    fl_b = to_np(robot.get_dofs_frictionloss())
    solver.set_dofs_frictionloss([9.9], dofs_idx=list(knee.dofs_idx), envs_idx=[4])
    fl_a = to_np(robot.get_dofs_frictionloss())
    fok = (
        abs(fl_a[4, knee.dofs_idx_local[0]] - 9.9) < 1e-4
        and abs(fl_a[0, knee.dofs_idx_local[0]] - fl_b[0, knee.dofs_idx_local[0]]) < 1e-4
    )

    kp_b = to_np(robot.get_dofs_kp())
    robot.set_dofs_kp([55.5], dofs_idx_local=knee.dofs_idx_local, envs_idx=[5])
    kp_a = to_np(robot.get_dofs_kp())
    kok = (
        abs(kp_a[5, knee.dofs_idx_local[0]] - 55.5) < 1e-3
        and abs(kp_a[0, knee.dofs_idx_local[0]] - kp_b[0, knee.dofs_idx_local[0]]) < 1e-3
    )

    robot.set_friction_ratio(
        np.linspace(1.0, 1.5, N_ENVS, dtype=np.float32)[:, None], links_idx_local=[foot_l.idx_local]
    )
    solver.apply_links_external_force(
        np.array([[0, 0, 50.0]], dtype=np.float32), links_idx=[torso.idx], envs_idx=[0]
    )
    scene.step()
    row(
        "8.dr",
        "OK" if mok and fok and kok else "FAIL",
        "实测",
        f"per-env DR round-trip: links_inertial_mass {mok}, dofs_frictionloss {fok}, dofs_kp {kok} "
        f"(require batch_links_info/batch_dofs_info=True at build); set_friction_ratio + "
        f"solver.apply_links_external_force callable (effect 未验证); entity-level apply_links_* "
        f"absent in 1.3.3 -> solver API with global idx",
    )

    print("PROBE_CONTRACT_DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
