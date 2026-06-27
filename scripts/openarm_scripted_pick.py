"""Scripted (non-RL) staged pick controller for the OpenArm demo, for a clean demo video.

Drives the left arm through a deterministic, textbook pick:
  1. APPROACH  -- move the TCP above the cube (xy aligned, z = cube_z + pregrasp_h), gripper open.
  2. DESCEND   -- lower the TCP to the cube (z = cube_z + grasp_dz), gripper still open.
  3. CLOSE     -- hold the arm, close the gripper for a few steps.
  4. LIFT      -- raise the (now grasped) cube straight up, gripper held closed.

Control is operational-space: a damped least-squares IK delta on the left-arm
Jacobian (``get_site_jacobian_w`` on the ``openarm_left_tcp`` site) drives the TCP
toward the per-phase world target; the resulting joint delta is normalized by
``action_scale`` into the same [-1, 1] action vector the policy uses, so this runs
through the exact same ``run_playback_mode`` pipeline (no checkpoint needed).

Example:
    MUJOCO_GL=egl HIP_VISIBLE_DEVICES=0 uv run scripts/openarm_scripted_pick.py \
        task=openarm_demo_pick/mujoco_lift3d_contgrip training.play_steps=260
"""

import sys
from pathlib import Path
from typing import Any, cast

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

ROOT_DIR = Path(__file__).parent.parent
SRC_DIR = ROOT_DIR / "src"
for _p in (SRC_DIR, ROOT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from train_rsl_rl import (  # type: ignore[import-not-found]
    _algo_config_dict,
    _resolve_ppo_wrapper_cls,
    build_ppo_play_env_cfg_override,
)

from unilab.envs.common.rotation import np_quat_canonicalize, np_quat_inv, np_quat_mul
from unilab.training import (
    apply_configured_training_seed,
    create_env,
    ensure_registries,
    log_playback_plan,
)

# Phase ids.
APPROACH, DESCEND, CLOSE, LIFT = 0, 1, 2, 3

# Controller constants (metres / radians / steps). Tuned for the lift3d cell.
PREGRASP_H = 0.12  # height above the cube for the pre-grasp pose
GRASP_DZ = 0.01  # TCP target height relative to cube centre at grasp
LIFT_TARGET_Z = 1.26  # absolute world z to raise the cube to
TOL_XY = 0.045  # xy alignment tolerance to advance a phase
TOL_ANG = 0.35  # orientation alignment tolerance (rad); arm cannot reach
# perfectly vertical here, ~mostly-down is the best top-down pose
ABOVE_MARGIN = 0.04  # TCP must clear the cube by this much before descending
GRASP_REACH = 0.03  # TCP within this height of the cube => jaws are around it
APPROACH_MAX = 170  # max steps in APPROACH before forcing DESCEND (deterministic)
DESCEND_MAX = 90  # max steps in DESCEND before forcing CLOSE
CLOSE_STEPS = 25  # steps to hold the close command before lifting
IK_DAMPING = 0.12  # damped least-squares lambda
ORI_GAIN = 0.8  # weight on the orientation error block in the IK solve
SPEED = 0.35  # per-step command scale; <1 avoids position-ctrl windup/overshoot
GRIPPER_OPEN = -1.0  # action that drives the finger ctrl toward open (lo)
GRIPPER_CLOSE = 1.0  # action that drives the finger ctrl toward closed (hi)
ARM_JOINTS = [f"openarm_left_joint{i}" for i in range(1, 8)]
# Target world orientation: identity quat -> gripper approach axis (local -z)
# points straight down (world -z), jaws open along world y. See home-pose probe.
TARGET_QUAT_W = np.array([1.0, 0.0, 0.0, 0.0])


class ScriptedPickController:
    def __init__(self, env: Any, device: str) -> None:
        self.env = env
        self.device = device
        self.backend = env._backend
        self.cube_ids = env._cube_body_ids
        self.action_scale = float(env._cfg.action_scale)
        self.n = int(env.num_envs)
        self.num_actions = int(env.action_space.shape[0])
        self.site_id = int(self.backend.get_site_ids(["openarm_left_tcp"])[0])
        self.jac_dof = self.backend.get_joint_dof_indices(ARM_JOINTS)
        self.ee_body_ids = self.backend.get_body_ids(["openarm_left_ee_base_link"])
        self.n_arm = len(ARM_JOINTS)
        self.phase = np.zeros(self.n, dtype=np.int32)
        self.close_ctr = np.zeros(self.n, dtype=np.int32)
        self.phase_steps = np.zeros(self.n, dtype=np.int32)

    def reset(self) -> None:
        self.phase[:] = APPROACH
        self.close_ctr[:] = 0
        self.phase_steps[:] = 0

    def _tcp_w(self) -> np.ndarray:
        return np.asarray(self.env._grasp_point_w(), dtype=np.float64)

    def _cube_w(self) -> np.ndarray:
        return np.asarray(self.backend.get_body_pos_w(self.cube_ids)[:, 0, :], dtype=np.float64)

    def _ee_quat_w(self) -> np.ndarray:
        return np.asarray(self.backend.get_body_quat_w(self.ee_body_ids)[:, 0, :], dtype=np.float64)

    def _orn_err_w(self, curr_quat: np.ndarray) -> np.ndarray:
        """World-frame rotation-vector error toward the fixed point-down target."""
        goal = np.broadcast_to(TARGET_QUAT_W, curr_quat.shape)
        rel = np_quat_canonicalize(np_quat_mul(goal, np_quat_inv(curr_quat)))
        sign = np.where(rel[:, 0:1] < 0.0, -1.0, 1.0)
        return rel[:, 1:] * sign

    def _ik_dq(self, target_w: np.ndarray, tcp_w: np.ndarray, orn_err: np.ndarray) -> np.ndarray:
        """6-DOF damped least-squares joint delta: drive TCP to ``target_w`` and
        the gripper toward the point-down orientation (world-frame Jacobian)."""
        jacp, jacr = self.backend.get_site_jacobian_w(self.site_id, self.jac_dof)
        jacp = np.asarray(jacp, dtype=np.float64)  # (n, 3, n_arm)
        jacr = np.asarray(jacr, dtype=np.float64) * ORI_GAIN  # (n, 3, n_arm)
        jac = np.concatenate([jacp, jacr], axis=1)  # (n, 6, n_arm)
        pos_err = target_w - tcp_w
        dpose = np.concatenate([pos_err, orn_err * ORI_GAIN], axis=1)[:, :, None]
        jjt = np.matmul(jac, np.swapaxes(jac, 1, 2))  # (n, 6, 6)
        jjt += np.eye(6)[None] * (IK_DAMPING**2)
        solved = np.linalg.solve(jjt, dpose)  # (n, 6, 1)
        dq = np.matmul(np.swapaxes(jac, 1, 2), solved)[:, :, 0]  # (n, n_arm)
        return dq

    def step_actions(self) -> torch.Tensor:
        tcp = self._tcp_w()
        cube = self._cube_w()
        orn_err = self._orn_err_w(self._ee_quat_w())
        xy_err = np.linalg.norm(tcp[:, :2] - cube[:, :2], axis=-1)
        ang_err = np.linalg.norm(orn_err, axis=-1)

        # Per-phase world-space TCP target and gripper command.
        target = np.empty_like(tcp)
        gripper = np.full(self.n, GRIPPER_OPEN, dtype=np.float64)

        is_app = self.phase == APPROACH
        is_desc = self.phase == DESCEND
        is_close = self.phase == CLOSE
        is_lift = self.phase == LIFT

        target[:, 0] = cube[:, 0]
        target[:, 1] = cube[:, 1]
        # z target by phase.
        target[is_app, 2] = cube[is_app, 2] + PREGRASP_H
        target[is_desc, 2] = cube[is_desc, 2] + GRASP_DZ
        target[is_close, 2] = tcp[is_close, 2]  # hold position
        target[is_close, 0] = tcp[is_close, 0]
        target[is_close, 1] = tcp[is_close, 1]
        target[is_lift, 0] = tcp[is_lift, 0]  # keep xy where it grasped
        target[is_lift, 1] = tcp[is_lift, 1]
        target[is_lift, 2] = LIFT_TARGET_Z

        gripper[is_close | is_lift] = GRIPPER_CLOSE

        dq = self._ik_dq(target, tcp, orn_err)
        arm_act = np.clip(dq / self.action_scale, -1.0, 1.0) * SPEED

        actions = np.zeros((self.n, self.num_actions), dtype=np.float32)
        actions[:, : self.n_arm] = arm_act
        actions[:, -1] = gripper

        self._advance_phases(tcp, cube, xy_err, ang_err)
        return torch.from_numpy(actions).to(self.device)

    def _advance_phases(self, tcp, cube, xy_err, ang_err) -> None:
        self.phase_steps += 1
        above = tcp[:, 2] - cube[:, 2]

        # APPROACH -> DESCEND: aligned above the cube and (mostly) pointing down,
        # or a max-dwell fallback so the deterministic demo always progresses.
        aligned = (xy_err < TOL_XY) & (ang_err < TOL_ANG) & (above > ABOVE_MARGIN)
        adv_app = (self.phase == APPROACH) & (aligned | (self.phase_steps >= APPROACH_MAX))
        self._set_phase(adv_app, DESCEND)

        # DESCEND -> CLOSE: TCP has reached the cube (jaws around it), or fallback.
        reached = (xy_err < TOL_XY + 0.02) & (above < GRASP_REACH)
        adv_desc = (self.phase == DESCEND) & (reached | (self.phase_steps >= DESCEND_MAX))
        self._set_phase(adv_desc, CLOSE)

        # CLOSE -> LIFT after holding the close command for CLOSE_STEPS.
        in_close = self.phase == CLOSE
        self.close_ctr[in_close] += 1
        self._set_phase(in_close & (self.close_ctr >= CLOSE_STEPS), LIFT)

    def _set_phase(self, mask: np.ndarray, new_phase: int) -> None:
        self.phase[mask] = new_phase
        self.phase_steps[mask] = 0


@hydra.main(version_base="1.3", config_path="../conf/ppo", config_name="config")
def main(cfg: DictConfig) -> None:
    ensure_registries()
    apply_configured_training_seed(cfg, torch_runtime=True, cuda=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    num_envs = int(OmegaConf.select(cfg, "training.play_env_num", default=16))
    num_steps = int(OmegaConf.select(cfg, "training.play_steps", default=200))

    env = create_env(cfg, num_envs=num_envs, env_cfg_override=build_ppo_play_env_cfg_override(cfg))
    wrapper_cls = _resolve_ppo_wrapper_cls(_algo_config_dict(cfg))
    wrapped_env = wrapper_cls(env, device=device)

    controller = ScriptedPickController(env, device)

    out_dir = ROOT_DIR / "logs" / "rsl_rl_ppo" / "OpenArmDemoPick" / "scripted_pick"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_video = out_dir / "play_video_scripted.mp4"

    def _initialize():
        obs = wrapped_env.reset()[0]
        controller.reset()
        return obs

    def _step(_obs):
        actions = controller.step_actions()
        return wrapped_env.step(actions)[0]

    with torch.inference_mode():
        env.run_playback_mode(
            play_render_mode=getattr(cfg.training, "play_render_mode", "auto"),
            play_steps=num_steps,
            output_video=output_video,
            render_spacing=float(getattr(cfg.training, "render_spacing", 1.0)),
            render_offset_mode=str(getattr(env.cfg, "render_offset_mode", "grid")),
            initialize=_initialize,
            step=_step,
            camera_kwargs={
                "cam_distance": cfg.training.cam_distance,
                "cam_elevation": cfg.training.cam_elevation,
                "cam_azimuth": cfg.training.cam_azimuth,
                "cam_lookat": getattr(cfg.training, "cam_lookat", None),
                "play_hide_geom_groups": getattr(cfg.training, "play_hide_geom_groups", None),
                "play_video_fps": getattr(cfg.training, "play_video_fps", None),
            },
            on_plan=log_playback_plan,
        )
    print(f"Scripted pick video: {output_video}")


if __name__ == "__main__":
    main()
