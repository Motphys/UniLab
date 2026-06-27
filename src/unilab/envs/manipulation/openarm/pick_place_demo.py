"""OpenArm bimanual pick / place bring-up on vendored ``demo.xml`` (cell + cube + frame)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.backend import create_backend
from unilab.base.base import EnvCfg
from unilab.base.np_env import NpEnv, NpEnvState
from unilab.base.scene import SceneCfg
from unilab.dr.provider import DomainRandomizationProvider
from unilab.dr.types import DomainRandomizationCapabilities, ResetPlan
from unilab.dtype_config import get_global_dtype
from unilab.envs.common.rotation import np_quat_apply_batched

# Position actuators in the vendored model, mapped to their qpos addresses. The lifter prismatic
# joint sits at index 9 in the full actuator vector and is dropped from the policy action space.
_ACTUATOR_QPOS_INDICES: tuple[int, ...] = (
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
)
_NU_FULL = len(_ACTUATOR_QPOS_INDICES)
_LIFTER_ACTUATOR_NAME = "lifter_ctrl"
_RIGHT_FREEZE_JOINT_NAMES: tuple[str, ...] = (
    "openarm_right_joint1",
    "openarm_right_joint2",
    "openarm_right_joint3",
    "openarm_right_joint4",
    "openarm_right_joint5",
    "openarm_right_joint6",
    "openarm_right_joint7",
    "openarm_right_finger_joint1",
    "openarm_right_finger_joint2",
)


def _ctrl_from_full_qpos(qpos: np.ndarray) -> np.ndarray:
    """Extract position-actuator commands from a full MuJoCo qpos vector."""
    idx = np.asarray(_ACTUATOR_QPOS_INDICES, dtype=np.int64)
    return np.asarray(qpos[idx], dtype=np.float64)


@dataclass
class OpenArmDemoPickRewardConfig:
    """Scalar reward weights (Hydra ``reward.scales`` maps onto these names)."""

    scales: dict[str, float] = field(
        default_factory=lambda: {
            "reach": 1.0,
            "place": 0.8,
            "lift": 0.35,
            "grasp": 0.22,
            "settle": 0.0,
            "success": 25.0,
            "hold_success": 0.0,
            "action": -0.01,
            "drop": -1.0,
            "lift_bonus": 0.0,
            "goal_track_coarse": 0.0,
            "goal_track_fine": 0.0,
            # Staged-grasp shaping (off by default; enabled via the contgrip config).
            "approach": 0.0,
            "premature_close": 0.0,
            "action_rate": 0.0,
            "firm_grasp": 0.0,
        }
    )

    fall_z: float = 0.92
    success_xy: float = 0.028
    lift_z: float = 1.04

    table_z: float = 1.008
    lift_shaping_span: float = 0.12

    grasp_proximity_decay: float = 6.0

    # ``place`` is gated by finger closure × EE-cube proximity.
    place_use_grasp_gate: bool = True

    # Legacy flag kept for backward compatibility; superseded by ``lift_gate_mode``.
    lift_use_grasp_gate: bool = False

    # ``grasp`` (closure×proximity) / ``proximity`` (EE-cube only) / ``none``.
    lift_gate_mode: str | None = None

    settle_xy_decay: float = 12.0
    settle_arm_vel_coeff: float = 0.03
    settle_use_lift_gate: bool = True

    terminate_on_success: bool = True

    reach_use_tcp: bool = False

    # Local offset (m) applied at the fingertip TCP, expressed in the inner-finger frame.
    reach_tcp_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)

    goal_success_mode: str = "xy_lift"

    success_dist: float = 0.05

    lift_margin: float = 0.05

    goal_track_coarse_std: float = 0.1
    goal_track_fine_std: float = 0.03

    # --- Staged-grasp shaping (used by ``approach`` / ``premature_close``) ---
    # Pre-grasp clearance above the cube that the ``approach`` term rewards
    # hovering at (with the gripper still open).
    pregrasp_h: float = 0.10
    approach_xy_decay: float = 8.0
    approach_z_decay: float = 8.0
    # Closure far from the cube (in xy) is penalized to discourage grasping at air.
    premature_close_decay: float = 8.0


@registry.envcfg("OpenArmDemoPick")
@dataclass
class OpenArmDemoPickCfg(EnvCfg):
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "openarm_mujoco_v2" / "demo.xml")
        )
    )

    sim_dt: float = 0.002
    ctrl_dt: float = 0.02
    max_episode_seconds: float = 40.0
    render_spacing: float = 2.0

    base_name: str = "openarm_lifter_link"
    action_scale: float = 0.04
    reward_config: OpenArmDemoPickRewardConfig | None = None

    goal_pos: tuple[float, float, float] = (0.52, 0.0, 1.03)
    cube_xy_range: tuple[float, float, float, float] = (0.4, 0.5, -0.06, 0.06)
    cube_spawn_z: float = 1.05
    fix_lifter: bool = True
    left_arm_only: bool = False
    rigid_freeze_right_arm: bool = False

    binary_gripper: bool = False

    playback_hide_geom_groups: tuple[int, ...] | None = None
    playback_video_fps: float | None = None

    def validate(self) -> None:
        super().validate()
        if self.reward_config is None:
            raise ValueError("reward_config must be set (typically via Hydra `reward` block)")
        if self.left_arm_only and not self.fix_lifter:
            raise ValueError("left_arm_only=True requires fix_lifter=True")
        if self.rigid_freeze_right_arm and not self.left_arm_only:
            raise ValueError("rigid_freeze_right_arm=True requires left_arm_only=True")
        mode = self.reward_config.goal_success_mode
        if mode not in ("xy_lift", "lift3d"):
            raise ValueError(
                f"reward_config.goal_success_mode must be 'xy_lift' or 'lift3d', got {mode!r}"
            )


class OpenArmDemoPickDRProvider(DomainRandomizationProvider):
    """Keyframe robot reset + randomized cube pose on the table."""

    def validate(self, env: Any, capabilities: DomainRandomizationCapabilities) -> None:
        pass

    def build_reset_plan(self, env: "OpenArmDemoPickEnv", env_ids: np.ndarray) -> ResetPlan:
        cfg = env._cfg
        n = int(env_ids.shape[0])
        home = np.asarray(env._backend.get_keyframe_qpos("home"), dtype=np.float64)
        qpos = np.broadcast_to(home, (n, home.shape[0])).copy()
        (low_x, high_x, low_y, high_y) = cfg.cube_xy_range
        qpos[:, 19] = np.random.uniform(low_x, high_x, size=(n,))
        qpos[:, 20] = np.random.uniform(low_y, high_y, size=(n,))
        qpos[:, 21] = float(cfg.cube_spawn_z)
        qpos[:, 22:26] = np.tile(np.array([1, 0, 0, 0], dtype=np.float64), (n, 1))
        qvel = np.zeros((n, env.nv), dtype=np.float64)
        home_ctrl = _ctrl_from_full_qpos(home)
        home_ctrl_b = np.broadcast_to(home_ctrl, (n, _NU_FULL)).astype(
            get_global_dtype(), copy=False
        )
        info_updates = {
            "last_actions": np.zeros((n, env.num_policy_actions), dtype=get_global_dtype()),
            "prev_actions": np.zeros((n, env.num_policy_actions), dtype=get_global_dtype()),
            "prev_ctrl": home_ctrl_b.copy(),
            "goal_pos": np.broadcast_to(
                np.asarray(cfg.goal_pos, dtype=get_global_dtype()), (n, 3)
            ).copy(),
        }
        return ResetPlan(
            env_ids=env_ids, qpos=qpos, qvel=qvel, info_updates=info_updates, randomization=None
        )

    def build_reset_observation(
        self,
        env: "OpenArmDemoPickEnv",
        env_ids: np.ndarray,
        info_updates: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        rows = np.asarray(env_ids, dtype=np.intp)
        obs = env._assemble_obs_rows(rows, info_updates["goal_pos"], info_updates["last_actions"])
        return {"obs": obs}


@registry.env("OpenArmDemoPick", sim_backend="mujoco")
class OpenArmDemoPickEnv(NpEnv):
    """Pick / place on the orange cube: EE-centric reach shaping (left or right arm per cfg)."""

    _cfg: OpenArmDemoPickCfg

    def __init__(self, cfg: OpenArmDemoPickCfg, num_envs: int, backend_type: str) -> None:
        if cfg.reward_config is None:
            raise ValueError("reward_config must be provided via Hydra configuration")
        cfg.validate()

        arm = "left" if cfg.left_arm_only else "right"
        ee_body = f"openarm_{arm}_ee_base_link"
        extra_track_body_names = (
            ee_body,
            f"openarm_{arm}_ee_inner_finger",
            f"openarm_{arm}_ee_outer_finger",
        )

        backend = create_backend(
            backend_type,
            cfg.scene,
            num_envs,
            cfg.sim_dt,
            base_name=cfg.base_name,
            add_body_sensors=True,
            extra_track_body_names=extra_track_body_names,
        )
        super().__init__(cfg, backend, num_envs)

        self._np_dtype = get_global_dtype()
        self._reward_cfg = cfg.reward_config

        self._reward_fns = {
            "reach": self._reward_reach,
            "place": self._reward_place,
            "lift": self._reward_lift,
            "grasp": self._reward_grasp,
            "settle": self._reward_settle,
            "success": self._reward_success,
            "hold_success": self._reward_hold_success,
            "action": self._reward_action,
            "drop": self._reward_drop,
            "lift_bonus": self._reward_lift_bonus,
            "goal_track_coarse": self._reward_goal_track_coarse,
            "goal_track_fine": self._reward_goal_track_fine,
            "approach": self._reward_approach,
            "premature_close": self._reward_premature_close,
            "action_rate": self._reward_action_rate,
            "firm_grasp": self._reward_firm_grasp,
        }

        act_range = np.asarray(self._backend.get_actuator_ctrl_range(), dtype=self._np_dtype)
        if int(act_range.shape[0]) != _NU_FULL:
            raise ValueError(f"Expected {_NU_FULL} actuators, got {act_range.shape[0]}")
        self._ctrl_lo = act_range[:, 0]
        self._ctrl_hi = act_range[:, 1]
        self._ctrl_mid = (self._ctrl_lo + self._ctrl_hi) * 0.5
        self._ctrl_span = (self._ctrl_hi - self._ctrl_lo) + 1e-08

        self._fix_lifter = bool(cfg.fix_lifter)
        self._left_arm_only = bool(cfg.left_arm_only)

        if self._fix_lifter:
            import mujoco

            m = self._backend.model
            li = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, _LIFTER_ACTUATOR_NAME))
            if li < 0:
                raise ValueError(
                    f"fix_lifter=True requires actuator {_LIFTER_ACTUATOR_NAME!r} in the scene."
                )
            self._lifter_actuator_idx = li
            home = np.asarray(self._backend.get_keyframe_qpos("home"), dtype=np.float64)
            home_ctrl = _ctrl_from_full_qpos(home)
            lifter_u = home_ctrl[self._lifter_actuator_idx]
            self._lifter_fixed_ctrl = self._np_dtype.type(lifter_u)

            if self._left_arm_only:
                self._policy_active_actuator_idx = np.arange(1, 9, dtype=np.int64)
                self._right_actuator_idx = np.arange(9, 17, dtype=np.int64)
                self._policy_qpos_ix = np.asarray(_ACTUATOR_QPOS_INDICES[1:9], dtype=np.int64)
                self._num_policy_action = 8
                self._right_fixed_ctrl = np.asarray(
                    home_ctrl[self._right_actuator_idx], dtype=self._np_dtype
                )
                self._ctrl_mid_policy = self._ctrl_mid[self._policy_active_actuator_idx]
                self._ctrl_span_policy = self._ctrl_span[self._policy_active_actuator_idx]
            else:
                self._right_actuator_idx = np.empty(0, dtype=np.int64)
                self._right_fixed_ctrl = np.empty(0, dtype=self._np_dtype)
                self._policy_active_actuator_idx = np.array(
                    [i for i in range(_NU_FULL) if i != li], dtype=np.int64
                )
                self._arm_actuator_idx = self._policy_active_actuator_idx.copy()
                if int(self._arm_actuator_idx.size) != _NU_FULL - 1:
                    raise RuntimeError("internal: expected one lifter actuator to drop from policy")
                self._policy_qpos_ix = np.asarray(_ACTUATOR_QPOS_INDICES[1:], dtype=np.int64)
                self._num_policy_action = _NU_FULL - 1
                self._ctrl_mid_policy = self._ctrl_mid[self._arm_actuator_idx]
                self._ctrl_span_policy = self._ctrl_span[self._arm_actuator_idx]
        else:
            self._lifter_actuator_idx = -1
            self._arm_actuator_idx = np.arange(_NU_FULL, dtype=np.int64)
            self._policy_active_actuator_idx = self._arm_actuator_idx.copy()
            self._right_actuator_idx = np.empty(0, dtype=np.int64)
            self._right_fixed_ctrl = np.empty(0, dtype=self._np_dtype)
            self._policy_qpos_ix = np.asarray(_ACTUATOR_QPOS_INDICES, dtype=np.int64)
            self._num_policy_action = _NU_FULL
            self._lifter_fixed_ctrl = self._np_dtype.type(0.0)
            self._ctrl_mid_policy = self._ctrl_mid
            self._ctrl_span_policy = self._ctrl_span

        self._grasp_finger_actuator_idx = None
        if self._fix_lifter and self._left_arm_only:
            self._grasp_finger_actuator_idx = int(self._policy_active_actuator_idx[-1])

        self._binary_gripper = bool(cfg.binary_gripper)
        if self._binary_gripper and self._grasp_finger_actuator_idx is None:
            raise ValueError(
                "binary_gripper=True requires a resolvable gripper actuator "
                "(fix_lifter=True and left_arm_only=True)."
            )

        self._policy_obs_dim = self._num_policy_action * 2 + 6

        self._cube_body_ids = self._backend.get_body_ids(["orange_cube"])
        self._ee_body_ids = self._backend.get_body_ids([ee_body])

        self._use_tcp = bool(self._reward_cfg.reach_use_tcp)
        self._tcp_body_ids = None

        self._tcp_offset_local = np.asarray(self._reward_cfg.reach_tcp_offset, dtype=self._np_dtype)
        self._tcp_has_offset = bool(np.any(self._tcp_offset_local != 0.0))

        if self._use_tcp:
            arm = "left" if self._left_arm_only else "right"
            finger_bodies = [
                f"openarm_{arm}_ee_inner_finger",
                f"openarm_{arm}_ee_outer_finger",
            ]
            try:
                self._tcp_body_ids = self._backend.get_body_ids(finger_bodies)
            except (ValueError, KeyError):
                self._tcp_body_ids = None
            if self._tcp_body_ids is None:
                raise ValueError(
                    f"reach_use_tcp=True requires finger bodies {finger_bodies!r} in the scene."
                )

        self._settle_dof_vel_ix = None
        if self._backend.backend_type == "mujoco":
            arm = "left" if self._left_arm_only else "right"
            jn = tuple(f"openarm_{arm}_joint{i}" for i in range(1, 8)) + (
                f"openarm_{arm}_finger_joint1",
            )
            try:
                self._settle_dof_vel_ix = np.asarray(
                    self._backend.get_joint_dof_vel_indices(jn), dtype=np.int64
                )
            except ValueError:
                self._settle_dof_vel_ix = None

        self._action_space = gym.spaces.Box(-1.0, 1.0, (self._num_policy_action,), dtype=np.float32)
        self.nq = int(self._backend.model.nq)
        self.nv = int(self._backend.model.nv)

        self._init_domain_randomization(OpenArmDemoPickDRProvider())

        self._rigid_freeze_right_arm = bool(cfg.rigid_freeze_right_arm)
        self._right_freeze_qadr = None
        self._right_freeze_dofadr = None
        self._home_right_qpos_freeze = None
        if (
            self._left_arm_only
            and self._rigid_freeze_right_arm
            and self._backend.backend_type == "mujoco"
        ):
            import mujoco

            mm = self._backend.model
            qadr: list[int] = []
            dofadr: list[int] = []
            for joint_name in _RIGHT_FREEZE_JOINT_NAMES:
                jid = int(mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_JOINT, joint_name))
                if jid < 0:
                    raise ValueError(f"joint {joint_name!r} not found for rigid_freeze_right_arm")
                qadr.append(int(mm.jnt_qposadr[jid]))
                dofadr.append(int(mm.jnt_dofadr[jid]))
            self._right_freeze_qadr = np.asarray(qadr, dtype=np.int64)
            self._right_freeze_dofadr = np.asarray(dofadr, dtype=np.int64)
            home_q = np.asarray(self._backend.get_keyframe_qpos("home"), dtype=np.float64)
            self._home_right_qpos_freeze = home_q[self._right_freeze_qadr].astype(
                self._np_dtype, copy=False
            )

    def after_physics_substeps(self, actions: np.ndarray) -> None:
        del actions
        if not (
            self._left_arm_only
            and self._rigid_freeze_right_arm
            and self._right_freeze_qadr is not None
            and self._home_right_qpos_freeze is not None
            and self._right_freeze_dofadr is not None
        ):
            return
        fn = getattr(self._backend, "hard_set_joint_subset_and_forward", None)
        if fn is None:
            return
        fn(
            qpos_adrs=self._right_freeze_qadr,
            qpos_values=self._home_right_qpos_freeze,
            dof_adrs=self._right_freeze_dofadr,
        )

    @property
    def num_policy_actions(self) -> int:
        return int(self._num_policy_action)

    @property
    def action_space(self) -> gym.spaces.Box:
        return self._action_space

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {"obs": int(self._policy_obs_dim)}

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        clipped = np.clip(np.asarray(actions, dtype=self._np_dtype), -1.0, 1.0)
        prev_actions = state.info.get("last_actions")
        if prev_actions is not None:
            state.info["prev_actions"] = np.asarray(prev_actions, dtype=self._np_dtype)
        state.info["last_actions"] = clipped
        prev = state.info["prev_ctrl"]
        if self._fix_lifter:
            nxt = np.asarray(prev, dtype=self._np_dtype, copy=True)
            li = int(self._lifter_actuator_idx)
            nxt[:, li] = self._lifter_fixed_ctrl
            if self._left_arm_only:
                ri = self._right_actuator_idx
                nxt[:, ri] = self._right_fixed_ctrl
                active = self._policy_active_actuator_idx
                nxt[:, active] = prev[:, active] + self._cfg.action_scale * clipped
                nxt[:, active] = np.clip(
                    nxt[:, active], self._ctrl_lo[active], self._ctrl_hi[active]
                )
                if self._binary_gripper and self._grasp_finger_actuator_idx is not None:
                    gi = int(self._grasp_finger_actuator_idx)
                    close = clipped[:, -1] > 0.0
                    nxt[:, gi] = np.where(close, self._ctrl_hi[gi], self._ctrl_lo[gi]).astype(
                        self._np_dtype
                    )
            else:
                arm_ix = self._arm_actuator_idx
                nxt[:, arm_ix] = prev[:, arm_ix] + self._cfg.action_scale * clipped
                nxt[:, arm_ix] = np.clip(
                    nxt[:, arm_ix], self._ctrl_lo[arm_ix], self._ctrl_hi[arm_ix]
                )
        else:
            nxt = prev + self._cfg.action_scale * clipped
            nxt = np.clip(nxt, self._ctrl_lo, self._ctrl_hi)
        state.info["prev_ctrl"] = np.asarray(nxt, dtype=self._np_dtype)
        return state.info["prev_ctrl"]

    def _dof_cmd_positions_policy(self) -> np.ndarray:
        full = np.asarray(self._backend.get_dof_pos(), dtype=self._np_dtype)
        return full[:, self._policy_qpos_ix]

    def _grasp_point_w(self, rows: np.ndarray | slice | None = None) -> np.ndarray:
        """End-effector reference point used by reach / grasp / obs.

        Returns the fingertip TCP (midpoint of the two finger bodies) when ``reach_use_tcp`` is
        set, otherwise the wrist ``ee_base_link`` pose. Shape ``(R, 3)``.
        """
        sel = slice(None) if rows is None else rows
        if self._use_tcp and self._tcp_body_ids is not None:
            fp = np.asarray(
                self._backend.get_body_pos_w(self._tcp_body_ids)[sel], dtype=self._np_dtype
            )
            mid = np.asarray(fp.mean(axis=1), dtype=self._np_dtype)
            if self._tcp_has_offset:
                inner_quat = np.asarray(
                    self._backend.get_body_quat_w(self._tcp_body_ids)[sel, 0, :],
                    dtype=self._np_dtype,
                )
                mid = mid + np_quat_apply_batched(inner_quat, self._tcp_offset_local)
            return mid
        return np.asarray(
            self._backend.get_body_pos_w(self._ee_body_ids)[sel, 0, :], dtype=self._np_dtype
        )

    def _assemble_obs_rows(
        self, env_rows: np.ndarray, goal: np.ndarray, last_actions: np.ndarray
    ) -> np.ndarray:
        dof_cmd = self._dof_cmd_positions_policy()[env_rows]
        dof_norm = (dof_cmd - self._ctrl_mid_policy) / (0.5 * self._ctrl_span_policy)
        ee = self._grasp_point_w(env_rows)
        cube = np.asarray(
            self._backend.get_body_pos_w(self._cube_body_ids)[env_rows, 0, :],
            dtype=self._np_dtype,
        )
        return np.concatenate(
            [dof_norm, ee - cube, goal - cube, last_actions], axis=1, dtype=self._np_dtype
        )

    def _compute_obs(self, info: dict[str, Any]) -> dict[str, np.ndarray]:
        rows = np.arange(self._num_envs, dtype=np.intp)
        obs = self._assemble_obs_rows(rows, info["goal_pos"], info["last_actions"])
        return {"obs": obs}

    def update_state(self, state: NpEnvState) -> NpEnvState:
        info = dict(state.info)
        obs = self._compute_obs(info)
        cube_pos = np.asarray(
            self._backend.get_body_pos_w(self._cube_body_ids)[:, 0, :], dtype=self._np_dtype
        )
        cube_z = cube_pos[:, 2]
        fallen = cube_z < self._reward_cfg.fall_z
        if self._reward_cfg.goal_success_mode == "lift3d":
            goal = np.asarray(info["goal_pos"], dtype=self._np_dtype)
            dist_3d = np.linalg.norm(cube_pos - goal, axis=-1)
            lifted = cube_z > (
                self._np_dtype.type(self._reward_cfg.table_z)
                + self._np_dtype.type(self._reward_cfg.lift_margin)
            )
            success = (dist_3d < self._reward_cfg.success_dist) & lifted
        else:
            cube_xy = cube_pos[:, :2]
            goal_xy = info["goal_pos"][:, :2]
            dist_xy = np.linalg.norm(cube_xy - goal_xy, axis=-1)
            lifted = cube_z > self._reward_cfg.lift_z
            success = (dist_xy < self._reward_cfg.success_xy) & lifted
        if self._reward_cfg.terminate_on_success:
            terminated = fallen | success
        else:
            terminated = fallen.copy()
        reward = self._compute_reward(info, fallen, success)
        info["pick_success"] = success.astype(bool)
        info["pick_fallen"] = fallen.astype(bool)
        return state.replace(
            obs=obs,
            reward=reward,
            terminated=terminated.astype(bool),
            info=info,
        )

    def _compute_reward(
        self, info: dict[str, Any], fallen: np.ndarray, success: np.ndarray
    ) -> np.ndarray:
        dtype = self._np_dtype
        out = np.zeros((self._num_envs,), dtype=dtype)
        for name, scale in self._reward_cfg.scales.items():
            if scale == 0.0 or name not in self._reward_fns:
                continue
            out += scale * self._reward_fns[name](info, fallen, success)
        return out * self._cfg.ctrl_dt

    def _reward_reach(
        self, info: dict[str, Any], fallen: np.ndarray, success: np.ndarray
    ) -> np.ndarray:
        del fallen, success
        ee = self._grasp_point_w()
        cube = np.asarray(
            self._backend.get_body_pos_w(self._cube_body_ids)[:, 0, :], dtype=self._np_dtype
        )
        dist = np.linalg.norm(ee - cube, axis=-1)
        return np.exp(-5.0 * dist).astype(self._np_dtype)

    def _reward_place(
        self, info: dict[str, Any], fallen: np.ndarray, success: np.ndarray
    ) -> np.ndarray:
        del fallen, success
        cube = np.asarray(
            self._backend.get_body_pos_w(self._cube_body_ids)[:, 0, :], dtype=self._np_dtype
        )
        goal = info["goal_pos"]
        dist = np.linalg.norm(cube - goal, axis=-1)
        place = np.exp(-8.0 * dist).astype(self._np_dtype)
        if self._reward_cfg.place_use_grasp_gate:
            place = place * self._grasp_gate(info)
        return place

    def _finger_closure(self, info: dict[str, Any]) -> np.ndarray:
        """[0,1] gripper closure derived from the commanded finger ctrl (0=open, 1=closed)."""
        n = self._num_envs
        if self._grasp_finger_actuator_idx is None:
            return np.zeros((n,), dtype=self._np_dtype)
        ji = int(self._grasp_finger_actuator_idx)
        prev = np.asarray(info["prev_ctrl"], dtype=self._np_dtype)
        f = prev[:, ji]
        lo = self._np_dtype.type(self._ctrl_lo[ji])
        hi = self._np_dtype.type(self._ctrl_hi[ji])
        span = (hi - lo) + self._np_dtype.type(1e-08)
        return np.clip((f - lo) / span, 0.0, 1.0).astype(self._np_dtype)

    def _grasp_gate(self, info: dict[str, Any]) -> np.ndarray:
        """[0,1] gate: finger closure × proximity of EE to cube (same as unscaled ``grasp`` term)."""
        n = self._num_envs
        if self._grasp_finger_actuator_idx is None:
            return np.ones((n,), dtype=self._np_dtype)
        closure = self._finger_closure(info)
        ee = self._grasp_point_w()
        cube = np.asarray(
            self._backend.get_body_pos_w(self._cube_body_ids)[:, 0, :], dtype=self._np_dtype
        )
        dist = np.linalg.norm(ee - cube, axis=-1)
        k = self._np_dtype.type(self._reward_cfg.grasp_proximity_decay)
        prox = np.exp(-k * dist).astype(self._np_dtype)
        return (closure * prox).astype(self._np_dtype)

    def _reward_action(
        self, info: dict[str, Any], fallen: np.ndarray, success: np.ndarray
    ) -> np.ndarray:
        del fallen, success
        a = info["last_actions"]
        return -np.sum(np.square(a), axis=-1).astype(self._np_dtype)

    def _reward_approach(
        self, info: dict[str, Any], fallen: np.ndarray, success: np.ndarray
    ) -> np.ndarray:
        """[0,1] reward for hovering above the cube with the gripper still open.

        Peaks when the TCP is aligned in xy with the cube and at ``pregrasp_h``
        above it, scaled by how open the gripper is (``1 - closure``). Encourages
        the deliberate "go above + keep open" phase before descending to grasp.
        """
        del fallen, success
        ee = self._grasp_point_w()
        cube = np.asarray(
            self._backend.get_body_pos_w(self._cube_body_ids)[:, 0, :], dtype=self._np_dtype
        )
        xy = np.linalg.norm(ee[:, :2] - cube[:, :2], axis=-1)
        pregrasp_z = cube[:, 2] + self._np_dtype.type(self._reward_cfg.pregrasp_h)
        dz = np.abs(ee[:, 2] - pregrasp_z)
        kx = self._np_dtype.type(self._reward_cfg.approach_xy_decay)
        kz = self._np_dtype.type(self._reward_cfg.approach_z_decay)
        openness = (1.0 - self._finger_closure(info)).astype(self._np_dtype)
        return (np.exp(-kx * xy) * np.exp(-kz * dz) * openness).astype(self._np_dtype)

    def _reward_premature_close(
        self, info: dict[str, Any], fallen: np.ndarray, success: np.ndarray
    ) -> np.ndarray:
        """Positive penalty magnitude for closing the gripper while far (in xy) from the cube.

        Returns ``closure × (1 - exp(-k·xy))`` in [0,1]; pair with a negative scale.
        Discourages grabbing at air / closing en route instead of over the cube.
        """
        del fallen, success
        ee = self._grasp_point_w()
        cube = np.asarray(
            self._backend.get_body_pos_w(self._cube_body_ids)[:, 0, :], dtype=self._np_dtype
        )
        xy = np.linalg.norm(ee[:, :2] - cube[:, :2], axis=-1)
        k = self._np_dtype.type(self._reward_cfg.premature_close_decay)
        far = (1.0 - np.exp(-k * xy)).astype(self._np_dtype)
        closure = self._finger_closure(info)
        return (closure * far).astype(self._np_dtype)

    def _reward_firm_grasp(
        self, info: dict[str, Any], fallen: np.ndarray, success: np.ndarray
    ) -> np.ndarray:
        """[0,1] reward for a *firm* grasp: high finger closure while the cube is
        lifted and the TCP is on it.

        ``closure × lift_gate × proximity_gate``. The lift gate means this only
        pays once the cube is actually off the table, so it rewards "hold the
        cube tightly while carrying it" rather than clamping the jaws at rest --
        it cannot create a low/no-lift "clamp and sit" optimum. Use to push the
        policy away from the loose tip-cradle grasp toward genuine pinch force.
        """
        del fallen, success
        closure = self._finger_closure(info)
        return (closure * self._lift_gate_binary() * self._lift_proximity_gate()).astype(
            self._np_dtype
        )

    def _reward_action_rate(
        self, info: dict[str, Any], fallen: np.ndarray, success: np.ndarray
    ) -> np.ndarray:
        """Positive penalty magnitude = squared change between consecutive actions.

        Returns ``sum((a_t - a_{t-1})^2)``; pair with a negative scale to smooth motion.
        """
        del fallen, success
        a = np.asarray(info["last_actions"], dtype=self._np_dtype)
        prev = info.get("prev_actions")
        if prev is None:
            return np.zeros((self._num_envs,), dtype=self._np_dtype)
        prev = np.asarray(prev, dtype=self._np_dtype)
        return np.sum(np.square(a - prev), axis=-1).astype(self._np_dtype)

    def _reward_drop(
        self, info: dict[str, Any], fallen: np.ndarray, success: np.ndarray
    ) -> np.ndarray:
        del info, success
        return fallen.astype(self._np_dtype)

    def _reward_lift(
        self, info: dict[str, Any], fallen: np.ndarray, success: np.ndarray
    ) -> np.ndarray:
        del fallen, success
        cube_z = np.asarray(
            self._backend.get_body_pos_w(self._cube_body_ids)[:, 0, 2], dtype=self._np_dtype
        )
        span = max(float(self._reward_cfg.lift_shaping_span), 1e-06)
        h = np.clip(cube_z - self._np_dtype.type(self._reward_cfg.table_z), 0.0, span)
        progress = (h / span).astype(self._np_dtype)
        mode = self._reward_cfg.lift_gate_mode
        if mode is None:
            mode = "grasp" if self._reward_cfg.lift_use_grasp_gate else "none"
        if mode == "grasp":
            progress = (progress * self._grasp_gate(info)).astype(self._np_dtype)
            return progress
        if mode == "proximity":
            progress = (progress * self._lift_proximity_gate()).astype(self._np_dtype)
            return progress
        if mode != "none":
            raise ValueError(
                f"lift_gate_mode must be one of 'grasp'/'proximity'/'none', got {mode!r}"
            )
        return progress

    def _lift_proximity_gate(self) -> np.ndarray:
        """[0,1] gate = exp(-grasp_proximity_decay · ||tcp - cube||); robust to closure wobble."""
        ee = self._grasp_point_w()
        cube = np.asarray(
            self._backend.get_body_pos_w(self._cube_body_ids)[:, 0, :], dtype=self._np_dtype
        )
        dist = np.linalg.norm(ee - cube, axis=-1)
        k = self._np_dtype.type(self._reward_cfg.grasp_proximity_decay)
        return np.exp(-k * dist).astype(self._np_dtype)

    def _reward_grasp(
        self, info: dict[str, Any], fallen: np.ndarray, success: np.ndarray
    ) -> np.ndarray:
        del fallen, success
        return self._grasp_gate(info)

    def _reward_settle(
        self, info: dict[str, Any], fallen: np.ndarray, success: np.ndarray
    ) -> np.ndarray:
        """Encourage low arm joint speed when the cube is near the goal (optional lift gate)."""
        del fallen, success
        n = self._num_envs
        if self._settle_dof_vel_ix is None:
            return np.zeros((n,), dtype=self._np_dtype)
        dof_vel = np.asarray(self._backend.get_dof_vel(), dtype=self._np_dtype)
        v = dof_vel[:, self._settle_dof_vel_ix]
        vel_pen = np.sum(v * v, axis=-1)
        c = self._np_dtype.type(self._reward_cfg.settle_arm_vel_coeff)
        exp_still = np.exp(-c * vel_pen).astype(self._np_dtype)
        cube = np.asarray(
            self._backend.get_body_pos_w(self._cube_body_ids)[:, 0, :], dtype=self._np_dtype
        )
        goal = np.asarray(info["goal_pos"], dtype=self._np_dtype)
        dist_xy = np.linalg.norm(cube[:, :2] - goal[:, :2], axis=-1)
        kxy = self._np_dtype.type(self._reward_cfg.settle_xy_decay)
        gate_xy = np.exp(-kxy * dist_xy).astype(self._np_dtype)
        if self._reward_cfg.settle_use_lift_gate:
            span = max(float(self._reward_cfg.lift_shaping_span), 1e-06)
            h = np.clip(
                cube[:, 2] - self._np_dtype.type(self._reward_cfg.table_z), 0.0, span
            ).astype(self._np_dtype)
            lift_gate = (h / self._np_dtype.type(span)).astype(self._np_dtype)
            gate = (gate_xy * lift_gate).astype(self._np_dtype)
        else:
            gate = gate_xy
        return (gate * exp_still).astype(self._np_dtype)

    def _reward_success(
        self, info: dict[str, Any], fallen: np.ndarray, success: np.ndarray
    ) -> np.ndarray:
        del info, fallen
        return success.astype(self._np_dtype)

    def _reward_hold_success(
        self, info: dict[str, Any], fallen: np.ndarray, success: np.ndarray
    ) -> np.ndarray:
        """Per-step bonus when ``pick_success`` is true (eval-aligned hold-at-goal)."""
        del info, fallen
        return success.astype(self._np_dtype)

    def _lift_gate_binary(self) -> np.ndarray:
        """1.0 when the cube is genuinely lifted above ``table_z + lift_margin`` (else 0.0)."""
        cube_z = np.asarray(
            self._backend.get_body_pos_w(self._cube_body_ids)[:, 0, 2], dtype=self._np_dtype
        )
        thresh = self._np_dtype.type(self._reward_cfg.table_z) + self._np_dtype.type(
            self._reward_cfg.lift_margin
        )
        return (cube_z > thresh).astype(self._np_dtype)

    def _reward_lift_bonus(
        self, info: dict[str, Any], fallen: np.ndarray, success: np.ndarray
    ) -> np.ndarray:
        """Binary bonus for lifting the cube clear of the table (gated ``object_is_lifted``)."""
        del info, fallen, success
        return self._lift_gate_binary()

    def _goal_track(self, info: dict[str, Any], std: float) -> np.ndarray:
        """Lift-gated 3D goal tracking: ``1[lifted] * (1 - tanh(||cube - goal||_3D / std))``."""
        cube = np.asarray(
            self._backend.get_body_pos_w(self._cube_body_ids)[:, 0, :], dtype=self._np_dtype
        )
        goal = np.asarray(info["goal_pos"], dtype=self._np_dtype)
        dist = np.linalg.norm(cube - goal, axis=-1)
        s = max(float(std), 1e-06)
        track = (1.0 - np.tanh(dist / s)).astype(self._np_dtype)
        return (self._lift_gate_binary() * track).astype(self._np_dtype)

    def _reward_goal_track_coarse(
        self, info: dict[str, Any], fallen: np.ndarray, success: np.ndarray
    ) -> np.ndarray:
        del fallen, success
        return self._goal_track(info, self._reward_cfg.goal_track_coarse_std)

    def _reward_goal_track_fine(
        self, info: dict[str, Any], fallen: np.ndarray, success: np.ndarray
    ) -> np.ndarray:
        del fallen, success
        return self._goal_track(info, self._reward_cfg.goal_track_fine_std)
