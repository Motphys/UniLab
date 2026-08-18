from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np

from unilab.algos.torch.bfm.bfm_obs import BfmObsBuilder, add_extend_bodies, compute_max_local_self
from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.backend import create_backend
from unilab.base.scene import SceneCfg
from unilab.dr.provider import DomainRandomizationProvider
from unilab.dr.types import ResetPlan
from unilab.envs.locomotion.g1.base import G1BaseCfg, G1BaseEnv

if TYPE_CHECKING:
    from unilab.algos.torch.bfm.motion_data import MotionBank


class _BfmResetProvider(DomainRandomizationProvider):
    def validate(self, env, capabilities) -> None:
        return None

    def build_reset_plan(self, env, env_ids) -> ResetPlan:
        n = len(env_ids)
        qpos = np.tile(env._init_qpos, (n, 1)).astype(env._init_qpos.dtype)
        qvel = np.tile(env._init_qvel, (n, 1)).astype(env._init_qvel.dtype)
        info_updates = {
            "current_actions": np.zeros((n, env._num_action), np.float32),
            "last_actions": np.zeros((n, env._num_action), np.float32),
        }
        return ResetPlan(
            env_ids=env_ids, qpos=qpos, qvel=qvel, info_updates=info_updates, randomization=None
        )

    def build_reset_observation(self, env, env_ids, info_updates) -> dict[str, np.ndarray]:
        return env._reset_obs(np.asarray(env_ids))  # type: ignore[no-any-return]


class _MotionResetProvider(DomainRandomizationProvider):
    def validate(self, env, capabilities) -> None:
        return None

    def build_reset_plan(self, env, env_ids) -> ResetPlan:
        n = len(env_ids)
        qpos, qvel = env._motion_bank.sample_reset(
            n, lie_down_prob=float(env._cfg.lie_down_init_prob), rng=env._reset_rng
        )
        info_updates = {
            "current_actions": np.zeros((n, env._num_action), np.float32),
            "last_actions": np.zeros((n, env._num_action), np.float32),
        }
        return ResetPlan(
            env_ids=env_ids,
            qpos=qpos.astype(env._init_qpos.dtype),
            qvel=qvel.astype(env._init_qvel.dtype),
            info_updates=info_updates,
            randomization=None,
        )

    def build_reset_observation(self, env, env_ids, info_updates) -> dict[str, np.ndarray]:
        return env._reset_obs(np.asarray(env_ids))  # type: ignore[no-any-return]


BODY_NAMES = (
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
)
FOOT_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")

EXTEND_BODIES = (("torso_link", (0.0, 0.0, 0.35)),)

AUX_REWARD_KEYS = (
    "penalty_torques",
    "penalty_action_rate",
    "limits_dof_pos",
    "limits_torque",
    "penalty_undesired_contact",
    "penalty_feet_ori",
    "penalty_ankle_roll",
    "penalty_slippage",
)

OBS_NOISE_SCALES = {
    "dof_pos": 0.01,
    "dof_vel": 0.5,
    "base_ang_vel": 0.2,
    "projected_gravity": 0.05,
}


def _wxyz2xyzw(q: np.ndarray) -> np.ndarray:
    return np.concatenate([q[..., 1:4], q[..., 0:1]], axis=-1)


@registry.envcfg("G1Bfm")
@dataclass
class G1BfmCfg(G1BaseCfg):
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat_bfm.xml")
        )
    )
    body_names: tuple[str, ...] = field(default_factory=lambda: BODY_NAMES)
    foot_names: tuple[str, ...] = field(default_factory=lambda: FOOT_NAMES)
    root_height_obs: bool = True

    sim_dt: float = 0.005
    max_episode_seconds: float = 10.0

    lie_down_init_prob: float = 0.3
    lafan_pkl: str | None = None
    max_motions: int | None = None


@registry.env("G1Bfm", sim_backend="mujoco")
@registry.env("G1Bfm", sim_backend="motrix")
class G1BfmEnv(G1BaseEnv):
    _cfg: G1BfmCfg
    _keyframe_name = "stand"
    _motion_bank: "MotionBank | None"

    def __init__(self, cfg: G1BfmCfg, num_envs: int = 1, backend_type: str = "mujoco"):

        if backend_type not in {"mujoco", "motrix"}:
            raise ValueError(
                f"G1Bfm supports only the mujoco and motrix backends, got {backend_type!r}"
            )

        penalised = [
            n for n in cfg.body_names if any(s in n for s in ("pelvis", "shoulder", "hip"))
        ]
        contact_bodies = list(dict.fromkeys(list(cfg.foot_names) + penalised))
        backend = create_backend(
            backend_type,
            cfg.scene,
            num_envs,
            cfg.sim_dt,
            base_name=cfg.asset.base_name,
            add_body_sensors=True,
            contact_sensor_bodies=contact_bodies,
        )
        self._contact_bodies = contact_bodies
        super().__init__(cfg, backend, num_envs)

        import os

        if cfg.lafan_pkl and os.path.exists(cfg.lafan_pkl):
            from unilab.algos.torch.bfm.motion_data import MotionBank

            self._motion_bank = MotionBank(
                cfg.lafan_pkl,
                num_dof=self._num_action,
                max_motions=cfg.max_motions,
                ctrl_dt=float(cfg.ctrl_dt),
            )
            self._init_domain_randomization(_MotionResetProvider())
        else:
            self._init_domain_randomization(_BfmResetProvider())

    def _init_buffers(self) -> None:
        super()._init_buffers()
        self._body_ids = self._backend.get_body_ids(list(self._cfg.body_names))
        self._foot_ids = self._backend.get_body_ids(list(self._cfg.foot_names))

        self._extends = [
            (self._cfg.body_names.index(pn), np.asarray(off, np.float32))
            for pn, off in EXTEND_BODIES
        ]
        self._obs_builder = BfmObsBuilder(num_envs=self.num_envs, num_dof=self._num_action)
        self._reset_rng = np.random.RandomState(0)

        self._add_obs_noise = False
        self._obs_noise_scales = OBS_NOISE_SCALES
        self._noise_rng = np.random.RandomState(1)
        self._last_action = np.zeros((self.num_envs, self._num_action), np.float32)
        self._prev_action = np.zeros((self.num_envs, self._num_action), np.float32)

        jr = self._backend.get_dof_pos_limits()
        self._dof_lower = jr[:, 0].astype(np.float32)
        self._dof_upper = jr[:, 1].astype(np.float32)
        self._default_dof = self.default_angles.astype(np.float32)

        kp, kd = self._backend.get_actuator_gains()
        self._kp = np.asarray(kp, np.float32)
        self._kd = np.asarray(kd, np.float32)
        self._torque_limits = np.abs(
            self._backend.get_actuator_force_range()[:, 1].astype(np.float32)
        )

        self._act_rescale = (self._torque_limits / self._kp).astype(np.float32)

        _wrist_idx = self._backend.get_joint_dof_pos_indices(
            (
                "left_wrist_pitch_joint",
                "left_wrist_yaw_joint",
                "right_wrist_pitch_joint",
                "right_wrist_yaw_joint",
            )
        )
        _GOLDEN_WRIST_RESCALE = 5.0 / 16.77833
        _got = self._act_rescale[np.asarray(_wrist_idx)]
        if not np.allclose(_got, _GOLDEN_WRIST_RESCALE, atol=1e-3):
            raise ValueError(
                "BFM wrist actuator rescale drifted from golden "
                f"{_GOLDEN_WRIST_RESCALE:.4f} (effort 5.0 / kp 16.77833); got {_got}. "
                "Check g1_bfm.xml wrist_pitch/yaw kp+forcerange."
            )

        _GOLDEN_WRIST_KV = 1.06814
        _got_kv = self._kd[np.asarray(_wrist_idx)]
        if not np.allclose(_got_kv, _GOLDEN_WRIST_KV, atol=1e-3):
            raise ValueError(
                f"BFM wrist actuator damping drifted from golden {_GOLDEN_WRIST_KV} "
                f"(g1_29dof_hard_waist damping wrist_pitch/yaw); got {_got_kv}. "
                "Check g1_bfm.xml wrist_pitch/yaw kv."
            )
        self._soft_dof_pos_limit = 0.95
        self._soft_torque_limit = 0.95

        self._ankle_roll_idx = self._backend.get_joint_dof_pos_indices(
            ("left_ankle_roll_joint", "right_ankle_roll_joint")
        )

        self._penalised_body_local = [
            i
            for i, n in enumerate(self._cfg.body_names)
            if any(s in n for s in ("pelvis", "shoulder", "hip"))
        ]

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        n = self._num_action
        n_body = len(self._cfg.body_names) + len(EXTEND_BODIES)

        max_self = 1 + (n_body * 3 - 3) + n_body * 6 + n_body * 3 + n_body * 3
        spec = {
            "state": 2 * n + 6,
            "last_action": n,
            "privileged_state": max_self,
            "history_actor": 4 * (3 + 3 + n + n + n),
        }
        if os.environ.get("UNILAB_BFM_B_INPUT", "default") in ("novel", "novel_only"):
            spec["state_novel"] = n + 6
        return spec

    def _projected_gravity(self) -> np.ndarray:
        base_quat = _wxyz2xyzw(self._backend.get_base_quat())
        g = np.array([0.0, 0.0, -1.0], np.float32)[None, :].repeat(self.num_envs, 0)
        from unilab.algos.torch.bfm.bfm_obs import quat_rotate_inverse

        return quat_rotate_inverse(base_quat, g)

    def _gather(self):
        dof_pos = self._backend.get_dof_pos() - self._default_dof
        dof_vel = self._backend.get_dof_vel()

        base_ang_vel = self._backend.get_base_ang_vel_b()
        proj_g = self._projected_gravity()
        bpos = self._backend.get_body_pos_w(self._body_ids)
        bquat = _wxyz2xyzw(self._backend.get_body_quat_w(self._body_ids))
        bvel = self._backend.get_body_lin_vel_w(self._body_ids)
        bang = self._backend.get_body_ang_vel_w(self._body_ids)
        bpos, bquat, bvel, bang = add_extend_bodies(bpos, bquat, bvel, bang, self._extends)

        foot_local = [self._cfg.body_names.index(nm) for nm in self._cfg.foot_names]
        fspeed = np.linalg.norm(bvel[:, foot_local], axis=-1)
        fheight = bpos[:, foot_local, 2]
        contact = ((fspeed < 0.4) & (fheight < 0.07)).astype(np.float32)
        max_self = compute_max_local_self(
            bpos, bquat, bvel, bang, contact, self._cfg.root_height_obs
        )
        return dof_pos, dof_vel, base_ang_vel, proj_g, max_self

    def _add_noise(self, x: np.ndarray, key: str) -> np.ndarray:

        return x + (
            self._noise_rng.uniform(-1.0, 1.0, size=x.shape).astype(np.float32)
            * self._obs_noise_scales[key]
        )

    def _compute_obs(self, push: bool = True) -> dict[str, np.ndarray]:
        dof_pos, dof_vel, base_ang_vel, proj_g, max_self = self._gather()
        if self._add_obs_noise:
            dof_pos = self._add_noise(dof_pos, "dof_pos")
            dof_vel = self._add_noise(dof_vel, "dof_vel")
            base_ang_vel = self._add_noise(base_ang_vel, "base_ang_vel")
            proj_g = self._add_noise(proj_g, "projected_gravity")
        if push:
            self._obs_builder.push(base_ang_vel, proj_g, dof_pos, dof_vel, self._last_action)
        out = self._obs_builder.obs(
            dof_pos, dof_vel, proj_g, base_ang_vel, self._last_action, max_self
        )
        if os.environ.get("UNILAB_BFM_B_INPUT", "default") in ("novel", "novel_only"):
            n = self._num_action
            s = out["state"]
            out["state_novel"] = np.concatenate(
                [s[:, :n], s[:, 2 * n : 2 * n + 6]], axis=-1
            ).astype(np.float32)
        return out

    def _reset_obs(self, env_ids: np.ndarray) -> dict[str, np.ndarray]:
        self._obs_builder.reset_envs(env_ids)
        self._last_action[env_ids] = 0.0
        self._prev_action[env_ids] = 0.0
        full = self._compute_obs(push=False)
        return {k: v[env_ids] for k, v in full.items()}

    CONTACT_FORCE_THRESHOLD = 1.0

    def _touch(self, body_name: str) -> np.ndarray:
        return np.asarray(self._backend.get_body_contact_force(body_name), np.float32)

    def _foot_contact(self) -> np.ndarray:

        return np.stack(
            [
                (self._touch(n) > self.CONTACT_FORCE_THRESHOLD).astype(np.float32)
                for n in self._cfg.foot_names
            ],
            axis=-1,
        )

    def _aux_rewards(self) -> dict[str, np.ndarray]:
        from unilab.algos.torch.bfm.bfm_obs import quat_rotate_inverse

        dof_pos = self._backend.get_dof_pos()
        dof_vel = self._backend.get_dof_vel()

        target = (
            self._last_action * self._cfg.control_config.action_scale * self._act_rescale
            + self._default_dof
        )
        torque = self._kp * (target - dof_pos) - self._kd * dof_vel

        torque = np.clip(torque, -self._torque_limits, self._torque_limits)
        foot_contact = self._foot_contact()
        out = {}

        out["penalty_torques"] = np.sum(torque**2, axis=-1)

        out["penalty_action_rate"] = np.sum((self._last_action - self._prev_action) ** 2, axis=-1)

        m = 0.5 * (self._dof_lower + self._dof_upper)
        r = self._dof_upper - self._dof_lower
        lo = m - 0.5 * r * self._soft_dof_pos_limit
        hi = m + 0.5 * r * self._soft_dof_pos_limit
        out["limits_dof_pos"] = np.sum(
            np.clip(lo - dof_pos, 0, None) + np.clip(dof_pos - hi, 0, None), axis=-1
        )

        out["limits_torque"] = np.sum(
            np.clip(np.abs(torque) - self._torque_limits * self._soft_torque_limit, 0, None),
            axis=-1,
        )

        pen = [n for n in self._contact_bodies if n not in self._cfg.foot_names]
        if pen:
            pc = np.stack([self._touch(n) > self.CONTACT_FORCE_THRESHOLD for n in pen], axis=-1)
            out["penalty_undesired_contact"] = np.any(pc, axis=-1).astype(np.float32)
        else:
            out["penalty_undesired_contact"] = np.zeros(self.num_envs, np.float32)

        fquat = _wxyz2xyzw(self._backend.get_body_quat_w(self._foot_ids))
        g = (
            np.array([0, 0, -1.0], np.float32)[None, None, :]
            .repeat(self.num_envs, 0)
            .repeat(len(self._foot_ids), 1)
        )
        foot_grav = quat_rotate_inverse(fquat.reshape(-1, 4), g.reshape(-1, 3)).reshape(
            self.num_envs, len(self._foot_ids), 3
        )
        feet_ori = np.sqrt(np.sum(foot_grav[..., :2] ** 2, axis=-1)) * foot_contact
        out["penalty_feet_ori"] = np.sum(feet_ori, axis=-1)

        out["penalty_ankle_roll"] = np.sum(dof_pos[:, self._ankle_roll_idx] ** 2, axis=-1)

        fvel = self._backend.get_body_lin_vel_w(self._foot_ids)
        out["penalty_slippage"] = np.sum(np.linalg.norm(fvel, axis=-1) * foot_contact, axis=-1)
        return {k: out[k].astype(np.float32) for k in AUX_REWARD_KEYS}

    NORMALIZE_ACTION_TO = 5.0
    ACTION_CLIP = 5.0

    def _init_action_space(self) -> None:
        ctrl_range = np.asarray(self._backend.get_actuator_ctrl_range(), dtype=np.float64)
        ctrl_range = np.where(np.isfinite(ctrl_range), ctrl_range, 0.0)
        num_actuators = self._backend.num_actuators
        self._action_space = gym.spaces.Box(
            ctrl_range[:, 0], ctrl_range[:, 1], (num_actuators,), dtype=float
        )

    def apply_action(self, actions: np.ndarray, state):
        self._prev_action = self._last_action

        a = np.clip(
            np.asarray(actions, np.float32) * self.NORMALIZE_ACTION_TO,
            -self.ACTION_CLIP,
            self.ACTION_CLIP,
        )
        self._last_action = a

        return super().apply_action(a * self._act_rescale, state)

    def update_state(self, state):
        obs = self._compute_obs()
        info = dict(state.info)
        info["aux_rewards"] = self._aux_rewards()
        info["qpos"] = self._backend.get_physics_state()[:, : 7 + self._num_action]
        terminated = self._compute_terminated(state)
        return state.replace(
            obs=obs, reward=np.zeros(self.num_envs, np.float32), terminated=terminated, info=info
        )

    def _compute_terminated(self, state) -> np.ndarray:

        return np.zeros(self.num_envs, dtype=bool)
