"""G1 Whole-Body Tracking — sim2real-oriented SAC variant (task ``G1WBTObs``).

This module registers a strict subclass of :class:`G1MotionTrackingSACEnv` that
adds the training-pipeline pieces needed for ONNX-on-real-G1 deployment:

* drop deploy-unavailable channels from the actor obs
  (``base_lin_vel``, ``motion_anchor_pos_b``);
* per-step uniform noise on ``motion_anchor_ori_b`` (actor only);
* proprio observation history (``gyro`` / ``joint_pos_rel`` / ``dof_vel`` /
  ``last_actions``) flattened oldest-first per term, matching the deploy-side
  ``ObservationManager`` when ``use_gym_history=false``;
* per-episode encoder bias on ``joint_pos_rel`` (actor only);
* per-episode foot-geom friction sampled across regex-matched geoms;
* per-episode y / z COM offsets layered on top of the existing x offset;
* ``joint_acc_l2`` and ``joint_torque_l2`` reward terms.

All extensions are gated by flags on :class:`G1WBTObsCfg`; the bases
(``G1MotionTrackingSACCfg`` / ``G1MotionTrackingSACEnv`` /
``G1MotionTrackingEnv``) are untouched. Switch the pelvis IMU via the
yaml ``env.sensor.gyro``/``env.sensor.upvector``/``env.sensor.local_linvel``
fields (no XML duplication required — ``g1.xml`` already exposes both IMUs).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from unilab.base import registry
from unilab.dr import (
    DomainRandomizationCapabilities,
    ResetPlan,
    ResetRandomizationPayload,
)
from unilab.dr.dr_utils import (
    build_common_reset_randomization,
    zero_actions,
)
from unilab.dr.types import RESET_TERM_GEOM_FRICTION
from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.g1.base import NoiseConfig

from .tracking import (
    Domain_Rand,
    G1MotionTrackingDomainRandomizationProvider,
    _build_motion_reference_state,
)
from .tracking_sac import G1MotionTrackingSACCfg, G1MotionTrackingSACEnv

# --------------------------------------------------------------------------- #
# Config extensions
# --------------------------------------------------------------------------- #


@dataclass
class ObsNoiseConfig(NoiseConfig):
    """Actor obs masking flags + proprio history depth.

    Defaults preserve baseline behaviour so the parent ``NoiseConfig`` remains
    a drop-in replacement; ``G1WBTObs`` flips the flags via its task yaml.
    """

    # Drop ``base_lin_vel`` from actor obs (G1 has no on-robot linvel sensor).
    enable_zero_linvel: bool = False
    # Drop ``motion_anchor_pos_b`` from actor obs (no torso-pose estimator).
    enable_zero_anchor_pos: bool = False
    # Per-step uniform noise on ``motion_anchor_ori_b`` (actor only).
    enable_anchor_ori_noise: bool = False
    scale_anchor_ori: float = 0.05
    # When > 1, proprio terms (gyro / joint_pos_rel / dof_vel / last_actions)
    # are flattened oldest-first as an H-step history block. Reference terms
    # stay single-step. Critic stays single-step. Mirrors deploy-side
    # ``ObservationManager`` with ``use_gym_history=false``.
    obs_history_length: int = 1


@dataclass
class ObsDomainRand(Domain_Rand):
    """y / z COM offsets, per-episode encoder bias, foot-geom friction."""

    randomize_com_y: bool = False
    com_offset_y: list[float] = field(default_factory=lambda: [-0.05, 0.05])
    randomize_com_z: bool = False
    com_offset_z: list[float] = field(default_factory=lambda: [-0.05, 0.05])

    # Per-episode additive bias on actor's joint_pos channel.
    enable_encoder_bias: bool = False
    encoder_bias_range: list[float] = field(default_factory=lambda: [-0.01, 0.01])

    # Per-reset foot-geom friction. ``shared_random=True`` — a single scalar
    # is broadcast across all foot geoms of one env, applied to the
    # sliding-friction column. Matches mjlab.
    randomize_geom_friction: bool = False
    friction_range: list[float] = field(default_factory=lambda: [0.3, 1.2])
    friction_geom_pattern: str = r"^(left|right)_foot[1-7]_collision$"


@registry.envcfg("G1WBTObs")
@dataclass
class G1WBTObsCfg(G1MotionTrackingSACCfg):
    """SAC whole-body tracking with sim2real obs / DR / reward extensions."""

    noise_config: ObsNoiseConfig = field(default_factory=ObsNoiseConfig)  # type: ignore[assignment]
    domain_rand: ObsDomainRand = field(default_factory=ObsDomainRand)  # type: ignore[assignment]


# --------------------------------------------------------------------------- #
# DR provider extension
# --------------------------------------------------------------------------- #


class G1WBTObsDomainRandomizationProvider(G1MotionTrackingDomainRandomizationProvider):
    """Extends the SAC tracking DR provider with encoder bias, foot-geom
    friction, y/z COM offsets, and post-reset ``prev_dof_vel`` seeding."""

    def __init__(
        self,
        *,
        base_kp: np.ndarray | None = None,
        base_kd: np.ndarray | None = None,
        base_geom_friction: np.ndarray | None = None,
        foot_geom_ids: np.ndarray | None = None,
    ) -> None:
        super().__init__(base_kp=base_kp, base_kd=base_kd)
        self._base_geom_friction = base_geom_friction
        self._foot_geom_ids = foot_geom_ids

    def validate(self, env: Any, capabilities: DomainRandomizationCapabilities) -> None:
        super().validate(env, capabilities)
        if not getattr(env.cfg.domain_rand, "randomize_geom_friction", False):
            return
        if not capabilities.supports_reset_term(RESET_TERM_GEOM_FRICTION):
            raise NotImplementedError(
                f"{env._backend.backend_type} backend does not support "
                "geom-friction reset randomization"
            )
        if (
            self._base_geom_friction is None
            or self._foot_geom_ids is None
            or self._foot_geom_ids.size == 0
        ):
            raise ValueError("randomize_geom_friction=True but provider has no foot geom IDs")

    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        num_reset = len(env_ids)
        motion_frames = env.motion_sampler.sample_frames(env_ids)
        motion_data = env.motion_loader.get_motion_at_frame(motion_frames)
        qpos, qvel = _build_motion_reference_state(env, env_ids, motion_data)

        info_updates: dict[str, Any] = {
            "current_actions": zero_actions(num_reset, env._num_action),
            "last_actions": zero_actions(num_reset, env._num_action),
            # Seed prev_dof_vel with the post-reset joint velocity so the first
            # joint_acc_l2 sample is physically meaningful (Δv from the new
            # starting velocity, not a spurious step from pre-termination).
            "prev_dof_vel": qvel[:, 6:].astype(get_global_dtype()),
        }

        dr_cfg = env.cfg.domain_rand
        if getattr(dr_cfg, "enable_encoder_bias", False):
            low, high = dr_cfg.encoder_bias_range
            info_updates["joint_pos_obs_bias"] = np.random.uniform(
                low, high, size=(num_reset, env._num_action)
            ).astype(get_global_dtype())

        randomization = build_common_reset_randomization(
            env, num_reset, base_kp=self._base_kp, base_kd=self._base_kd
        )

        # Foot-geom friction.
        if getattr(dr_cfg, "randomize_geom_friction", False):
            assert self._base_geom_friction is not None
            assert self._foot_geom_ids is not None
            payload = randomization or ResetRandomizationPayload()
            low, high = dr_cfg.friction_range
            scale = np.random.uniform(low, high, size=(num_reset, 1)).astype(np.float64)
            geom_friction = np.broadcast_to(
                self._base_geom_friction,
                (num_reset, *self._base_geom_friction.shape),
            ).copy()
            geom_friction[:, self._foot_geom_ids, 0] = scale * np.ones(
                (1, self._foot_geom_ids.size)
            )
            payload.geom_friction = geom_friction
            randomization = payload

        # y / z COM offsets, layered on top of parent's x-only common build.
        has_com_y = getattr(dr_cfg, "randomize_com_y", False)
        has_com_z = getattr(dr_cfg, "randomize_com_z", False)
        if has_com_y or has_com_z:
            payload = randomization or ResetRandomizationPayload()
            com_offset = payload.base_com_offset
            if com_offset is None:
                com_offset = np.zeros((num_reset, 3), dtype=np.float64)
            if has_com_y:
                low, high = dr_cfg.com_offset_y
                com_offset[:, 1] = np.random.uniform(low, high, size=(num_reset,))
            if has_com_z:
                low, high = dr_cfg.com_offset_z
                com_offset[:, 2] = np.random.uniform(low, high, size=(num_reset,))
            payload.base_com_offset = com_offset
            randomization = payload

        return ResetPlan(
            env_ids=env_ids,
            qpos=qpos,
            qvel=qvel,
            info_updates=info_updates,
            randomization=randomization,
        )


# --------------------------------------------------------------------------- #
# Env
# --------------------------------------------------------------------------- #


@registry.env("G1WBTObs", sim_backend="mujoco")
@registry.env("G1WBTObs", sim_backend="motrix")
class G1WBTObsEnv(G1MotionTrackingSACEnv):
    """SAC WBT with deploy-aligned obs, proprio history, and extra DR/rewards.

    All extensions live in this subclass — base classes are untouched. Flags
    on ``G1WBTObsCfg`` are toggled from the task yaml.
    """

    _cfg: G1WBTObsCfg

    def __init__(self, cfg: G1WBTObsCfg, num_envs: int = 1, backend_type: str = "mujoco"):
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)

        # Cache base actuator gains for joint_torque_l2.
        # Position-control torque approx: τ ≈ kp·(target_q − q) − kd·qd.
        # DR (kp/kd ±10–15%) leaves small error vs true per-env torque, but
        # the gradient direction (penalise large action / large Δq) is preserved.
        base_kp, base_kd = self._backend.get_actuator_gains()
        self._base_kp = np.asarray(base_kp, dtype=get_global_dtype())
        self._base_kd = np.asarray(base_kd, dtype=get_global_dtype())

        # Proprio history buffers — per-term, oldest-first. Allocated only when
        # H > 1 so H = 1 is zero-overhead.
        H = max(1, int(cfg.noise_config.obs_history_length))
        self._hist_len = H
        self._hist_buf: dict[str, np.ndarray] | None = None
        if H > 1:
            n = self._num_action
            dtype = get_global_dtype()
            self._hist_buf = {
                "gyro": np.zeros((num_envs, H, 3), dtype=dtype),
                "joint_pos_rel": np.zeros((num_envs, H, n), dtype=dtype),
                "dof_vel": np.zeros((num_envs, H, n), dtype=dtype),
                "last_actions": np.zeros((num_envs, H, n), dtype=dtype),
            }
        # Plumbs ``info`` from ``_compute_obs`` down to ``_build_actor_obs``
        # without changing the base-class hook signature.
        self._obs_compute_info: dict | None = None

        # Swap to the extended DR provider whenever an extended flag is on.
        dr_cfg = cfg.domain_rand
        needs_extended = (
            getattr(dr_cfg, "enable_encoder_bias", False)
            or getattr(dr_cfg, "randomize_geom_friction", False)
            or getattr(dr_cfg, "randomize_com_y", False)
            or getattr(dr_cfg, "randomize_com_z", False)
        )
        if needs_extended:
            kp = self._base_kp if (dr_cfg.randomize_kp or dr_cfg.randomize_kd) else None
            kd = self._base_kd if (dr_cfg.randomize_kp or dr_cfg.randomize_kd) else None
            base_geom_friction = None
            foot_geom_ids = None
            if dr_cfg.randomize_geom_friction:
                base_geom_friction = self._backend.get_geom_friction()
                geom_names = self._backend.get_geom_names()
                pattern = re.compile(dr_cfg.friction_geom_pattern)
                foot_geom_ids = np.asarray(
                    [i for i, name in enumerate(geom_names) if name and pattern.match(name)],
                    dtype=np.int64,
                )
                if foot_geom_ids.size == 0:
                    raise ValueError(
                        "friction_geom_pattern "
                        f"'{dr_cfg.friction_geom_pattern}' did not match any geom"
                    )
            extended_provider = G1WBTObsDomainRandomizationProvider(
                base_kp=kp,
                base_kd=kd,
                base_geom_friction=base_geom_friction,
                foot_geom_ids=foot_geom_ids,
            )
            # Swap the per-reset DR provider directly. ``_init_domain_randomization``
            # cannot be called twice — it materializes the backend at the end and
            # MuJoCo's pool raises on a second materialize. The parent's call
            # already (a) ran init randomization and (b) materialized; we only
            # need the new provider's ``build_reset_plan`` for per-episode DR.
            from unilab.dr import DomainRandomizationManager

            self._dr_manager = DomainRandomizationManager(self, extended_provider)

    # ------------------------------------------------------------------ #
    # Rewards
    # ------------------------------------------------------------------ #

    def _init_reward_functions(self) -> None:
        super()._init_reward_functions()
        self._reward_fns["joint_acc_l2"] = self._reward_joint_acc_l2
        self._reward_fns["joint_torque_l2"] = self._reward_joint_torque_l2

    def _reward_joint_acc_l2(self, info: dict) -> np.ndarray:
        dof_vel = info["dof_vel"]
        prev_dof_vel = info.get("prev_dof_vel")
        if prev_dof_vel is None or prev_dof_vel.shape != dof_vel.shape:
            return np.zeros((self._num_envs,), dtype=get_global_dtype())
        joint_acc = (dof_vel - prev_dof_vel) / self._cfg.ctrl_dt
        return np.asarray(np.sum(np.square(joint_acc), axis=1), dtype=get_global_dtype())

    def _reward_joint_torque_l2(self, info: dict) -> np.ndarray:
        dof_pos = info["dof_pos"]
        dof_vel = info["dof_vel"]
        last_actions = info.get("last_actions")
        if last_actions is None:
            return np.zeros((self._num_envs,), dtype=get_global_dtype())
        target_q = (
            last_actions * self._cfg.control_config.action_scale + self._effective_default_angles()
        )
        torque = self._base_kp * (target_q - dof_pos) - self._base_kd * dof_vel
        return np.asarray(np.sum(np.square(torque), axis=1), dtype=get_global_dtype())

    # ------------------------------------------------------------------ #
    # Obs
    # ------------------------------------------------------------------ #

    def _actor_obs_dim(self, n: int) -> int:
        nc = self._cfg.noise_config
        H = max(1, int(nc.obs_history_length))
        single_step = 2 * n + 6  # command(2n) + anchor_ori(6)
        if not nc.enable_zero_anchor_pos:
            single_step += 3
        if not nc.enable_zero_linvel:
            single_step += 3
        proprio_step = 3 + 3 * n  # gyro + joint_pos_rel + dof_vel + last_actions
        return single_step + H * proprio_step

    def _compute_obs(
        self,
        info: dict,
        motion_data: Any,
        linvel: np.ndarray,
        gyro: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        robot_body_pos_w: np.ndarray,
        robot_body_quat_w: np.ndarray,
    ) -> dict[str, np.ndarray]:
        # Stash so the overridden ``_build_actor_obs`` (called inside super)
        # can read env_ids / joint_pos_obs_bias without a signature change.
        self._obs_compute_info = info
        try:
            obs = super()._compute_obs(
                info,
                motion_data,
                linvel,
                gyro,
                dof_pos,
                dof_vel,
                robot_body_pos_w,
                robot_body_quat_w,
            )
        finally:
            self._obs_compute_info = None

        # Cache for next-step joint_acc_l2. The reset path overwrites this
        # via the DR provider's ``prev_dof_vel`` info_update.
        info["prev_dof_vel"] = dof_vel.copy()
        return obs

    def _build_actor_obs(
        self,
        *,
        command: np.ndarray,
        motion_anchor_pos_b: np.ndarray,
        motion_anchor_ori_b: np.ndarray,
        noisy_linvel: np.ndarray,
        noisy_gyro: np.ndarray,
        noisy_joint_pos_rel: np.ndarray,
        noisy_dof_vel: np.ndarray,
        last_actions: np.ndarray,
    ) -> np.ndarray:
        info = self._obs_compute_info or {}
        # Reset path is signalled by ``env_ids`` in obs_info (set by parent's
        # ``_refresh_observation_rows`` and the DR provider's
        # ``build_reset_observation``). In that case fill history slots; in
        # the per-step path we push (oldest out, current in).
        env_ids = info.get("env_ids")
        is_reset = env_ids is not None

        nc = self._cfg.noise_config

        # Per-episode encoder bias on actor's joint_pos channel.
        bias = info.get("joint_pos_obs_bias")
        if bias is not None and bias.shape == noisy_joint_pos_rel.shape:
            noisy_joint_pos_rel = np.asarray(
                noisy_joint_pos_rel + bias, dtype=noisy_joint_pos_rel.dtype
            )

        # Per-step anchor_ori noise (actor only).
        actor_anchor_ori_b = motion_anchor_ori_b
        if nc.enable_anchor_ori_noise:
            actor_anchor_ori_b = self._obs_noise(motion_anchor_ori_b, nc.scale_anchor_ori)

        # Single-step reference terms, dropping deploy-unavailable channels.
        actor_terms: list[np.ndarray] = [command]
        if not nc.enable_zero_anchor_pos:
            actor_terms.append(motion_anchor_pos_b)
        actor_terms.append(actor_anchor_ori_b)
        if not nc.enable_zero_linvel:
            actor_terms.append(noisy_linvel)

        # Proprio history (or single-step pass-through when H = 1).
        if self._hist_buf is not None:
            components = {
                "gyro": noisy_gyro,
                "joint_pos_rel": noisy_joint_pos_rel,
                "dof_vel": noisy_dof_vel,
                "last_actions": last_actions,
            }
            if is_reset:
                self._fill_obs_history(env_ids, components)
            else:
                self._push_obs_history(env_ids, components)
            sel = slice(None) if env_ids is None else env_ids
            for key in ("gyro", "joint_pos_rel", "dof_vel", "last_actions"):
                buf = self._hist_buf[key][sel]  # (n_e, H, D)
                actor_terms.append(buf.reshape(buf.shape[0], -1))
        else:
            actor_terms.extend([noisy_gyro, noisy_joint_pos_rel, noisy_dof_vel, last_actions])

        return np.concatenate(actor_terms, axis=1, dtype=get_global_dtype())

    # ------------------------------------------------------------------ #
    # Proprio history buffer maintenance.
    # Mirrors deploy ``ObservationManager`` / ``ObservationTermCfg``:
    #   * On reset: fill all H slots with the current value (matches
    #     ``ObservationTermCfg::reset`` which calls ``add()`` H times).
    #   * On step: pop oldest, push current at end.
    #   * Read order is oldest-first, so ``flatten(buf[env, :, :])`` yields
    #     ``[t-H+1, t-H+2, ..., t]`` — matches deploy
    #     ``ObservationTermCfg::get`` (deque front-to-back).
    # ------------------------------------------------------------------ #

    def _push_obs_history(
        self, env_ids: np.ndarray | None, components: dict[str, np.ndarray]
    ) -> None:
        if self._hist_buf is None:
            return
        sel = slice(None) if env_ids is None else env_ids
        for key, val in components.items():
            buf = self._hist_buf[key]
            buf[sel, :-1] = buf[sel, 1:]
            buf[sel, -1] = val

    def _fill_obs_history(
        self, env_ids: np.ndarray | None, components: dict[str, np.ndarray]
    ) -> None:
        if self._hist_buf is None:
            return
        sel = slice(None) if env_ids is None else env_ids
        for key, val in components.items():
            self._hist_buf[key][sel, :] = val[:, None, :]
