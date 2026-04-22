from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence, cast

import gymnasium as gym
import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base.backend import SimBackend
from unilab.base.base import EnvCfg
from unilab.base.dtype_config import get_global_dtype
from unilab.base.np_env import NpEnv, NpEnvState
from unilab.utils.math_utils import np_quat_apply, np_quat_mul

DEFAULT_ACTUATED_JOINT_NAMES: list[str] = [
    "right_thumb_CMC_FE",
    "right_thumb_CMC_AA",
    "right_thumb_MCP_FE",
    "right_thumb_MCP_AA",
    "right_thumb_IP",
    "right_index_MCP_FE",
    "right_index_MCP_AA",
    "right_index_PIP",
    "right_index_DIP",
    "right_middle_MCP_FE",
    "right_middle_MCP_AA",
    "right_middle_PIP",
    "right_middle_DIP",
    "right_ring_MCP_FE",
    "right_ring_MCP_AA",
    "right_ring_PIP",
    "right_ring_DIP",
    "right_pinky_CMC",
    "right_pinky_MCP_FE",
    "right_pinky_MCP_AA",
    "right_pinky_PIP",
    "right_pinky_DIP",
]

DEFAULT_FINGERTIP_BODY_NAMES: list[str] = [
    "right_thumb_DP",
    "right_index_DP",
    "right_middle_DP",
    "right_ring_DP",
    "right_pinky_DP",
]

# Source parity anchor from sharpa-rl-lab:
# rl_isaaclab/tasks/inhand_rotate/sharpa_wave_env_cfg.py (hand init_state joint_pos).
SOURCE_DEFAULT_HAND_JOINT_POS_DEG: tuple[float, ...] = (
    95.12771,
    -3.11244,
    14.81626,
    -1.03493,
    12.23986,
    65.21091,
    6.1133,
    15.58495,
    5.90325,
    31.74149,
    -0.95812,
    41.88173,
    12.844,
    31.72383,
    9.84458,
    35.22366,
    18.02839,
    10.9712,
    68.30895,
    7.99151,
    5.89626,
    5.89875,
)


@dataclass
class SharpaControlConfig:
    action_scale: float = 1.0 / 24.0
    p_gain: float = 1.0
    d_gain: float = 0.1


@dataclass
class SharpaSensorConfig:
    tactile_force_sensor_names: list[str] = field(default_factory=list)


@dataclass
class SharpaDomainRandConfig:
    randomize_base_mass: bool = False
    added_mass_range: list[float] = field(default_factory=lambda: [0.0, 0.0])
    random_com: bool = False
    com_offset_x: list[float] = field(default_factory=lambda: [0.0, 0.0])


@dataclass
class SharpaInhandBaseCfg(EnvCfg):
    model_file: str = str(ASSETS_ROOT_PATH / "robots" / "sharpa_wave" / "scene.xml")
    max_episode_seconds: float = 20.0
    sim_dt: float = 1.0 / 240.0
    ctrl_dt: float = 12.0 / 240.0

    action_space: int = 22
    observation_space: int = 192
    prop_hist_len: int = 30
    critic_info_dim: int = 8

    clip_obs: float = 5.0
    clip_actions: float = 1.0
    # NOTE: Sharpa MuJoCo XML uses position actuators; true torque-control mode is not implemented.
    torque_control: bool = False

    num_hand_dofs: int = 22
    frame_obs_dim: int = 64
    obs_lag_steps: int = 3
    obs_history_len: int = 80

    base_name: str = "right_hand_C_MC"
    object_body_name: str = "object"
    object_geom_name: str = "object"
    actuated_joint_names: list[str] = field(
        default_factory=lambda: list(DEFAULT_ACTUATED_JOINT_NAMES)
    )
    fingertip_body_names: list[str] = field(
        default_factory=lambda: list(DEFAULT_FINGERTIP_BODY_NAMES)
    )

    control_config: SharpaControlConfig = field(default_factory=SharpaControlConfig)
    sensor: SharpaSensorConfig = field(default_factory=SharpaSensorConfig)  # type: ignore[assignment]
    domain_rand: SharpaDomainRandConfig = field(default_factory=SharpaDomainRandConfig)

    reset_height_lower: float = 0.59906
    reset_height_upper: float = 0.63906
    reset_angle_diff: float = 45.0 / 180.0 * np.pi
    reset_random_quat: bool = False

    rot_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)

    grasp_cache_path: str = "cache/sharpa_grasp_linspace"

    joint_noise_scale: float = 0.02

    enable_tactile: bool = True
    binary_contact: bool = False
    enable_contact_pos: bool = False
    disable_tactile_ids: list[int] = field(default_factory=list)
    contact_smooth: float = 0.5
    contact_threshold: float = 0.05
    contact_latency: float = 0.005
    contact_sensor_noise: float = 0.01

    dof_limits_scale: float = 0.9

    scale_range: list[float] = field(default_factory=lambda: [0.5, 0.5, 1.0])

    randomize_pd_gains: bool = True
    randomize_p_gain_scale_lower: float = 0.5
    randomize_p_gain_scale_upper: float = 2.0
    randomize_d_gain_scale_lower: float = 0.5
    randomize_d_gain_scale_upper: float = 2.0

    randomize_friction: bool = True
    randomize_friction_scale_lower: float = 0.5
    randomize_friction_scale_upper: float = 2.0
    elastomer_base_friction: float = 0.8
    metal_base_friction: float = 0.1
    object_base_friction: float = 0.5

    randomize_com: bool = True
    randomize_com_lower: float = -0.01
    randomize_com_upper: float = 0.01

    randomize_mass: bool = True
    randomize_mass_lower: float = 0.01
    randomize_mass_upper: float = 0.25

    force_scale: float = 2.0
    random_force_prob_scalar: float = 0.25
    force_decay: float = 0.9
    force_decay_interval: float = 0.08

    gravity_curriculum: bool = True

    debug_show_axes: bool = False


def format_scale_tag(scale_range: Sequence[float]) -> str:
    if len(scale_range) != 3:
        raise ValueError(f"scale_range must have 3 values [lower, upper, num], got {scale_range}")
    return f"{float(scale_range[0]):g}-{float(scale_range[1]):g}-{int(scale_range[2])}"


def resolve_grasp_cache_file(grasp_cache_path: str, scale_range: Sequence[float]) -> Path:
    base = Path(grasp_cache_path)
    if base.suffix == ".npy":
        return base
    return Path(f"{grasp_cache_path}_{format_scale_tag(scale_range)}.npy")


def sample_bucketed_grasp_cache(
    grasp_cache: np.ndarray,
    scale_ids: np.ndarray,
    num_scales: int,
) -> np.ndarray:
    num_envs = scale_ids.shape[0]
    if num_scales <= 0:
        raise ValueError(f"num_scales must be positive, got {num_scales}")
    if grasp_cache.shape[1] < 29:
        raise ValueError(f"Expected cached grasp shape (?, 29), got {grasp_cache.shape}")
    if grasp_cache.shape[0] % num_scales != 0:
        raise ValueError(
            f"grasp_cache rows {grasp_cache.shape[0]} not divisible by num_scales={num_scales}"
        )

    bucket = grasp_cache.shape[0] // num_scales
    sampled = np.zeros((num_envs, 29), dtype=np.float64)
    for scale_idx in range(num_scales):
        env_ids = np.flatnonzero(scale_ids == scale_idx)
        if len(env_ids) == 0:
            continue
        sample_ids = np.random.randint(0, bucket, size=len(env_ids)) + scale_idx * bucket
        sampled[env_ids] = grasp_cache[sample_ids]
    return sampled


def repeat_obs_history(init_frame: np.ndarray, history_len: int) -> np.ndarray:
    history = np.broadcast_to(
        init_frame[:, None, :], (init_frame.shape[0], history_len, init_frame.shape[1])
    ).copy()
    return np.asarray(history, dtype=init_frame.dtype)


def apply_random_rotation_to_positions(
    positions: np.ndarray,
    center: np.ndarray,
    random_quat: np.ndarray,
) -> np.ndarray:
    rotated = np_quat_apply(random_quat, positions - center)
    return np.asarray(rotated + center, dtype=positions.dtype)


class SharpaInhandBaseEnv(NpEnv):
    _cfg: SharpaInhandBaseCfg

    def __init__(self, cfg: SharpaInhandBaseCfg, backend: SimBackend, num_envs: int = 1) -> None:
        super().__init__(cfg, backend, num_envs)

        self._np_dtype = get_global_dtype()

        self._num_action = int(cfg.num_hand_dofs)
        actuator_range = np.asarray(self._backend.get_actuator_ctrl_range(), dtype=self._np_dtype)
        if actuator_range.shape[0] < self._num_action:
            raise ValueError(
                f"Model has {actuator_range.shape[0]} actuators, but Sharpa task needs {self._num_action}"
            )

        self._ctrl_lower = np.asarray(actuator_range[: self._num_action, 0], dtype=self._np_dtype)
        self._ctrl_upper = np.asarray(actuator_range[: self._num_action, 1], dtype=self._np_dtype)

        self._init_qpos = self._resolve_init_qpos()
        self._init_qvel = np.asarray(self._backend.get_init_qvel(), dtype=np.float64)
        self.nq = int(self._init_qpos.shape[0])
        self.nv = int(self._init_qvel.shape[0])

        if self.nq < self._num_action + 7:
            raise ValueError(
                f"Model qpos dim {self.nq} is too small for {self._num_action} hand DoFs + object pose"
            )

        self._obj_pos_slice = slice(self._num_action, self._num_action + 3)
        self._obj_quat_slice = slice(self._num_action + 3, self._num_action + 7)

        source_default_angles = np.deg2rad(
            np.asarray(SOURCE_DEFAULT_HAND_JOINT_POS_DEG, dtype=np.float64)
        )
        if source_default_angles.shape[0] != self._num_action:
            raise ValueError(
                "Source default hand joint pose size mismatch: "
                f"{source_default_angles.shape[0]} vs expected {self._num_action}"
            )
        self.default_angles = np.asarray(source_default_angles, dtype=self._np_dtype)

        self._action_space = gym.spaces.Box(
            low=-float(cfg.clip_actions),
            high=float(cfg.clip_actions),
            shape=(self._num_action,),
            dtype=np.float32,
        )

        self._object_body_ids = self._backend.get_body_ids([cfg.object_body_name])
        self._fingertip_body_ids = self._backend.get_body_ids(cfg.fingertip_body_names)
        self._object_geom_base_size = self._resolve_object_geom_base_size()

        self._num_tactile = len(cfg.fingertip_body_names)
        self.last_contacts = np.zeros((num_envs, self._num_tactile), dtype=self._np_dtype)

        self.object_default_pose = np.zeros((num_envs, 7), dtype=self._np_dtype)

        self.obs_buf_lag_history = np.zeros(
            (num_envs, cfg.obs_history_len, cfg.frame_obs_dim), dtype=self._np_dtype
        )
        self.proprio_hist_buf = np.zeros(
            (num_envs, cfg.prop_hist_len, cfg.frame_obs_dim), dtype=self._np_dtype
        )
        self.critic_info_buf = np.zeros((num_envs, cfg.critic_info_dim), dtype=self._np_dtype)

        self.scale_ids, self._num_scales, self._bucket_env = self._build_scale_ids(
            num_envs, cfg.scale_range
        )
        self.scale_values = self._build_scale_values(cfg.scale_range)

    @property
    def action_space(self) -> gym.spaces.Box:
        return self._action_space  # type: ignore[no-any-return]

    def _resolve_init_qpos(self) -> np.ndarray:
        for key_name in ("home", "stand", "default"):
            try:
                return np.asarray(self._backend.get_keyframe_qpos(key_name), dtype=np.float64)
            except Exception:
                continue

        model = self._backend.model
        if hasattr(model, "qpos0"):
            return np.asarray(model.qpos0, dtype=np.float64)
        if hasattr(model, "compute_init_dof_pos"):
            return np.asarray(model.compute_init_dof_pos(), dtype=np.float64)

        raise ValueError("Could not resolve initial qpos from backend keyframes/model")

    def _build_scale_ids(
        self, num_envs: int, scale_range: Sequence[float]
    ) -> tuple[np.ndarray, int, int]:
        """Build deterministic near-even environment assignments for each scale.

        Args:
            num_envs: Number of vectorized environments to assign.
            scale_range: Scale range config in [lower, upper, num_scales] form.

        Returns:
            Tuple of scale id per environment, total scale count, and the minimum
            number of environments assigned to any scale.
        """
        num_scales = int(scale_range[2])
        if num_scales <= 0:
            raise ValueError(f"scale_range[2] must be >= 1, got {scale_range[2]}")

        bucket_env = num_envs // num_scales
        remainder = num_envs % num_scales
        # Assign the remainder to the lowest scale ids to keep bucket sizes within one env.
        counts = np.full((num_scales,), bucket_env, dtype=np.int32)
        counts[:remainder] += 1
        scale_ids = np.repeat(np.arange(num_scales, dtype=np.int32), counts)
        return scale_ids, num_scales, bucket_env

    def _build_scale_values(self, scale_range: Sequence[float]) -> np.ndarray:
        lower = float(scale_range[0])
        upper = float(scale_range[1])
        if lower <= 0.0 or upper <= 0.0:
            raise ValueError(f"scale_range bounds must be positive, got {scale_range[:2]}")
        return np.asarray(np.linspace(lower, upper, self._num_scales), dtype=np.float64)

    def _resolve_object_geom_base_size(self) -> np.ndarray | None:
        if getattr(self._backend, "backend_type", None) != "mujoco":
            return None

        import mujoco

        geom_id = mujoco.mj_name2id(
            self._backend.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            self._cfg.object_geom_name,
        )
        if geom_id < 0:
            raise ValueError(f"Geom '{self._cfg.object_geom_name}' not found in MuJoCo model")
        return cast(
            np.ndarray,
            np.asarray(self._backend.model.geom_size[geom_id], dtype=np.float64).copy(),
        )

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        clipped_actions = np.clip(actions, -self._cfg.clip_actions, self._cfg.clip_actions)
        clipped_actions = np.asarray(clipped_actions[:, : self._num_action], dtype=self._np_dtype)

        state.info["last_actions"] = state.info.get("current_actions", clipped_actions.copy())
        state.info["current_actions"] = clipped_actions

        prev_targets = state.info.get(
            "prev_targets",
            np.broadcast_to(self.default_angles, (self._num_envs, self._num_action)).copy(),
        )
        targets = prev_targets + self._cfg.control_config.action_scale * clipped_actions
        targets = np.clip(targets, self._ctrl_lower, self._ctrl_upper)
        prev_targets = np.asarray(targets, dtype=self._np_dtype)
        state.info["prev_targets"] = prev_targets
        return prev_targets

    def get_hand_dof_pos(self) -> np.ndarray:
        return np.asarray(self._backend.get_dof_pos()[:, : self._num_action], dtype=self._np_dtype)

    def get_hand_dof_vel(self) -> np.ndarray:
        return np.asarray(self._backend.get_dof_vel()[:, : self._num_action], dtype=self._np_dtype)

    def get_fingertip_pos(self) -> np.ndarray:
        return np.asarray(
            self._backend.get_body_pos_w(self._fingertip_body_ids), dtype=self._np_dtype
        )

    def get_object_pos(self) -> np.ndarray:
        return np.asarray(
            self._backend.get_body_pos_w(self._object_body_ids)[:, 0, :], dtype=self._np_dtype
        )

    def get_object_quat(self) -> np.ndarray:
        return np.asarray(
            self._backend.get_body_quat_w(self._object_body_ids)[:, 0, :], dtype=self._np_dtype
        )

    def _extract_sensor_scalar(self, sensor_name: str) -> np.ndarray:
        data = np.asarray(self._backend.get_sensor_data(sensor_name), dtype=self._np_dtype)
        if data.ndim == 1:
            return data
        if data.ndim == 2 and data.shape[1] == 1:
            return data[:, 0]
        if data.ndim == 2 and data.shape[1] >= 3:
            return np.asarray(np.linalg.norm(data[:, :3], axis=1), dtype=self._np_dtype)
        flat = data.reshape(data.shape[0], -1)
        return np.asarray(flat[:, 0], dtype=self._np_dtype)

    def _compute_tactile_observation(self) -> np.ndarray:
        tactile = np.zeros((self._num_envs, self._num_tactile), dtype=self._np_dtype)

        if self._cfg.enable_tactile and self._cfg.sensor.tactile_force_sensor_names:
            for i, sensor_name in enumerate(
                self._cfg.sensor.tactile_force_sensor_names[: self._num_tactile]
            ):
                try:
                    tactile[:, i] = self._extract_sensor_scalar(sensor_name)
                except Exception:
                    tactile[:, i] = 0.0

            for disabled_id in self._cfg.disable_tactile_ids:
                if 0 <= disabled_id < self._num_tactile:
                    tactile[:, disabled_id] = 0.0

            latency = np.where(
                np.random.rand(self._num_envs, self._num_tactile) < self._cfg.contact_latency,
                1.0,
                0.0,
            ).astype(self._np_dtype)

            if self._cfg.binary_contact:
                tactile = (tactile > self._cfg.contact_threshold).astype(self._np_dtype)
                self.last_contacts = self.last_contacts * latency + tactile * (1.0 - latency)
                noise_mask = (
                    np.random.rand(self._num_envs, self._num_tactile)
                    >= self._cfg.contact_sensor_noise
                ).astype(self._np_dtype)
                tactile = np.where(self.last_contacts > 0.1, self.last_contacts * noise_mask, 0.0)
            else:
                smooth_contact = tactile * self._cfg.contact_smooth + self.last_contacts * (
                    1.0 - self._cfg.contact_smooth
                )
                self.last_contacts = self.last_contacts * latency + smooth_contact * (1.0 - latency)
                tactile = self.last_contacts.copy()
        else:
            self.last_contacts.fill(0.0)

        return tactile

    def _compute_contact_positions(self, tactile: np.ndarray) -> np.ndarray:
        del tactile
        # TODO(sharpa_inhand): IsaacLab contact positions are defined through contact sensor
        # frame transforms. Backend-level contact point parity is not available yet in the
        # current UniLab sensor contract, so we keep this channel as zeros for now.
        return np.zeros((self._num_envs, self._num_tactile * 3), dtype=self._np_dtype)

    def _normalize_joint_pos(self, dof_pos: np.ndarray) -> np.ndarray:
        return np.asarray(
            (2.0 * dof_pos - self._ctrl_upper - self._ctrl_lower)
            / (self._ctrl_upper - self._ctrl_lower + 1.0e-8),
            dtype=self._np_dtype,
        )

    def _sample_pd_scales(self, lower: float, upper: float, shape: tuple[int, int]) -> np.ndarray:
        if lower > 1.0 or upper < 1.0:
            raise ValueError("PD randomization scales must satisfy lower <= 1 <= upper")
        small = np.random.uniform(lower, 1.0, size=shape)
        large = np.random.uniform(1.0, upper, size=shape)
        use_small = np.random.rand(*shape) > 0.5
        return np.where(use_small, small, large).astype(self._np_dtype)

    def _resolve_pd_gains(self, info: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        p_gain = info.get(
            "p_gain",
            np.full(
                (self._num_envs, self._num_action),
                self._cfg.control_config.p_gain,
                dtype=self._np_dtype,
            ),
        )
        d_gain = info.get(
            "d_gain",
            np.full(
                (self._num_envs, self._num_action),
                self._cfg.control_config.d_gain,
                dtype=self._np_dtype,
            ),
        )
        return np.asarray(p_gain, dtype=self._np_dtype), np.asarray(d_gain, dtype=self._np_dtype)

    def _update_proprio_history(self, obs_history: np.ndarray) -> np.ndarray:
        return np.asarray(obs_history[:, -self._cfg.prop_hist_len :], dtype=self._np_dtype)

    def _rotate_axis(self, axis: np.ndarray, quat: np.ndarray) -> np.ndarray:
        return np.asarray(np_quat_apply(quat, axis), dtype=self._np_dtype)

    def _rotate_quat(self, quat: np.ndarray, random_quat: np.ndarray) -> np.ndarray:
        return np.asarray(np_quat_mul(random_quat, quat), dtype=self._np_dtype)
