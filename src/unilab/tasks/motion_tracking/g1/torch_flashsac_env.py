"""Torch task runtime for the FlashSAC G1 motion-tracking owner.

This is a deliberately narrow GPU implementation of the canonical
``g1_motion_tracking`` FlashSAC Manager-Based contract.  Configuration,
backend construction, scene materialization, and motion loading remain cold
CPU work in their owning modules.  The per-step action transform, motion
update, observations, reward, termination, and selected-row reset run on one
CUDA device through UniSim's public tensor lifecycle.  A HOST_BRIDGE backend
keeps its CPU-authoritative physics plane and performs only its negotiated
packed boundaries; generic Manager NumPy observations never enter the CUDA
hot path.  Unsupported task or backend terms fail closed rather than falling
back silently.
"""

from __future__ import annotations

import gc
from time import perf_counter
from typing import Any, Mapping

import numpy as np
import torch
from unisim.backend.base import SimBackend, TensorExecution, tensor_device_matches

from unilab.base.backend_factory import create_backend, env_backend_kwargs
from unilab.base.base import ABEnv, EnvPlayCapabilities
from unilab.base.cpu_runtime import apply_env_cpu_runtime
from unilab.base.entity import EntityCfg
from unilab.base.torch_env import TorchEnv, TorchEnvState
from unilab.envs.manager_based_rl_env import (
    ManagerBasedRlEnv,
    ManagerBasedRlEnvCfg,
    _resolve_backend_entity_contract,
)
from unilab.managers._noise.noise_cfg import UniformNoiseCfg
from unilab.managers.observation_manager import ObservationGroupCfg
from unilab.managers.reward_manager import RewardTermCfg
from unilab.tasks.motion_tracking.common.manager_terms import (
    MotionCommand,
    MotionCommandCfg,
    MotionJointPositionAction,
    MotionJointPositionActionCfg,
)
from unilab.tasks.motion_tracking.common.tensor_runtime import (
    TensorEpisodeMetrics,
    TensorObservationNoise,
    TensorResetPlan,
    semantic_fingerprint,
)
from unilab.tasks.motion_tracking.common.tensor_state_store import TensorDeviceStateStore


def _to_device(value: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.array(value, dtype=value.dtype, order="C", copy=True)).to(device)


def _qualified_name(value: Any) -> str:
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _torch_g1_flashsac_owner_identity(cfg: ManagerBasedRlEnvCfg) -> str:
    """Return a versioned semantic fingerprint for the narrow Torch owner."""
    if cfg.fixed_model_variants is not None:
        raise ValueError("Torch G1 FlashSAC v1 does not support fixed model variants")
    if cfg.scene is None:
        raise ValueError("Torch G1 FlashSAC requires a scene owner")
    return semantic_fingerprint(
        "unilab.motion_tracking.g1.tensor.v1",
        (
            1,
            cfg.scene,
            cfg.sim_dt,
            cfg.ctrl_dt,
            cfg.max_episode_seconds,
            cfg.observations,
            cfg.actions,
            cfg.commands,
            cfg.rewards,
            cfg.terminations,
            cfg.events,
            cfg.curriculum,
            cfg.metrics,
            cfg.recorders,
            cfg.auto_reset,
            cfg.is_finite_horizon,
            cfg.scale_rewards_by_dt,
            cfg.policy_observation_group,
            cfg.critic_observation_group,
        ),
    )


_TORCH_G1_FLASHSAC_OWNER_IDENTITY_V1 = (
    "21f829c6ccda078a15b306ac5afd294ede8b69ce22cd3eda2ea4c79c6e4fd681"
)
# V2 only adds the explicit tensor-read entity binding required to compile
# named sensors into one packed public read; no semantic term/equation changes.
_TORCH_G1_FLASHSAC_OWNER_IDENTITY_V2 = (
    "6838841dfebc4d64ddaec3a2fcf123b29f28f858b397495a5bf2680f00af1b60"
)
_TORCH_G1_SAC_OWNER_IDENTITY_V1 = "90236c9e02e460817208b6a8e14f614ae4d2a797f16d1a5bfd0d306f4059d385"
_TORCH_G1_FLIP_SAC_OWNER_IDENTITY_V1 = (
    "764e0d5061654c52b3658bb8b074684773f531167a958c8c7c57f467c5b5e784"
)


def _validate_torch_g1_flashsac_owner_contract(cfg: ManagerBasedRlEnvCfg) -> None:
    identity = _torch_g1_flashsac_owner_identity(cfg)
    if identity not in {
        _TORCH_G1_FLASHSAC_OWNER_IDENTITY_V1,
        _TORCH_G1_FLASHSAC_OWNER_IDENTITY_V2,
        _TORCH_G1_SAC_OWNER_IDENTITY_V1,
        _TORCH_G1_FLIP_SAC_OWNER_IDENTITY_V1,
    }:
        raise ValueError(
            "Torch G1 tensor runtime supports only canonical owner contracts; "
            f"semantic identity {identity} is not recognized"
        )


def _all_finite(*values: torch.Tensor) -> bool:
    valid = torch.isfinite(values[0]).all()
    for value in values[1:]:
        valid = valid & torch.isfinite(value).all()
    return bool(valid)


def _scalar_noise_bound(value: Any, *, label: str) -> float:
    if isinstance(value, (list, tuple, np.ndarray)):
        raise TypeError(f"FlashSAC G1 Torch runtime requires scalar {label}")
    return float(value)


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(dim=-1)
    bw, bx, by, bz = b.unbind(dim=-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dim=-1,
    )


def _quat_inv(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[..., 0:1], -q[..., 1:4]), dim=-1)


def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    qv = q[..., 1:4]
    t = 2.0 * torch.linalg.cross(qv, v, dim=-1)
    return v + q[..., 0:1] * t + torch.linalg.cross(qv, t, dim=-1)


def _quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return _quat_apply(_quat_inv(q), v)


def _quat_error_squared(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    rel = _quat_mul(_quat_inv(a), b)
    xyz = torch.linalg.vector_norm(rel[..., 1:4], dim=-1)
    angle = 2.0 * torch.atan2(xyz, rel[..., 0].abs().clamp(max=1.0))
    return angle.square()


def _gravity_z_in_body(q: torch.Tensor) -> torch.Tensor:
    return 2.0 * (q[..., 1] ** 2 + q[..., 2] ** 2) - 1.0


def _adaptive_failure_counts(
    bin_indices: torch.Tensor, terminated: torch.Tensor, n_bins: int
) -> torch.Tensor:
    discard_bin = n_bins
    failed_bins = torch.where(
        terminated,
        bin_indices,
        torch.full_like(bin_indices, discard_bin),
    )
    return torch.bincount(failed_bins, minlength=discard_bin + 1)[:n_bins].to(torch.float32)


def _adaptive_failure_alpha(terminated: torch.Tensor, alpha: float) -> torch.Tensor:
    return torch.any(terminated).to(dtype=torch.float32) * alpha


def _rot6(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(dim=-1)
    return torch.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
        ),
        dim=-1,
    )


def _yaw_quat(robot_q: torch.Tensor, motion_q: torch.Tensor) -> torch.Tensor:
    delta = _quat_mul(robot_q, _quat_inv(motion_q))
    yaw = torch.atan2(
        2.0 * (delta[..., 0] * delta[..., 3] + delta[..., 1] * delta[..., 2]),
        1.0 - 2.0 * (delta[..., 2] * delta[..., 2] + delta[..., 3] * delta[..., 3]),
    )
    half = 0.5 * yaw
    return torch.stack(
        (torch.cos(half), torch.zeros_like(half), torch.zeros_like(half), torch.sin(half)), dim=-1
    )


def _euler_xyz_quat(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    cr, sr = torch.cos(0.5 * roll), torch.sin(0.5 * roll)
    cp, sp = torch.cos(0.5 * pitch), torch.sin(0.5 * pitch)
    cy, sy = torch.cos(0.5 * yaw), torch.sin(0.5 * yaw)
    return torch.stack(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ),
        dim=-1,
    )


class _DeviceResidentColdContractProxy(ManagerBasedRlEnv):
    """Manager proxy that extracts cold contract metadata without a CPU read plane.

    The direct FlashSAC runtime owns all state reads, observations, and resets
    on CUDA.  DEVICE_RESIDENT backends deliberately do not expose a CPU tensor
    read plane, while a normal ``ManagerBasedRlEnv`` compiles its generic
    observation reads during construction.  This proxy retains manager and
    scene construction for contract extraction, but has no hot read lifecycle.
    """

    def _compile_tensor_read_plan(self) -> None:
        self.scene._tensor_read_plan = None


class TorchG1MotionTrackingFlashSACEnv(TorchEnv):
    """GPU runtime for the exact canonical G1 FlashSAC owner contract."""

    is_vector_env = True

    _qpos: torch.Tensor | None
    _qvel: torch.Tensor | None
    _joint_pos: torch.Tensor
    _joint_vel: torch.Tensor
    _linvel: torch.Tensor
    _gyro: torch.Tensor
    _terminations: dict[str, dict[str, Any]]
    _observation_noise: TensorObservationNoise
    _episode_metrics: TensorEpisodeMetrics
    _state_store: TensorDeviceStateStore

    @property
    def _manager_cfg(self) -> ManagerBasedRlEnvCfg:
        if not isinstance(self._cfg, ManagerBasedRlEnvCfg):
            raise TypeError(type(self._cfg).__name__)
        return self._cfg

    def _require_cpu_env(self) -> ManagerBasedRlEnv:
        if self._cpu_env is None:
            raise RuntimeError("cold Manager-Based proxy is not available")
        return self._cpu_env

    def __init__(
        self,
        cfg: ManagerBasedRlEnvCfg,
        backend: SimBackend,
        num_envs: int,
        *,
        device: str | torch.device = "cuda",
    ):
        import torch

        self._backend = backend
        self._num_envs = int(num_envs)
        self._cfg = cfg
        self._device = torch.device(device)
        if self._device.type == "cuda" and self._device.index is None:
            self._device = torch.device("cuda", index=torch.cuda.current_device())
        self._dtype = torch.float32
        self._truncated_scratch = torch.zeros(
            (self._num_envs,), dtype=torch.bool, device=self._device
        )
        self._final_observation_scratch = None
        self._tensor_runtime_bound = False
        self.step_counter = 0
        self._autoreset = True
        self._autoreset_reset_active = False
        self._rgb_array_renderer_ready = False
        self._nan_guard = None

        self._torch = torch
        if self._device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("TorchG1MotionTrackingFlashSACEnv requires CUDA")
        if backend.num_envs != self._num_envs:
            raise ValueError("backend num_envs does not match the environment")
        # External-worker backends publish their runtime-dependent tensor
        # capability only after the public cold-path materialization point.
        # Negotiate before constructing the proxy; otherwise IsaacGym's
        # still-unmaterialized model info correctly (but prematurely) reports
        # UNSUPPORTED and the task factory fails closed.
        # Some in-process backends materialize while their EntityScene is
        # constructed; SimBackend exposes no idempotence query, so the backend
        # owner's capability is the public readiness signal. Re-materialize
        # only when negotiation still reports no tensor lifecycle.
        if backend.tensor_execution() is TensorExecution.UNSUPPORTED:
            backend.materialize()
        if backend.backend_type == "isaacgym":
            # Preview 4 does not publish fresh rigid-body/sensor views until
            # its first CUDA-IPC step.  This is a cold-path worker
            # initialization barrier; the direct runtime resets the state
            # before exposing the first observation.
            backend.step_tensor(
                torch.zeros(
                    (self._num_envs, backend.num_actuators),
                    dtype=torch.float32,
                    device=self._device,
                )
            )
        self._validate_backend()
        # Cold-path contract extraction only. Keep the Manager proxy on CPU so
        # its temporary observation computation does not require every generic
        # term to be CUDA-tensor-native; the direct runtime owns all hot tensors
        # on ``self.device``.
        saved_tensor_runtime = (cfg.tensor_runtime, cfg.tensor_runtime_device)
        cfg.tensor_runtime = False
        cfg.tensor_runtime_device = "cpu"
        self._cpu_env: ManagerBasedRlEnv | None = None
        proxy_type = (
            _DeviceResidentColdContractProxy
            if backend.tensor_execution() is TensorExecution.DEVICE_RESIDENT
            else ManagerBasedRlEnv
        )
        self._cpu_env = proxy_type(cfg, backend, self._num_envs)
        try:
            self._extract_contract()
            self._state_store = TensorDeviceStateStore(
                backend=self._backend,
                device=self.device,
                num_envs=self._num_envs,
                joint_qpos_ids=self._joint_qpos_ids,
                joint_qvel_ids=self._joint_qvel_ids,
                body_names=self._body_names,
                body_ids=self._body_ids,
            )
            self._qpos = None
            self._qvel = None
            self._joint_pos = self._state_store.joint_pos
            self._joint_vel = self._state_store.joint_vel
            self._linvel = self._state_store.linvel
            self._gyro = self._state_store.gyro
            self._robot_body_pos = self._state_store.robot_body_pos
            self._robot_body_quat = self._state_store.robot_body_quat
            self._robot_body_lin_vel = self._state_store.robot_body_lin_vel
            self._robot_body_ang_vel = self._state_store.robot_body_ang_vel
            self._obs_groups_spec = {
                "obs": 160,
                "critic": 289,
            }
            self._observation_space = self._cpu_env.observation_space
            self._action_space = self._cpu_env.action_space
            self._state: TorchEnvState | None = None
            self._initial_seed = cfg.seed
            self._last_backend_reset_result: dict | None = None
        except BaseException:
            if self._cpu_env is not None:
                self._cpu_env.close()
            cfg.tensor_runtime, cfg.tensor_runtime_device = saved_tensor_runtime
            raise
        # The proxy compiled no packed reads, so this drop is enough to detach
        # any generic tensor-read terms from the cold-path proxy.
        self._cpu_env.scene._tensor_read_plan = None
        # HOST_BRIDGE has a valid CPU plane, so retain its historical seeded
        # proxy reset. DEVICE_RESIDENT has no CPU plane: its direct CUDA
        # runtime seeds and resets state in ``init_state`` / ``_reset_rows``.
        try:
            if backend.tensor_execution() is TensorExecution.HOST_BRIDGE:
                self._cpu_env.reset(seed=self._initial_seed)
        finally:
            cfg.tensor_runtime, cfg.tensor_runtime_device = saved_tensor_runtime
        del self._cpu_env
        self._episode_metrics = TensorEpisodeMetrics.create(self._num_envs, self.device)

    def _validate_backend(self) -> None:
        capabilities = self._backend.get_tensor_capabilities()
        execution = capabilities.execution
        if execution not in {TensorExecution.DEVICE_RESIDENT, TensorExecution.HOST_BRIDGE}:
            raise RuntimeError(
                "Torch G1 FlashSAC runtime requires DEVICE_RESIDENT or HOST_BRIDGE tensor "
                f"capabilities; received {execution!r}"
            )
        if self._backend.tensor_execution() is not execution:
            raise RuntimeError("backend tensor execution does not match its declared capabilities")
        if not tensor_device_matches(capabilities.torch_devices, self.device):
            raise RuntimeError(
                f"backend did not accept Torch device {str(self.device)!r}; "
                f"supported devices are {capabilities.torch_devices}"
            )
        if not {"qpos", "qvel"}.issubset(capabilities.state_fields):
            missing = sorted({"qpos", "qvel"} - set(capabilities.state_fields))
            raise RuntimeError(f"backend tensor state views are missing fields: {missing}")
        if not capabilities.state_views or not capabilities.sensor_views:
            raise RuntimeError("backend did not negotiate state and sensor tensor views")
        if not capabilities.stepping or not capabilities.selected_reset:
            raise RuntimeError("backend did not negotiate tensor stepping and selected reset")
        if (
            capabilities.execution is TensorExecution.HOST_BRIDGE
            and not capabilities.packed_host_bridge
        ):
            raise RuntimeError("Host-bridge backend did not negotiate packed tensor I/O")
        if self._cfg.fixed_model_variants is not None:
            raise ValueError("Torch G1 FlashSAC v1 does not support fixed model variants")
        state_views = self._backend.get_state_views(("qpos", "qvel"), device=self.device)
        for name, value in state_views.items():
            if value.device != self.device:
                raise RuntimeError(
                    f"backend tensor state {name!r} lives on {value.device}, expected {self.device}"
                )
        command_cfg = self._manager_cfg.commands.get("motion")
        if not isinstance(command_cfg, MotionCommandCfg):
            raise TypeError("Torch G1 FlashSAC requires MotionCommandCfg for sensor preflight")
        required_sensors = ["pelvis_local_linvel", "torso_gyro"]
        if execution is TensorExecution.DEVICE_RESIDENT:
            required_sensors.extend(
                f"{prefix}_{name}"
                for name in command_cfg.body_names
                for prefix in (
                    "track_pos_w",
                    "track_quat_w",
                    "track_linvel_w",
                    "track_angvel_w",
                )
            )
        for name in required_sensors:
            value = self._backend.get_sensor_view(name, device=self.device)
            if value.device != self.device:
                raise RuntimeError(
                    f"backend tensor sensor {name!r} lives on {value.device}, "
                    f"expected {self.device}"
                )
        self._backend.get_body_ids(command_cfg.body_names)

    def _extract_contract(self) -> None:
        cfg = self._manager_cfg
        if set(cfg.actions) != {"joint_pos"}:
            raise ValueError(f"unsupported FlashSAC G1 actions: {sorted(cfg.actions)}")
        action_cfg = cfg.actions["joint_pos"]
        if not isinstance(action_cfg, MotionJointPositionActionCfg):
            raise TypeError("FlashSAC G1 action must be MotionJointPositionActionCfg")
        if action_cfg.command_name != "motion" or action_cfg.simulate_action_latency:
            raise ValueError("unsupported FlashSAC G1 action parameters")
        if set(cfg.commands) != {"motion"} or not isinstance(
            cfg.commands["motion"], MotionCommandCfg
        ):
            raise TypeError("FlashSAC G1 requires exactly one canonical MotionCommandCfg")
        command_cfg = cfg.commands["motion"]
        if command_cfg.sampling_mode not in {"adaptive", "mixed"}:
            raise ValueError("G1 tensor runtime requires adaptive or mixed sampling")
        if not command_cfg.params.truncate_on_clip_end:
            raise ValueError("G1 tensor runtime requires truncating sampling")
        if any(term is not None for term in cfg.events.values()):
            raise ValueError("G1 tensor runtime requires a DR-free event lifecycle")
        if cfg.curriculum or cfg.metrics or cfg.recorders:
            raise ValueError("FlashSAC G1 Torch runtime does not support extra lifecycle managers")
        if not cfg.auto_reset or cfg.is_finite_horizon:
            raise ValueError("FlashSAC G1 Torch runtime requires auto-reset infinite horizon")
        cpu_env = self._require_cpu_env()
        if not np.array_equal(cpu_env.scene.env_origins, np.zeros_like(cpu_env.scene.env_origins)):
            raise ValueError("FlashSAC G1 Torch runtime requires zero scene env_origins")

        actor_group = cfg.observations.get("actor")
        critic_group = cfg.observations.get("critic")
        if not isinstance(actor_group, ObservationGroupCfg) or not isinstance(
            critic_group, ObservationGroupCfg
        ):
            raise TypeError("FlashSAC G1 requires actor and critic observation groups")
        actor_terms = tuple(actor_group.terms)
        critic_terms = tuple(critic_group.terms)
        if any(term is None for term in actor_terms) or any(term is None for term in critic_terms):
            raise ValueError("FlashSAC G1 observation terms must all be concrete")
        expected_actor = (
            "command",
            "motion_anchor_pos_b",
            "motion_anchor_ori_b",
            "base_lin_vel",
            "base_ang_vel",
            "joint_pos",
            "joint_vel",
            "actions",
        )
        expected_critic = (*expected_actor, "body_pos", "body_ori", "sac_base_lin_vel")
        expected_sac_critic = (
            "command",
            "motion_anchor_pos_b",
            "motion_anchor_ori_b",
            "base_lin_vel",
            "base_ang_vel",
            "joint_vel",
            "actions",
            "joint_pos",
            "body_pos",
            "body_ori",
            "sac_base_lin_vel",
        )
        if actor_terms != expected_actor or critic_terms not in {
            expected_critic,
            expected_sac_critic,
        }:
            raise ValueError("unsupported FlashSAC G1 observation declaration")
        self._critic_prefix_names = tuple(critic_terms[:-3])

        reward_terms: dict[str, RewardTermCfg] = {}
        for name, term in cfg.rewards.items():
            if not isinstance(term, RewardTermCfg):
                raise TypeError("FlashSAC G1 rewards must all be concrete RewardTermCfg terms")
            reward_terms[name] = term
        core_rewards = {
            "motion_global_root_pos",
            "motion_global_root_ori",
            "motion_body_pos",
            "motion_body_ori",
            "motion_body_lin_vel",
            "motion_body_ang_vel",
            "motion_joint_pos",
            "motion_joint_vel",
            "action_rate_l2",
            "joint_limit",
            "undesired_contacts",
        }
        supported_rewards = {
            frozenset(core_rewards),
            frozenset({*core_rewards, "motion_ee_body_pos_z"}),
        }
        if frozenset(reward_terms) not in supported_rewards:
            raise ValueError(f"unsupported G1 tensor rewards: {sorted(reward_terms)}")
        expected_terminations = {
            frozenset(
                {
                    "time_out",
                    "motion_clip_end",
                    "anchor_pos",
                    "anchor_ori",
                    "ee_body_pos",
                }
            ),
            frozenset(
                {
                    "time_out",
                    "motion_clip_end",
                    "anchor_pos",
                    "anchor_ori",
                    "ee_body_pos",
                    "undesired_contacts",
                }
            ),
        }
        if frozenset(cfg.terminations) not in expected_terminations:
            raise ValueError(f"unsupported G1 tensor terminations: {sorted(cfg.terminations)}")
        if any(term is None for term in cfg.terminations.values()):
            raise ValueError("FlashSAC G1 terminations must all be concrete termination terms")

        cpu_env = self._require_cpu_env()
        cpu_command = cpu_env.command_manager.get_term("motion")
        cpu_action = cpu_env.action_manager.get_term("joint_pos")
        if not isinstance(cpu_command, MotionCommand):
            raise TypeError("cold Manager-Based construction did not build MotionCommand")
        if not isinstance(cpu_action, MotionJointPositionAction):
            raise TypeError("cold Manager-Based construction did not build motion action")
        robot = cpu_env.scene[command_cfg.entity_name]
        self._command = cpu_command
        self._action_cfg = action_cfg
        self._command_cfg = command_cfg
        self._reward_terms = reward_terms
        self._sampling_mode = command_cfg.sampling_mode
        self._robot = robot
        self._anchor_idx = command_cfg.body_names.index(command_cfg.anchor_body_name)
        self._body_names = tuple(command_cfg.body_names)
        self._body_ids = np.asarray(self._backend.get_body_ids(self._body_names), dtype=np.intp)
        self._joint_qpos_ids = np.asarray(
            self._backend.get_joint_state_qpos_indices(robot.joint_names), dtype=np.int64
        )
        self._joint_qvel_ids = np.asarray(
            self._backend.get_joint_state_qvel_indices(robot.joint_names), dtype=np.int64
        )
        target_ids = np.asarray(cpu_action._target_ids, dtype=np.int64)
        local_actuators = np.asarray(robot._joint_to_actuator_local, dtype=np.int64)[target_ids]
        self._action_to_actuator = np.asarray(robot._actuator_ids, dtype=np.int64)[local_actuators]
        self._identity_action_map = np.array_equal(
            self._action_to_actuator, np.arange(self._action_to_actuator.size, dtype=np.int64)
        )
        assert cfg.scene is not None
        robot_cfg = cfg.scene.entities["robot"]
        if not isinstance(robot_cfg, EntityCfg):
            raise TypeError("FlashSAC G1 robot scene declaration must be EntityCfg")
        if command_cfg.body_names[0] != robot_cfg.root_body_name:
            raise ValueError("canonical G1 motion body 0 must be the root body")

        self._terminations = {
            name: dict(term.params) for name, term in cfg.terminations.items() if term is not None
        }
        timeout_cfg = cfg.terminations["time_out"]
        clip_end_cfg = cfg.terminations["motion_clip_end"]
        if timeout_cfg is None or clip_end_cfg is None:
            raise ValueError("canonical FlashSAC timeout termination terms are required")
        self._terminations["time_out"]["time_out"] = bool(timeout_cfg.time_out)
        self._terminations["motion_clip_end"]["time_out"] = bool(clip_end_cfg.time_out)
        ee_names = tuple(self._terminations["ee_body_pos"]["body_names"])
        self._ee_ids = torch.tensor(
            [self._body_names.index(name) for name in ee_names],
            device=self.device,
            dtype=torch.int64,
        )
        undesired_names = (
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "right_hip_roll_link",
            "right_knee_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
        )
        self._undesired_ids = torch.tensor(
            [self._body_names.index(name) for name in undesired_names],
            device=self.device,
            dtype=torch.int64,
        )
        if cfg.max_episode_seconds is None:
            raise ValueError("FlashSAC G1 requires max_episode_seconds")
        self._max_episode_length = int(round(cfg.max_episode_seconds / cfg.ctrl_dt))
        action_scale = np.asarray(cpu_action._scale, dtype=np.float32)
        if action_scale.ndim == 0:
            action_scale = np.full(self._joint_qpos_ids.shape, action_scale, dtype=np.float32)
        elif action_scale.ndim == 2:
            # The cold Manager-Based resolver repeats one declared scale row per
            # environment. Its canonical semantic identity is checked above.
            action_scale = np.ascontiguousarray(action_scale[0])
        if action_scale.shape != self._joint_qpos_ids.shape or not np.all(
            np.isfinite(action_scale)
        ):
            raise ValueError("G1 tensor action scales must match joints and be finite")
        self._action_scale = _to_device(action_scale, self.device)
        self._actor_corruption = actor_group.enable_corruption
        noise_terms: list[UniformNoiseCfg] = []
        for name in ("base_lin_vel", "base_ang_vel", "joint_pos", "joint_vel"):
            noise_term = actor_group.terms[name]
            if noise_term is None:
                raise ValueError("FlashSAC G1 noisy observation terms are required")
            noise = noise_term.noise
            if not isinstance(noise, UniformNoiseCfg):
                raise TypeError(
                    f"FlashSAC G1 noisy observations require UniformNoiseCfg: {type(noise)}"
                )
            noise_terms.append(noise)
        noise_widths = (3, 3, 29, 29)
        for uniform_noise in noise_terms:
            _scalar_noise_bound(uniform_noise.n_min, label="noise minimum")
            _scalar_noise_bound(uniform_noise.n_max, label="noise maximum")
        self._observation_noise = TensorObservationNoise.from_uniform_terms(
            tuple(noise_terms), noise_widths, self.device
        )
        joint_pos_term = actor_group.terms["joint_pos"]
        if joint_pos_term is None:
            raise ValueError("G1 actor joint_pos term is required")
        self._actor_joint_pos_biased = (
            _qualified_name(joint_pos_term.func)
            == "unilab.tasks.motion_tracking.common.manager_terms.motion_joint_pos_rel_biased"
        )

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def cfg(self) -> ManagerBasedRlEnvCfg:
        return self._manager_cfg

    @property
    def state(self) -> TorchEnvState | None:
        return self._state

    @property
    def observation_space(self) -> Any:
        return self._observation_space

    @property
    def action_space(self) -> Any:
        return self._action_space

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return self._obs_groups_spec

    @property
    def play_capabilities(self) -> EnvPlayCapabilities:
        return EnvPlayCapabilities()

    def apply_action(self, actions: torch.Tensor, state: TorchEnvState) -> torch.Tensor:
        """Direct runtime actions are fused into :meth:`step`; not reusable."""
        raise NotImplementedError("TorchG1MotionTrackingFlashSACEnv fuses apply_action into step")

    def update_state(self, state: TorchEnvState) -> TorchEnvState:
        """Direct runtime state updates are fused into :meth:`step`; not reusable."""
        raise NotImplementedError("TorchG1MotionTrackingFlashSACEnv fuses update_state into step")

    def init_state(self) -> TorchEnvState:
        if self._state is not None:
            return self._state
        self._upload_cold_state()
        self._refresh_motion_buffers(self.current_frames)
        self._read_robot_state()
        self._refresh_relative_transforms()
        obs = self._compute_observations()
        self._state = TorchEnvState(
            obs=obs,
            reward=torch.zeros(self._num_envs, device=self.device),
            terminated=torch.zeros(self._num_envs, dtype=torch.bool, device=self.device),
            truncated=torch.zeros(self._num_envs, dtype=torch.bool, device=self.device),
            info={
                "log": {},
                "steps": torch.zeros(self._num_envs, dtype=torch.int64, device=self.device),
                "episode_metrics": self._episode_metrics.snapshot(),
            },
        )
        return self._state

    def _upload_cold_state(self) -> None:
        torch = self._torch
        command = self._command
        motion = command.motion
        robot_data = self._robot.data
        self._rng = torch.Generator(device=self.device)
        self._rng.manual_seed(int(self._initial_seed or 0))

        motion_fields = tuple(
            _to_device(value, self.device).reshape(value.shape[0], -1)
            for value in (
                motion.joint_pos,
                motion.joint_vel,
                motion.body_pos_w,
                motion.body_quat_w,
                motion.body_lin_vel_w,
                motion.body_ang_vel_w,
            )
        )
        self._motion_features = torch.cat(motion_fields, dim=1)
        self.current_frames = _to_device(command.sampler.current_frames, self.device)
        self._clip_ends = _to_device(command.sampler.current_clip_end_frames, self.device)
        self._bin_failed = torch.zeros(command.sampler.bin_count, device=self.device)
        self._adaptive_kernel = _to_device(command.sampler.kernel, self.device)
        self._clip_offsets = _to_device(command.motion.clip_offsets, self.device)
        self._clip_ends_store = _to_device(command.motion.clip_end_frames, self.device)
        self._joint_default_bias = _to_device(command.joint_default_bias, self.device)
        self._default_joint_pos = _to_device(robot_data.default_joint_pos, self.device)
        self._default_joint_vel = _to_device(robot_data.default_joint_vel, self.device)
        self._soft_limits = _to_device(robot_data.soft_joint_pos_limits, self.device)
        self._encoder_bias = _to_device(robot_data.encoder_bias, self.device)
        self._raw_actions = torch.zeros_like(self._joint_default_bias)
        self._previous_raw_actions = torch.zeros_like(self._raw_actions)
        self._steps = torch.zeros(self._num_envs, dtype=torch.int64, device=self.device)
        self._ctrl = torch.zeros(
            (self._num_envs, self._backend.num_actuators), dtype=torch.float32, device=self.device
        )
        self._pose_range = _to_device(command._pose_range, self.device)
        self._velocity_range = _to_device(command._velocity_range, self.device)
        cold_finite_tensors = (
            self._motion_features,
            self.current_frames,
            self._clip_ends,
            self._adaptive_kernel,
            self._clip_offsets,
            self._clip_ends_store,
            self._joint_default_bias,
            self._default_joint_pos,
            self._default_joint_vel,
            self._soft_limits,
            self._encoder_bias,
            self._pose_range,
            self._velocity_range,
        )
        if not _all_finite(*cold_finite_tensors):
            raise ValueError("Torch G1 FlashSAC cold state contains NaN or Inf")

        shape = (self._num_envs, len(self._body_names))
        motion_width = self._motion_features.shape[1]
        self._motion_state = torch.empty(
            (self._num_envs, motion_width), dtype=torch.float32, device=self.device
        )
        joint_width = motion_fields[0].shape[1]
        body_width = shape[1] * 3
        quat_width = shape[1] * 4
        cursor = 0
        self._motion_joint_pos = self._motion_state[:, cursor : cursor + joint_width]
        cursor += joint_width
        self._motion_joint_vel = self._motion_state[:, cursor : cursor + joint_width]
        cursor += joint_width
        self._motion_body_pos = self._motion_state[:, cursor : cursor + body_width].view(
            self._num_envs, shape[1], 3
        )
        cursor += body_width
        self._motion_body_quat = self._motion_state[:, cursor : cursor + quat_width].view(
            self._num_envs, shape[1], 4
        )
        cursor += quat_width
        self._motion_body_lin_vel = self._motion_state[:, cursor : cursor + body_width].view(
            self._num_envs, shape[1], 3
        )
        cursor += body_width
        self._motion_body_ang_vel = self._motion_state[:, cursor:motion_width].view(
            self._num_envs, shape[1], 3
        )
        self._body_pos_relative = torch.empty_like(self._motion_body_pos)
        self._body_quat_relative = torch.empty_like(self._motion_body_quat)
        self._motion_anchor_pos_b = torch.empty((self._num_envs, 3), device=self.device)
        self._motion_anchor_ori_b = torch.empty((self._num_envs, 6), device=self.device)
        self._robot_body_pos_b = torch.empty_like(self._robot_body_pos)
        self._robot_body_ori_b = torch.empty((*shape, 6), device=self.device)
        self._obs = {
            name: torch.empty((self._num_envs, dim), device=self.device)
            for name, dim in self.obs_groups_spec.items()
        }
        self._final_observation = {
            name: torch.empty_like(value) for name, value in self._obs.items()
        }

    def _motion_at(self, frames: torch.Tensor) -> tuple[torch.Tensor, ...]:
        selected = torch.index_select(self._motion_features, 0, frames.long())
        joint_width = self._motion_joint_pos.shape[1]
        body_count = self._motion_body_pos.shape[1]
        body_width = body_count * 3
        quat_width = body_count * 4
        cursor = 0
        values = []
        for width, body_shape in (
            (joint_width, (joint_width,)),
            (joint_width, (joint_width,)),
            (body_width, (body_count, 3)),
            (quat_width, (body_count, 4)),
            (body_width, (body_count, 3)),
            (body_width, (body_count, 3)),
        ):
            values.append(selected[:, cursor : cursor + width].view(-1, *body_shape))
            cursor += width
        return tuple(values)

    def _refresh_motion_buffers(
        self, frames: torch.Tensor, rows: torch.Tensor | None = None
    ) -> None:
        selected = torch.index_select(self._motion_features, 0, frames.long())
        target = slice(None) if rows is None else rows
        self._motion_state[target] = selected

    def _read_robot_state(self, rows: torch.Tensor | None = None) -> dict | None:
        store = self._state_store
        store.read(rows)
        qpos, qvel = store.qviews()
        self._qpos = qpos
        self._qvel = qvel
        self._joint_pos = store.joint_pos
        self._joint_vel = store.joint_vel
        self._linvel = store.linvel
        self._gyro = store.gyro
        self._robot_body_pos = store.robot_body_pos
        self._robot_body_quat = store.robot_body_quat
        self._robot_body_lin_vel = store.robot_body_lin_vel
        self._robot_body_ang_vel = store.robot_body_ang_vel
        return store.last_backend_result

    def _refresh_motion_relative_transforms(self, rows: torch.Tensor | None = None) -> None:
        target = slice(None) if rows is None else rows
        anchor = self._anchor_idx
        m_anchor_pos = self._motion_body_pos[target, anchor]
        m_anchor_quat = self._motion_body_quat[target, anchor]
        r_anchor_pos = self._robot_body_pos[target, anchor]
        r_anchor_quat = self._robot_body_quat[target, anchor]
        delta = _yaw_quat(r_anchor_quat, m_anchor_quat)
        rotated_body_quat = _quat_mul(
            delta[None, :] if delta.ndim == 1 else delta[:, None, :], self._motion_body_quat[target]
        )
        self._body_quat_relative[target] = rotated_body_quat
        local = self._motion_body_pos[target] - m_anchor_pos[:, None, :]
        rotated_local = _quat_apply(delta[None, :] if delta.ndim == 1 else delta[:, None, :], local)
        self._body_pos_relative[target] = rotated_local + r_anchor_pos[:, None, :]
        relative = self._body_pos_relative[target]
        relative[..., 2] = self._motion_body_pos[target][..., 2]
        self._body_pos_relative[target] = relative
        anchor_delta = m_anchor_pos - r_anchor_pos
        self._motion_anchor_pos_b[target] = _quat_apply_inverse(r_anchor_quat, anchor_delta)
        self._motion_anchor_ori_b[target] = _rot6(
            _quat_mul(_quat_inv(r_anchor_quat), m_anchor_quat)
        )

    def _refresh_robot_relative_transforms(self, rows: torch.Tensor | None = None) -> None:
        target = slice(None) if rows is None else rows
        anchor = self._anchor_idx
        r_anchor_pos = self._robot_body_pos[target, anchor]
        r_anchor_quat = self._robot_body_quat[target, anchor]
        robot_local = self._robot_body_pos[target] - r_anchor_pos[:, None, :]
        self._robot_body_pos_b[target] = _quat_apply_inverse(r_anchor_quat[:, None, :], robot_local)
        self._robot_body_ori_b[target] = _rot6(
            _quat_mul(_quat_inv(r_anchor_quat[:, None, :]), self._robot_body_quat[target])
        )

    def _refresh_relative_transforms(self, rows: torch.Tensor | None = None) -> None:
        self._refresh_motion_relative_transforms(rows)
        self._refresh_robot_relative_transforms(rows)

    def _exp_error(self, error: torch.Tensor, std: float) -> torch.Tensor:
        return torch.exp(error / (-(std * std)))

    def _compute_terminations(self) -> torch.Tensor:
        anchor_cfg = self._terminations["anchor_pos"]
        ee_cfg = self._terminations["ee_body_pos"]
        anchor_idx = self._anchor_idx
        anchor_pos_bad = (
            self._motion_body_pos[:, anchor_idx, 2] - self._robot_body_pos[:, anchor_idx, 2]
        ).abs() > float(anchor_cfg["threshold"])
        anchor_ori_cfg = self._terminations["anchor_ori"]
        motion_z = _gravity_z_in_body(self._motion_body_quat[:, anchor_idx])
        robot_z = _gravity_z_in_body(self._robot_body_quat[:, anchor_idx])
        anchor_ori_bad = (motion_z - robot_z).abs() > float(anchor_ori_cfg["threshold"])
        ee_bad = torch.any(
            (
                self._body_pos_relative[:, self._ee_ids, 2]
                - self._robot_body_pos[:, self._ee_ids, 2]
            ).abs()
            > float(ee_cfg["threshold"]),
            dim=-1,
        )
        failure = anchor_pos_bad | anchor_ori_bad | ee_bad
        undesired_cfg = self._terminations.get("undesired_contacts")
        if undesired_cfg is not None:
            failure |= torch.any(
                self._robot_body_pos[:, self._undesired_ids, 2] < float(undesired_cfg["threshold"]),
                dim=-1,
            )
        return failure

    def _compute_reward(self) -> torch.Tensor:
        anchor = self._anchor_idx
        weighted_terms: list[torch.Tensor] = []
        self.last_reward_terms: dict[str, torch.Tensor] = {}

        def add_term(name: str, value: torch.Tensor) -> None:
            self.last_reward_terms[name] = value
            weighted_terms.append(float(self._reward_terms[name].weight) * value)

        pos_error = (
            (self._motion_body_pos[:, anchor] - self._robot_body_pos[:, anchor])
            .square()
            .sum(dim=-1)
        )
        add_term(
            "motion_global_root_pos",
            self._exp_error(
                pos_error, float(self._reward_terms["motion_global_root_pos"].params["std"])
            ),
        )
        ori_error = _quat_error_squared(
            self._motion_body_quat[:, anchor], self._robot_body_quat[:, anchor]
        )
        add_term(
            "motion_global_root_ori",
            self._exp_error(
                ori_error, float(self._reward_terms["motion_global_root_ori"].params["std"])
            ),
        )
        body_specs = (
            ("motion_body_pos", self._body_pos_relative, self._robot_body_pos),
            ("motion_body_ori", self._body_quat_relative, self._robot_body_quat),
            ("motion_body_lin_vel", self._motion_body_lin_vel, self._robot_body_lin_vel),
            ("motion_body_ang_vel", self._motion_body_ang_vel, self._robot_body_ang_vel),
        )
        for name, reference, actual in body_specs:
            std = float(self._reward_terms[name].params["std"]) * (reference.shape[1] ** 0.5)
            if reference is self._body_quat_relative:
                error = _quat_error_squared(reference, actual).sum(dim=-1)
            else:
                error = (reference - actual).square().sum(dim=(-1, -2))
            add_term(name, self._exp_error(error, std))

        if "motion_ee_body_pos_z" in self._reward_terms:
            ee_error = (
                (
                    self._body_pos_relative[:, self._ee_ids, 2]
                    - self._robot_body_pos[:, self._ee_ids, 2]
                )
                .square()
                .mean(dim=-1)
            )
            add_term(
                "motion_ee_body_pos_z",
                self._exp_error(
                    ee_error, float(self._reward_terms["motion_ee_body_pos_z"].params["std"])
                ),
            )

        action_rate = (self._raw_actions - self._previous_raw_actions).square().sum(dim=-1)
        add_term("action_rate_l2", action_rate)
        lower_violation = (self._soft_limits[..., 0] - self._joint_pos).clamp_min(0.0)
        upper_violation = (self._joint_pos - self._soft_limits[..., 1]).clamp_min(0.0)
        joint_violation = (lower_violation + upper_violation).square().sum(dim=-1)
        add_term("joint_limit", joint_violation)

        contact_cfg = self._reward_terms["undesired_contacts"].params
        contacts = (
            self._robot_body_pos[:, self._undesired_ids, 2] < float(contact_cfg["threshold"])
        ).sum(dim=-1)
        add_term("undesired_contacts", contacts.to(torch.float32))

        reward = weighted_terms[0]
        for term in weighted_terms[1:]:
            reward = reward + term
        return reward * self._cfg.ctrl_dt

    def _compute_observations(
        self, *, corrupt: bool | None = None, rows: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        target = slice(None) if rows is None else rows
        command = torch.cat(
            (self._motion_joint_pos[target], self._motion_joint_vel[target]), dim=-1
        )
        joint_pos = (
            self._joint_pos[target]
            - self._default_joint_pos[target]
            - self._joint_default_bias[target]
        )
        joint_vel = self._joint_vel[target] - self._default_joint_vel[target]
        linvel = self._linvel[target]
        gyro = self._gyro[target]
        segments = {
            "command": command,
            "motion_anchor_pos_b": self._motion_anchor_pos_b[target],
            "motion_anchor_ori_b": self._motion_anchor_ori_b[target],
            "base_lin_vel": linvel,
            "base_ang_vel": gyro,
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "actions": self._raw_actions[target],
        }
        critic_prefix = torch.cat(
            tuple(segments[name] for name in self._critic_prefix_names), dim=-1
        )
        actor = torch.cat(
            (
                command,
                self._motion_anchor_pos_b[target],
                self._motion_anchor_ori_b[target],
                linvel,
                gyro,
                joint_pos,
                joint_vel,
                self._raw_actions[target],
            ),
            dim=-1,
        )
        if self._actor_joint_pos_biased:
            joint_cursor = (
                command.shape[-1]
                + self._motion_anchor_pos_b.shape[-1]
                + self._motion_anchor_ori_b.shape[-1]
                + linvel.shape[-1]
                + gyro.shape[-1]
            )
            actor[..., joint_cursor : joint_cursor + self._encoder_bias.shape[-1]] += (
                self._encoder_bias[target]
            )
        noisy = self._actor_corruption if corrupt is None else corrupt
        if noisy:
            cursor = 58 + 3 + 6
            actor = self._observation_noise.apply(actor, cursor=cursor, generator=self._rng)
        critic = torch.cat(
            (
                critic_prefix,
                self._robot_body_pos_b[target].reshape(target_length := actor.shape[0], -1),
                self._robot_body_ori_b[target].reshape(target_length, -1),
                linvel,
            ),
            dim=-1,
        )
        return {"obs": actor, "critic": critic}

    def _sample_frames(self, rows: torch.Tensor) -> torch.Tensor:
        if self._sampling_mode == "mixed":
            count = rows.numel()
            use_start = torch.rand(count, device=self.device, generator=self._rng) < float(
                self._command_cfg.params.sampling_start_ratio
            )
            uniform_frames = torch.randint(
                0,
                self._motion_features.shape[0],
                (count,),
                device=self.device,
                generator=self._rng,
                dtype=torch.int32,
            )
            frames = torch.where(use_start, torch.zeros_like(uniform_frames), uniform_frames)
            self.current_frames[rows] = frames
            clip_indices = (
                torch.searchsorted(self._clip_offsets.int(), frames.int(), right=True) - 1
            )
            clip_indices = clip_indices.clamp_(min=0)
            self._clip_ends[rows] = self._clip_ends_store[clip_indices]
            return frames

        probs = (
            self._bin_failed
            + float(self._command_cfg.params.adaptive_uniform_ratio) / self._bin_failed.numel()
        )
        kernel_size = int(self._command_cfg.params.adaptive_kernel_size)
        if kernel_size > 1:
            probs = torch.nn.functional.pad(probs, (0, kernel_size - 1), mode="replicate")
            probs = torch.nn.functional.conv1d(
                probs[None, None, :], self._adaptive_kernel[None, None, :], padding=0
            )[0, 0]
        probs = probs / probs.sum()
        bins = torch.multinomial(probs, rows.numel(), replacement=True, generator=self._rng)
        offsets = torch.rand(rows.numel(), device=self.device, generator=self._rng)
        frames = (
            (bins.to(torch.float32) + offsets)
            / self._bin_failed.numel()
            * (self._motion_features.shape[0] - 1)
        ).to(torch.int32)
        self.current_frames[rows] = frames
        clip_indices = (
            torch.searchsorted(
                self._clip_offsets.int(),
                frames.int(),
                right=True,
            )
            - 1
        )
        clip_indices = clip_indices.clamp_(min=0)
        self._clip_ends[rows] = self._clip_ends_store[clip_indices]
        return frames

    def _reset_rows(
        self, rows: torch.Tensor, state_obs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        torch = self._torch
        if rows.numel() == 0:
            return state_obs
        frames = self._sample_frames(rows)
        motion = self._motion_at(frames)
        count = rows.numel()
        pose = torch.rand((count, 6), device=self.device, generator=self._rng)
        pose = (
            pose * (self._pose_range[None, :, 1] - self._pose_range[None, :, 0])
            + self._pose_range[None, :, 0]
        )
        velocity = torch.rand((count, 6), device=self.device, generator=self._rng)
        velocity = (
            velocity * (self._velocity_range[None, :, 1] - self._velocity_range[None, :, 0])
            + self._velocity_range[None, :, 0]
        )
        root_pos = motion[2][:, 0] + pose[:, :3]
        root_quat = _quat_mul(_euler_xyz_quat(pose[:, 3], pose[:, 4], pose[:, 5]), motion[3][:, 0])
        root_lin_vel = motion[4][:, 0] + velocity[:, :3]
        root_ang_vel = motion[5][:, 0] + velocity[:, 3:]
        joint_range = self._command_cfg.params.joint_position_range
        joint_noise_scale = float(joint_range[1] - joint_range[0])
        joint_noise = torch.rand(
            (count, self._motion_joint_pos.shape[1]),
            device=self.device,
            generator=self._rng,
        ) * joint_noise_scale + float(joint_range[0])
        joint_pos = motion[0] + joint_noise
        joint_pos = joint_pos.clamp(self._soft_limits[:, 0], self._soft_limits[:, 1])
        qpos_view = self._qpos
        qvel_view = self._qvel
        if qpos_view is None or qvel_view is None:
            raise RuntimeError("Torch G1 FlashSAC selected reset requires initialized state views")
        qpos = qpos_view.index_select(0, rows).clone()
        qvel = qvel_view.index_select(0, rows).clone()
        qpos[:, :3] = root_pos
        qpos[:, 3:7] = root_quat
        qpos[:, self._joint_qpos_ids] = joint_pos
        qvel[:, :3] = root_lin_vel
        qvel[:, 3:6] = root_ang_vel
        qvel[:, self._joint_qvel_ids] = motion[1]
        if not _all_finite(qpos, qvel):
            raise ValueError("Torch G1 FlashSAC reset qpos/qvel contain NaN or Inf")
        qpos_view[rows] = qpos
        qvel_view[rows] = qvel
        default_range = self._command_cfg.params.joint_default_position_range
        default_noise_scale = float(default_range[1] - default_range[0])
        self._joint_default_bias[rows] = torch.rand(
            (count, self._joint_default_bias.shape[1]), device=self.device, generator=self._rng
        ) * default_noise_scale + float(default_range[0])
        self._raw_actions[rows] = 0.0
        self._previous_raw_actions[rows] = 0.0
        self._ctrl[rows] = 0.0
        self._steps[rows] = 0
        self._episode_metrics.reset(rows)
        self._refresh_motion_buffers(frames, rows)
        backend_result = self._state_store.apply_reset(rows, qpos, qvel)
        read_result = self._read_robot_state(rows)
        combined_timing: dict[str, float] = {}
        if isinstance(backend_result, dict):
            combined_timing.update(backend_result.get("timing", {}))
        if isinstance(read_result, dict):
            combined_timing.update(read_result.get("timing", {}))
        self._last_backend_reset_result = {"timing": combined_timing}
        self._refresh_motion_relative_transforms(rows)
        self._refresh_robot_relative_transforms(rows)
        reset_obs = self._compute_observations(rows=rows)
        for name in state_obs:
            state_obs[name][rows] = reset_obs[name]
        return state_obs

    def step(self, actions: torch.Tensor | np.ndarray) -> TorchEnvState:
        torch = self._torch
        if self._state is None:
            raise RuntimeError("call init_state() before step()")
        action = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        if action.shape != self._raw_actions.shape:
            raise ValueError(
                f"expected action shape {tuple(self._raw_actions.shape)}, got {tuple(action.shape)}"
            )

        timing: dict[str, float] = {}
        started = perf_counter()
        self._previous_raw_actions.copy_(self._raw_actions)
        self._raw_actions.copy_(action)
        processed = self._raw_actions * self._action_scale
        if self._action_cfg.use_default_offset:
            processed = processed + self._default_joint_pos
        target = processed + self._joint_default_bias - self._encoder_bias
        if self._identity_action_map:
            self._ctrl.copy_(target)
        else:
            self._ctrl.zero_()
            self._ctrl[:, self._action_to_actuator] = target
        if not _all_finite(action, self._ctrl):
            if not bool(torch.isfinite(action).all()):
                raise ValueError("actions contain NaN or Inf")
            raise ValueError("transformed controls contain NaN or Inf")
        backend_result = self._state_store.step_tensor(self._ctrl, self._cfg.sim_substeps)
        if isinstance(backend_result, dict):
            timing.update(backend_result.get("timing", {}))
        timing["action_backend_step_ms"] = (perf_counter() - started) * 1000.0

        phase = perf_counter()
        self._steps += 1
        read_result = self._read_robot_state()
        if isinstance(read_result, dict):
            timing.update(read_result.get("timing", {}))
        self._refresh_robot_relative_transforms()
        terminated = self._compute_terminations()
        clip_end = self.current_frames >= self._clip_ends
        timeout = self._steps >= self._max_episode_length
        truncated = clip_end | timeout
        reward = self._compute_reward()

        if self._sampling_mode == "adaptive":
            bin_indices = (
                self.current_frames.to(torch.int64)
                * self._bin_failed.numel()
                // self._motion_features.shape[0]
            ).clamp_(max=self._bin_failed.numel() - 1)
            n_bins = self._bin_failed.numel()
            failures = _adaptive_failure_counts(bin_indices, terminated, n_bins)
            failure_alpha = _adaptive_failure_alpha(
                terminated, self._command_cfg.params.adaptive_alpha
            ).to(dtype=self._bin_failed.dtype)
            self._bin_failed.mul_(1.0 - failure_alpha).add_(failures * failure_alpha)
        active = ~(terminated | truncated)
        self.current_frames += active.to(torch.int32)
        self._refresh_motion_buffers(self.current_frames)
        self._refresh_motion_relative_transforms()
        obs = self._compute_observations()
        if not _all_finite(reward, obs["obs"], obs["critic"]):
            raise ValueError("Torch G1 tensor reward or observations contain NaN or Inf")
        timing["update_state_ms"] = (perf_counter() - phase) * 1000.0

        phase = perf_counter()
        reset_plan = TensorResetPlan(terminated=terminated, truncated=truncated)
        done = reset_plan.done
        self._episode_metrics.update(reward, done)
        final_obs = None
        rows = reset_plan.rows
        finished_episode_metrics = self._episode_metrics.finished_values(rows)
        if rows.numel():
            final_obs = self._final_observation
            for name, values in obs.items():
                final_obs[name].index_copy_(0, rows, values.index_select(0, rows))
        obs = self._reset_rows(rows, obs)
        if rows.numel() and isinstance(self._last_backend_reset_result, dict):
            timing.update(self._last_backend_reset_result.get("timing", {}))
        timing["reset_done_ms"] = (perf_counter() - phase) * 1000.0
        timing["step_ms"] = (perf_counter() - started) * 1000.0

        self._state = TorchEnvState(
            obs=obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info={
                "log": {},
                "steps": self._steps.clone(),
                "timing": timing,
                "finished_episode_metrics": finished_episode_metrics,
                "episode_metrics": self._episode_metrics.snapshot(),
            },
            final_observation=final_obs,
        )
        return self._state

    def reset(
        self,
        env_indices: torch.Tensor | np.ndarray | None = None,
        *,
        seed: int | None = None,
        env_ids: torch.Tensor | np.ndarray | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        del options
        if env_indices is not None and env_ids is not None:
            raise ValueError("pass either env_indices or env_ids, not both")
        source = env_indices if env_indices is not None else env_ids
        if seed is not None:
            self._initial_seed = seed
            if self._state is not None:
                self._rng.manual_seed(seed)
        if self._state is None:
            state = self.init_state()
            return {name: value.clone() for name, value in state.obs.items()}, {"log": {}}
        rows = (
            torch.arange(self._num_envs, device=self.device, dtype=torch.int64)
            if source is None
            else torch.as_tensor(source, dtype=torch.int64, device=self.device)
        )
        obs = self._reset_rows(rows, {k: v.clone() for k, v in self._state.obs.items()})
        self._state.terminated[rows] = False
        self._state.truncated[rows] = False
        return {name: values[rows].clone() for name, values in obs.items()}, {"log": {}}

    def close(self) -> None:
        """Release owner-held tensor views before backend cleanup.

        A CUDA IPC plan correctly refuses to detach while caller-held views
        remain.  This owner therefore relinquishes its hot-path view aliases
        first, without copying or moving tensor state.  Views retained through
        another public API reference intentionally keep the backend's fail-closed
        cleanup check active.
        """
        view_refs = (
            "_state_store",
            "_qpos",
            "_qvel",
            "_joint_pos",
            "_joint_vel",
            "_state",
        )
        for name in view_refs:
            object.__setattr__(self, name, None)
        # Make Python release the task-owner views now. CUDA IPC view liveness
        # is refcount-based; the next backend cleanup barrier depends on these
        # decrements reaching zero before close executes.
        gc.collect()
        self._backend.cleanup_scene_assets()

    cleanup = close


def make_torch_g1_motion_tracking_flashsac_env(
    cfg: ManagerBasedRlEnvCfg,
    num_envs: int = 1,
    backend_type: str = "mujoco",
) -> ABEnv:
    """Registry factory using the normal Manager-Based cold path and assets."""
    from unilab.envs import make_manager_based_rl_env

    if not isinstance(cfg, ManagerBasedRlEnvCfg):
        raise TypeError("Torch G1 FlashSAC factory expected ManagerBasedRlEnvCfg")
    if not cfg.tensor_runtime:
        env = make_manager_based_rl_env(cfg, num_envs=num_envs, backend_type=backend_type)
        return env
    _validate_torch_g1_flashsac_owner_contract(cfg)
    cfg.validate()
    assert cfg.scene is not None
    apply_env_cpu_runtime(cfg.cpu_ids)
    base_name, body_state_requested, tracked_body_names = _resolve_backend_entity_contract(cfg)
    kwargs = env_backend_kwargs(cfg)
    kwargs["base_name"] = base_name
    if backend_type == "mujoco" and tracked_body_names is not None:
        kwargs["tracked_body_names"] = tracked_body_names
    backend = create_backend(
        backend_type,
        cfg.scene,
        num_envs,
        cfg.sim_dt,
        body_state_required=body_state_requested,
        **kwargs,
    )
    try:
        return TorchG1MotionTrackingFlashSACEnv(cfg, backend, num_envs, device="cuda")
    except BaseException:
        backend.cleanup_scene_assets()
        raise


__all__ = [
    "TorchG1MotionTrackingFlashSACEnv",
    "make_torch_g1_motion_tracking_flashsac_env",
]
