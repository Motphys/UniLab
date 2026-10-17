"""Go2 owner terms for tensor-resident locomotion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch

from unilab.managers.manager_base import ManagerTermBase
from unilab.managers.reward_manager import RewardTermCfg
from unilab.managers.scene_entity_config import SceneEntityCfg
from unilab.tasks.locomotion.common.manager_terms import _command, _real

if TYPE_CHECKING:
    from unilab.base.entity import Entity
    from unilab.managers._types import ManagerBasedRlEnv

    class _Go2Env(ManagerBasedRlEnv, Protocol):
        @property
        def common_step_counter(self) -> int: ...


_CONTACT_SENSORS = (
    "FL_foot_contact",
    "FR_foot_contact",
    "RL_foot_contact",
    "RR_foot_contact",
)
_FOOT_POSITION_SENSORS = ("FL_pos", "FR_pos", "RL_pos", "RR_pos")
_DEFAULT_OFFSETS = (0.0, 0.5, 0.5, 0.0)


@dataclass(kw_only=True)
class Go2WalkRewardPackCfg(RewardTermCfg):
    """Fused canonical Go2 flat-walk reward family."""

    entity_name: str = "robot"
    root_body_name: str = "base"
    command_name: str = "twist"
    tracking_sigma: float = 0.25
    target_height: float = 0.3
    gait_frequency: float = 2.0
    phase_offsets: tuple[float, ...] = _DEFAULT_OFFSETS
    contact_threshold: float = 0.1
    stance_threshold: float = 0.6
    swing_target_height: float = 0.1
    swing_kernel: float = 0.01
    swing_start: float = 0.6


class Go2WalkRewardPack(ManagerTermBase):
    """Evaluate the Go2 reward family with one packed carrier read."""

    returns_transient_tensor = True
    last_step_timing_ms: dict[str, float] = {}

    def __init__(self, cfg: Go2WalkRewardPackCfg, env: _Go2Env):
        super().__init__(env)
        if not isinstance(cfg, Go2WalkRewardPackCfg):
            raise TypeError("Go2WalkRewardPack requires Go2WalkRewardPackCfg")
        self.cfg = cfg
        self._device = torch.device(env.device)
        self._entity_name = cfg.entity_name
        self._root_body_name = cfg.root_body_name
        self._tracking_sigma = _real(
            "Go2WalkRewardPack",
            "tracking_sigma",
            cfg.tracking_sigma,
            minimum=0.0,
            strict_minimum=True,
        )
        self._target_height = _real("Go2WalkRewardPack", "target_height", cfg.target_height)
        self._contact_threshold = _real(
            "Go2WalkRewardPack",
            "contact_threshold",
            cfg.contact_threshold,
            minimum=0.0,
        )
        self._stance_threshold = _real(
            "Go2WalkRewardPack",
            "stance_threshold",
            cfg.stance_threshold,
            minimum=0.0,
            maximum=1.0,
        )
        self._swing_target = _real(
            "Go2WalkRewardPack",
            "swing_target_height",
            cfg.swing_target_height,
            minimum=0.0,
        )
        self._swing_kernel = _real(
            "Go2WalkRewardPack",
            "swing_kernel",
            cfg.swing_kernel,
            minimum=0.0,
            strict_minimum=True,
        )
        self._swing_start = _real(
            "Go2WalkRewardPack",
            "swing_start",
            cfg.swing_start,
            minimum=0.0,
            maximum=1.0,
        )
        self._frequency = _real(
            "Go2WalkRewardPack",
            "gait_frequency",
            cfg.gait_frequency,
            minimum=0.0,
        )
        self._offsets = torch.tensor(cfg.phase_offsets, dtype=torch.float32, device=self._device)
        if self._offsets.shape != (4,):
            raise ValueError("Go2WalkRewardPack phase_offsets must contain 4 values")
        self._step_dt = _real(
            "Go2WalkRewardPack", "step_dt", env.step_dt, minimum=0.0, strict_minimum=True
        )
        self._phase_value = torch.zeros(env.num_envs, dtype=torch.float32, device=self._device)
        self._last_counter = int(env.common_step_counter)
        _command(env, "Go2WalkRewardPack", cfg.command_name)
        self._entity = env.scene[cfg.entity_name]
        self._asset_cfg = SceneEntityCfg(cfg.entity_name)
        sensor_names = self.tensor_sensor_names
        try:
            sensor_view = env.scene.bind_sensor_data(sensor_names)
        except (KeyError, TypeError, ValueError, NotImplementedError) as exc:
            raise type(exc)(
                f"Go2WalkRewardPack named-sensor capability failed for {sensor_names}: {exc}"
            ) from exc
        dimensions = sensor_view.dimensions
        if len(dimensions) != 9:
            raise ValueError("Go2WalkRewardPack requires its 9 declared sensors")
        if dimensions[0] != 3:
            raise ValueError("Go2WalkRewardPack gyro must expose 3 values")
        contact_dimensions = dimensions[1:5]
        if any(width not in (1, 3) for width in contact_dimensions):
            raise ValueError("Go2WalkRewardPack contact sensors must expose 1 or 3 values")
        if any(width != 3 for width in dimensions[5:9]):
            raise ValueError("Go2WalkRewardPack foot-position sensors must expose xyz values")

    @property
    def tensor_sensor_names(self) -> tuple[str, ...]:
        return ("gyro", *_CONTACT_SENSORS, *_FOOT_POSITION_SENSORS)

    @property
    def tensor_body_names(self) -> tuple[str, ...]:
        return (self._root_body_name,)

    @property
    def entity_name(self) -> str:
        return self._entity_name

    @property
    def tensor_state_entity(self) -> str:
        return self._entity_name

    def reset(self, reset_mask: torch.Tensor) -> None:
        del reset_mask

    def _quat_apply_inverse(self, quat_wxyz: torch.Tensor, vector_w: torch.Tensor) -> torch.Tensor:
        quat = torch.cat((quat_wxyz[..., 0:1], -quat_wxyz[..., 1:4]), dim=-1)
        quat_xyz = quat[..., 1:4]
        cross = torch.linalg.cross(quat_xyz, vector_w, dim=-1)
        second_cross = torch.linalg.cross(quat_xyz, cross, dim=-1)
        return vector_w + 2.0 * quat[..., 0:1] * cross + 2.0 * second_cross

    def _phase(self, env: _Go2Env) -> torch.Tensor:
        counter = int(env.common_step_counter)
        delta = counter - self._last_counter
        if delta < 0:
            raise ValueError("Go2WalkRewardPack common_step_counter cannot move backwards")
        increment = self._step_dt * self._frequency
        if delta == 1:
            self._phase_value = torch.fmod(self._phase_value + increment, 1.0)
        elif delta > 1:
            for _ in range(delta):
                self._phase_value = torch.fmod(self._phase_value + increment, 1.0)
        self._last_counter = counter
        return torch.remainder(self._phase_value[:, None] + self._offsets[None, :], 1.0)

    def __call__(self, env: _Go2Env) -> torch.Tensor:
        read_plan = getattr(env.scene, "_tensor_read_plan", None)
        if read_plan is None:
            raise NotImplementedError("Go2WalkRewardPack requires the packed tensor read plan")
        entity = env.scene[self._entity_name]
        values = read_plan.sensor_tensor_views(entity, self.tensor_sensor_names).values
        body = read_plan.body_tensor_view(self._entity_name, (self._root_body_name,))
        root_velocity_w = body.lin_vel_w[:, 0, :]
        root_quat_w = body.quat_w[:, 0, :]
        root_velocity_b = self._quat_apply_inverse(root_quat_w, root_velocity_w)
        command = _command(env, "Go2WalkRewardPack", self.cfg.command_name).to(
            device=root_velocity_b.device, dtype=root_velocity_b.dtype
        )
        tracking_lin_error = (command[:, :2] - root_velocity_b[:, :2]).square().sum(dim=1)
        tracking_lin_vel = torch.exp(-tracking_lin_error / self._tracking_sigma)
        gyro = values["gyro"]
        tracking_ang_error = (command[:, 2] - gyro[:, 2]).square()
        tracking_ang_vel = torch.exp(-tracking_ang_error / self._tracking_sigma)
        lin_vel_z = root_velocity_b[:, 2].square()
        ang_vel_xy = gyro[:, :2].square().sum(dim=1)
        base_height = (body.pos_w[:, 0, 2] - self._target_height).square()
        actions = env.action_manager.action - env.action_manager.prev_action
        action_rate = actions.square().sum(dim=1)
        joint_position = read_plan.joint_tensor_view(self._entity_name).joint_pos
        default_position = self._entity.data.default_joint_pos_torch(env.device)
        similar_to_default = (joint_position - default_position).abs().sum(dim=1)
        phase = self._phase(env)
        contact = (
            torch.stack(
                tuple(values[name][:, 0] for name in _CONTACT_SENSORS),
                dim=1,
            )
            > self._contact_threshold
        )
        expected_contact = phase < self._stance_threshold
        if self._frequency < 1.0e-8:
            expected_contact.fill_(True)
        contact_reward = (contact == expected_contact).to(torch.float32).mean(dim=1)
        heights = torch.stack(
            tuple(values[name][:, 2] for name in _FOOT_POSITION_SENSORS),
            dim=1,
        )
        swing = phase >= self._swing_start
        swing_reward = (
            torch.exp(-(heights - self._swing_target).square() / self._swing_kernel)
            * swing.to(torch.float32)
        ).mean(dim=1)
        return torch.stack(
            (
                tracking_lin_vel,
                tracking_ang_vel,
                lin_vel_z,
                ang_vel_xy,
                base_height,
                action_rate,
                similar_to_default,
                contact_reward,
                swing_reward,
            ),
            dim=1,
        )


__all__ = ["Go2WalkRewardPack", "Go2WalkRewardPackCfg"]
