"""Opt-in command control and evaluation-only locomotion diagnostics."""

import math

import torch

DEFAULT_TASK = "go2_joystick_flat/rma"
FORWARD_TASK = "go2_joystick_flat/rma_forward"
NO_POSE_TASK = "go2_joystick_flat/rma_forward_no_pose"
TIGHT_TRACKING_TASK = "go2_joystick_flat/rma_forward_tight_tracking"
RMA_TASKS = (DEFAULT_TASK, FORWARD_TASK, NO_POSE_TASK, TIGHT_TRACKING_TASK)


def command_overrides(command):
    """Fix commands through their config owner, including after every reset."""
    if len(command) != 3 or not all(math.isfinite(float(x)) for x in command):
        raise ValueError("command must contain finite vx, vy, and yaw-rate values")
    prefix = "env.commands.twist."
    overrides = [
        prefix + f"ranges.{name}=[{float(value)},{float(value)}]"
        for name, value in zip(("lin_vel_x", "lin_vel_y", "ang_vel_z"), command)
    ]
    overrides += [prefix + "heading_command=false"]
    overrides += [
        prefix + name + "=0.0"
        for name in ("rel_standing_envs", "rel_heading_envs", "rel_world_envs", "rel_forward_envs")
    ]
    return overrides


class EvaluationDiagnostics:
    """Accumulate clean critic velocities; never alter actions or observations."""

    def __init__(self):
        self.count = 0
        self.command_sum = torch.zeros(3, dtype=torch.float64)
        self.measured_sum = torch.zeros(3, dtype=torch.float64)
        self.error_squared_sum = torch.zeros(3, dtype=torch.float64)
        self.absolute_yaw_sum = 0.0
        self.initial_command = None
        self.reward_sum = {}
        self.reward_steps = 0

    def record_state(self, critic):
        # critic[42:45] is command; [49:52] is clean local_linvel;
        # critic[0:3] is clean gyro. Policy gyro includes training noise.
        state = critic.detach().to(device="cpu", dtype=torch.float64)
        if state.ndim != 2 or state.shape[1] != 52:
            raise ValueError("Diagnostics require the current 52-dimensional Go2 critic contract")
        command = state[:, 42:45]
        measured = torch.cat((state[:, 49:51], state[:, 2:3]), dim=1)
        if self.initial_command is None:
            self.initial_command = command[0].tolist()
        self.command_sum += command.sum(0)
        self.measured_sum += measured.sum(0)
        self.error_squared_sum += (command - measured).square().sum(0)
        self.absolute_yaw_sum += measured[:, 2].abs().sum().item()
        self.count += len(state)

    def record_rewards(self, reward_manager):
        # Public API: already weighted, but NOT multiplied by the control dt.
        means = reward_manager.step_reward_means.detach().cpu().tolist()
        for name, mean in zip(reward_manager.step_reward_names, means, strict=True):
            self.reward_sum[name] = self.reward_sum.get(name, 0.0) + mean
        self.reward_steps += 1

    def report(self):
        count = self.count
        return {
            "velocity_axes": ["vx_m_per_s", "vy_m_per_s", "yaw_rate_rad_per_s"],
            "velocity_frame": "robot body; clean critic local_linvel xy and gyro z",
            "initial_command_env0": self.initial_command,
            "mean_command": (self.command_sum / count).tolist() if count else None,
            "mean_measured_velocity": (self.measured_sum / count).tolist() if count else None,
            "tracking_rmse_by_axis": (self.error_squared_sum / count).sqrt().tolist()
            if count
            else None,
            "mean_absolute_yaw_rate_rad_per_s": self.absolute_yaw_sum / count if count else None,
            "weighted_reward_terms_before_dt": {
                name: total / self.reward_steps for name, total in self.reward_sum.items()
            }
            if self.reward_steps
            else {},
            "reward_terms_note": "weighted terms BEFORE control-dt scaling; not per-step rewards",
        }
