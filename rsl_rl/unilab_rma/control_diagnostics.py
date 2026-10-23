"""Evaluation-only reads of public position-control signals, for one world."""

import csv
import json
from pathlib import Path

import numpy as np
import torch


def array(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.array(value, dtype=np.float64, copy=True)


class ControlDiagnostics:
    def __init__(self, raw, directory):
        if raw.num_envs != 1:
            raise ValueError("Control diagnostics require exactly one environment")
        self.raw = raw
        self.directory = Path(directory)
        self.term = raw.action_manager.get_term("joint_pos")
        self.robot = raw.scene[self.term.cfg.entity_name]
        self.names = list(self.term.target_names)
        joint_names = list(self.robot.joint_names)
        self.joint_ids = [joint_names.index(name) for name in self.names]
        actuator_names = list(raw.backend.get_actuator_names())
        actuator_joints = list(raw.backend.get_actuator_joint_names())
        self.actuator_ids = [actuator_joints.index(name) for name in self.names]
        if list(self.robot.actuator_names) != actuator_names:
            raise ValueError("Entity and backend actuator order differ; cannot interpret buffer")
        if len(self.names) != 12 or len(set(self.actuator_ids)) != 12:
            raise ValueError("Requires 12 distinct single-joint position actuators")
        self.rows = []
        self.step = self.episode = self.age = 0
        self.metadata = {
            "joint_names": self.names,
            "actuator_names": actuator_names,
            "action_to_actuator_indices": self.actuator_ids,
            "control_period_s": float(raw.step_dt),
            "alignment": "q_before/dq_before precede action; target is the authoritative control buffer after step; q_after precedes reset. No cross-reset rows.",
            "mapping_residual": "written target - (processed_action - encoder_bias), in joint order",
            "pd_demand_estimate": "kp*(written_target-q_before)-kd*dq_before; an algebraic estimate at the control boundary, NOT measured torque or a physics-substep sample.",
            "torque_measurement": "unavailable through the current public backend API; no torque saturation claim",
            "gains_scope": "public get_actuator_gains, refreshed each step; one environment only",
        }
        (self.directory / "control_metadata.json").write_text(
            json.dumps(self.metadata, indent=2), encoding="utf-8"
        )

    def before(self):
        return (
            array(self.robot.data.joint_pos)[0, self.joint_ids],
            array(self.robot.data.joint_vel)[0, self.joint_ids],
        )

    def record(self, before, action, terminated, truncated):
        q, dq = before
        after = array(self.robot.data.joint_pos)[0, self.joint_ids]
        written = array(self.robot.data.control_buffer)[0, self.actuator_ids]
        processed = array(self.term.processed_action)[0]
        bias = array(self.robot.data.encoder_bias)[0, self.joint_ids]
        kp, kd = self.raw.backend.get_actuator_gains()
        kp, kd = array(kp).reshape(-1)[self.actuator_ids], array(kd).reshape(-1)[self.actuator_ids]
        a = array(action)[0]
        self.age += 1
        for i, name in enumerate(self.names):
            self.rows.append(
                dict(
                    step=self.step,
                    episode=self.episode,
                    episode_step=self.age,
                    time_s=self.step * float(self.raw.step_dt),
                    joint=name,
                    action=a[i],
                    q_before_rad=q[i],
                    dq_before_rad_s=dq[i],
                    q_after_rad=after[i],
                    processed_target_rad=processed[i],
                    encoder_bias_rad=bias[i],
                    written_target_rad=written[i],
                    mapping_residual_rad=written[i] - (processed[i] - bias[i]),
                    kp=kp[i],
                    kd=kd[i],
                    pd_demand_estimate_nm=kp[i] * (written[i] - q[i]) - kd[i] * dq[i],
                    terminated=int(terminated),
                    truncated=int(truncated),
                )
            )
        self.step += 1
        if terminated or truncated:
            self.episode += 1
            self.age = 0

    def finish(self):
        if not self.rows:
            raise ValueError("No control samples")
        with (self.directory / "control_signals.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        report = {
            "max_abs_mapping_residual_rad": max(abs(r["mapping_residual_rad"]) for r in self.rows),
            "torque_measurement": self.metadata["torque_measurement"],
            "samples": len(self.rows),
            "joints_after_first_100_steps_per_episode": {},
        }
        for name in self.names:
            rows = [r for r in self.rows if r["joint"] == name and r["episode_step"] > 100]
            report["joints_after_first_100_steps_per_episode"][name] = {
                "samples": len(rows),
                "position_error_rmse_rad": float(
                    np.sqrt(
                        np.mean([(r["written_target_rad"] - r["q_after_rad"]) ** 2 for r in rows])
                    )
                )
                if rows
                else None,
            }
        (self.directory / "control_summary.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        self.plot()
        return report

    def plot(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(4, 3, figsize=(14, 11), constrained_layout=True)
        for index, name in enumerate(self.names):
            ax = axes.flat[index]
            for ep in sorted({r["episode"] for r in self.rows}):
                rows = [r for r in self.rows if r["joint"] == name and r["episode"] == ep]
                t = [(r["step"] + 1) * float(self.raw.step_dt) for r in rows]
                ax.plot(
                    t,
                    [r["written_target_rad"] for r in rows],
                    color="#1761A0",
                    label="Written target" if ep == 0 else None,
                )
                ax.plot(
                    t,
                    [r["q_after_rad"] for r in rows],
                    color="#D36C22",
                    label="Measured after step" if ep == 0 else None,
                )
            ax.set_title(name)
            ax.set_ylabel("Angle (rad)")
            ax.grid(alpha=0.25)
        axes[0, 0].legend(fontsize=8)
        for ax in axes[-1]:
            ax.set_xlabel("Evaluation time (s)")
        fig.suptitle("Authoritative actuator target vs measured angle; episodes plotted separately")
        fig.savefig(self.directory / "control_tracking.png", dpi=140)
        plt.close(fig)
