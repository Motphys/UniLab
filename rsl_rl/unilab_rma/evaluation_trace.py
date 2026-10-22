"""Evaluation-only transition recording; no simulator or policy mutation."""

import csv
import hashlib
import json
import math
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import torch


def _json_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def control_contract(raw):
    """Read resolved public action metadata, never infer action order from obs."""
    term = raw.action_manager.get_term("joint_pos")
    robot = raw.scene[term.cfg.entity_name]
    return {
        "action_type": type(term).__name__,
        "joint_observation_order": list(robot.joint_names),
        "action_target_joint_order": list(term.target_names),
        "action_target_joint_indices": [int(index) for index in term.target_ids],
        "scale": _json_value(term.scale),
        "offset_rad": _json_value(term.offset),
        "clip_config": term.cfg.clip,
        "note": (
            "Resolved position-action affine mapping: target = offset + scale * action, "
            "before configured clipping and encoder-bias compensation. This is configuration "
            "evidence, not a measurement of actuator torque or proof of correct physical response."
        ),
    }


class EvaluationTrace:
    """One CSV row per environment transition, unique directory per invocation."""

    def __init__(self, parent, raw, arguments, checkpoint, *, warmup_steps=100):
        self.num_envs = int(arguments["num_envs"])
        self.dt = float(raw.step_dt)
        if not math.isfinite(self.dt) or self.dt <= 0 or warmup_steps < 0:
            raise ValueError("Trace requires positive control dt and nonnegative warmup steps")
        self.warmup_steps = warmup_steps
        self.control = control_contract(raw)
        if len(self.control["action_target_joint_order"]) != 12:
            raise ValueError("Trace requires the 12-action Go2 contract")
        self.names = tuple(raw.reward_manager.step_reward_names)
        parent = Path(parent).resolve()
        parent.mkdir(parents=True, exist_ok=True)
        self.directory = Path(tempfile.mkdtemp(prefix="eval_", dir=parent))
        self.episode_ids = [0] * self.num_envs
        self.episode_steps = [0] * self.num_envs
        self.rows = 0
        self.steps = 0
        self.action_min = torch.full((12,), torch.inf, dtype=torch.float64)
        self.action_max = -self.action_min.clone()
        self.action_squared_sum = torch.zeros(12, dtype=torch.float64)
        self.action_near_bound = torch.zeros(12, dtype=torch.float64)
        self.velocity_sum = torch.zeros(3, dtype=torch.float64)
        self.error_squared_sum = torch.zeros(3, dtype=torch.float64)
        self.window_velocity_sum = torch.zeros(3, dtype=torch.float64)
        self.window_error_squared_sum = torch.zeros(3, dtype=torch.float64)
        self.window_count = 0
        self.reward_sums = {name: 0.0 for name in self.names}
        self._stream = None
        fields = ["step", "env_id", "episode_id", "episode_step", "time_s", "next_time_s"]
        fields += [f"command_{axis}" for axis in ("vx_m_s", "vy_m_s", "wz_rad_s")]
        fields += [f"next_command_{axis}" for axis in ("vx_m_s", "vy_m_s", "wz_rad_s")]
        fields += [
            f"{phase}_{axis}"
            for phase in ("before", "after")
            for axis in ("vx_m_s", "vy_m_s", "wz_rad_s")
        ]
        fields += ["reward", "terminated", "truncated", "after_startup_window"]
        fields += [f"action_{i}" for i in range(12)]
        fields += [f"joint_pos_rel_{i}_rad" for i in range(12)]
        fields += [f"joint_vel_rel_{i}_rad_s" for i in range(12)]
        fields += [f"reward_rate/{name}" for name in self.names]
        root = Path(__file__).resolve().parents[2]
        try:
            revision = (
                subprocess.run(
                    ["git", "-C", str(root), "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    check=False,
                ).stdout.strip()
                or None
            )
            dirty = subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=False,
            )
            dirty = bool(dirty.stdout.strip()) if dirty.returncode == 0 else None
        except OSError:
            revision, dirty = None, None
        metadata = {
            "format": "unilab-rma-evaluation-trace-v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "arguments": arguments,
            "git_commit": revision,
            "git_dirty": dirty,
            "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
            "control_period_s": self.dt,
            "warmup_steps_per_episode": warmup_steps,
            "velocity_frame": "body; clean critic [49:51] and gyro z [2]",
            "transition_alignment": (
                "before = s_t; action = a_t; after = terminal s_(t+1), before any reset. "
                "Commands at both endpoints are retained; reward belongs to this transition. "
                "Full/window summaries use before velocity and command, matching --diagnostics."
            ),
            "reward_terms": (
                "Per-environment weighted rates from public get_active_iterable_terms(); "
                "before control-dt scaling. reward is the actual returned scalar. "
                "Do not compare rates directly with reward or across changed weights."
            ),
            "action_near_bound_definition": "abs(normalized policy action) >= 0.95; NOT torque saturation",
            "control_contract": self.control,
        }
        source_paths = (
            "scripts/evaluate_rma.py",
            "rsl_rl/unilab_rma/evaluation_trace.py",
            "rsl_rl/unilab_rma/env_adapter.py",
            "src/unilab/envs/mdp/actions/actions.py",
            "src/unilab/conf/ppo/task/go2_joystick_flat/base.yaml",
            "src/unilab/conf/ppo/task/go2_joystick_flat/mujoco.yaml",
            "src/unilab/conf/ppo/task/go2_joystick_flat/rma.yaml",
            "src/unilab/conf/ppo/task/go2_joystick_flat/rma_forward.yaml",
            "src/unilab/conf/ppo/task/go2_joystick_flat/rma_forward_no_pose.yaml",
        )
        metadata["runtime_source_sha256"] = {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in source_paths
            if (root / name).is_file()
        }
        config_dir = self.directory / "task_source_snapshot"
        config_dir.mkdir()
        for name in source_paths:
            if name.endswith(".yaml") and (root / name).is_file():
                (config_dir / Path(name).name).write_bytes((root / name).read_bytes())
        (self.directory / "run_config.json").write_text(
            json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8"
        )
        self._stream = (self.directory / "transitions.csv").open("w", newline="", encoding="utf-8")
        self._csv = csv.DictWriter(self._stream, fieldnames=fields)
        self._csv.writeheader()
        self._stream.flush()

    def record(self, before, action, terminal, reward, terminated, truncated, reward_manager):
        before = before.detach().to(device="cpu", dtype=torch.float64)
        after = terminal.detach().to(device="cpu", dtype=torch.float64)
        actions = action.detach().to(device="cpu", dtype=torch.float64)
        if before.shape != (self.num_envs, 52) or after.shape != before.shape:
            raise ValueError("Trace requires clean critic tensors of shape (num_envs, 52)")
        if actions.shape != (self.num_envs, 12):
            raise ValueError("Trace requires actions of shape (num_envs, 12)")
        if tuple(reward_manager.step_reward_names) != self.names:
            raise ValueError("Reward terms changed during trace")
        measured = torch.cat((before[:, 49:51], before[:, 2:3]), dim=1)
        next_measured = torch.cat((after[:, 49:51], after[:, 2:3]), dim=1)
        errors = (before[:, 42:45] - measured).square()
        rewards = reward.detach().cpu().tolist()
        terms = terminated.detach().cpu().tolist()
        truncs = truncated.detach().cpu().tolist()
        for index in range(self.num_envs):
            rates = dict(reward_manager.get_active_iterable_terms(index))
            if set(rates) != set(self.names) or any(len(value) != 1 for value in rates.values()):
                raise ValueError("Unexpected public reward-term shape")
            self.episode_steps[index] += 1
            age = self.episode_steps[index]
            in_window = age > self.warmup_steps
            row = dict(
                step=self.steps,
                env_id=index,
                episode_id=self.episode_ids[index],
                episode_step=age,
                time_s=(age - 1) * self.dt,
                next_time_s=age * self.dt,
                reward=rewards[index],
                terminated=int(terms[index]),
                truncated=int(truncs[index]),
                after_startup_window=int(in_window),
            )
            for prefix, values in (
                ("command", before[index, 42:45]),
                ("next_command", after[index, 42:45]),
                ("before", measured[index]),
                ("after", next_measured[index]),
            ):
                row.update(
                    zip(
                        (f"{prefix}_{axis}" for axis in ("vx_m_s", "vy_m_s", "wz_rad_s")),
                        values.tolist(),
                        strict=True,
                    )
                )
            row.update({f"action_{i}": value for i, value in enumerate(actions[index].tolist())})
            row.update(
                {
                    f"joint_pos_rel_{i}_rad": value
                    for i, value in enumerate(before[index, 6:18].tolist())
                }
            )
            row.update(
                {
                    f"joint_vel_rel_{i}_rad_s": value
                    for i, value in enumerate(before[index, 18:30].tolist())
                }
            )
            for name in self.names:
                row[f"reward_rate/{name}"] = rates[name][0]
                self.reward_sums[name] += rates[name][0]
            self._csv.writerow(row)
            if in_window:
                self.window_count += 1
                self.window_velocity_sum += measured[index]
                self.window_error_squared_sum += errors[index]
            if terms[index] or truncs[index]:
                self.episode_ids[index] += 1
                self.episode_steps[index] = 0
        self.action_min = torch.minimum(self.action_min, actions.amin(0))
        self.action_max = torch.maximum(self.action_max, actions.amax(0))
        self.action_squared_sum += actions.square().sum(0)
        self.action_near_bound += (actions.abs() >= 0.95).sum(0)
        self.velocity_sum += measured.sum(0)
        self.error_squared_sum += errors.sum(0)
        self.rows += self.num_envs
        self.steps += 1
        if self.steps % 50 == 0:
            self._stream.flush()

    def report(self):
        def velocity_report(count, total, squared):
            return {
                "transitions": count,
                "mean_measured_velocity": (total / count).tolist() if count else None,
                "tracking_rmse_by_axis": (squared / count).sqrt().tolist() if count else None,
            }

        return {
            "directory": str(self.directory),
            "full": velocity_report(self.rows, self.velocity_sum, self.error_squared_sum),
            "after_startup_window": velocity_report(
                self.window_count, self.window_velocity_sum, self.window_error_squared_sum
            ),
            "action_target_joint_order": self.control["action_target_joint_order"],
            "normalized_action_min": self.action_min.tolist() if self.rows else None,
            "normalized_action_max": self.action_max.tolist() if self.rows else None,
            "normalized_action_rms": (self.action_squared_sum / self.rows).sqrt().tolist()
            if self.rows
            else None,
            "normalized_action_near_bound_fraction": (self.action_near_bound / self.rows).tolist()
            if self.rows
            else None,
            "mean_weighted_reward_rates_before_dt": {
                name: value / self.rows for name, value in self.reward_sums.items()
            }
            if self.rows
            else {},
        }

    def finish(self, evaluation_report):
        result = dict(evaluation_report, trace=self.report())
        (self.directory / "summary.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        self._stream.flush()
        return result

    def close(self):
        if self._stream is not None:
            self._stream.close()
            self._stream = None
