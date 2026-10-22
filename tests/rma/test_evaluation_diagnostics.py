"""Command/reset contract and diagnostic arithmetic, independent of assets."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rsl_rl"))
from unilab_rma.evaluation_diagnostics import (
    DEFAULT_TASK,
    FORWARD_TASK,
    EvaluationDiagnostics,
    command_overrides,
)


class EvaluationDiagnosticTests(unittest.TestCase):
    def test_fixed_command_is_a_config_override_not_a_one_time_tensor_patch(self):
        overrides = command_overrides((0.5, 0, 0))
        self.assertIn("env.commands.twist.ranges.lin_vel_x=[0.5,0.5]", overrides)
        self.assertIn("env.commands.twist.ranges.ang_vel_z=[0.0,0.0]", overrides)
        self.assertIn("env.commands.twist.rel_standing_envs=0.0", overrides)
        with self.assertRaises(ValueError):
            command_overrides((float("nan"), 0, 0))
        with self.assertRaises(ValueError):
            command_overrides((0, 0))

    def test_clean_body_velocity_and_reward_units(self):
        diagnostics = EvaluationDiagnostics()
        critic = torch.zeros(2, 52)
        critic[:, 42] = 0.5
        critic[:, 49] = 0.1
        critic[:, 2] = 0.2
        original = critic.clone()
        diagnostics.record_state(critic)
        manager = SimpleNamespace(
            step_reward_names=("tracking_lin_vel", "base_height"),
            step_reward_means=torch.tensor([0.4, -0.1]),
        )
        diagnostics.record_rewards(manager)
        manager.step_reward_means = torch.tensor([0.6, -0.3])
        diagnostics.record_rewards(manager)
        report = diagnostics.report()
        self.assertEqual(report["mean_command"], [0.5, 0, 0])
        self.assertAlmostEqual(report["mean_measured_velocity"][0], 0.1, places=6)
        self.assertAlmostEqual(report["mean_measured_velocity"][2], 0.2, places=6)
        self.assertAlmostEqual(report["tracking_rmse_by_axis"][0], 0.4, places=6)
        self.assertAlmostEqual(report["tracking_rmse_by_axis"][2], 0.2, places=6)
        terms = report["weighted_reward_terms_before_dt"]
        self.assertAlmostEqual(terms["tracking_lin_vel"], 0.5, places=6)
        self.assertAlmostEqual(terms["base_height"], -0.2, places=6)
        self.assertTrue(torch.equal(critic, original))

    def test_empty_report_and_invalid_contract(self):
        diagnostics = EvaluationDiagnostics()
        self.assertIsNone(diagnostics.report()["mean_command"])
        with self.assertRaises(ValueError):
            diagnostics.record_state(torch.zeros(1, 49))

    def test_forward_task_keeps_policy_contract_and_changes_only_experiment_drivers(self):
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf

        configs = []
        for task in (DEFAULT_TASK, FORWARD_TASK):
            with initialize_config_dir(
                config_dir=str(ROOT / "src/unilab/conf/ppo"), version_base=None
            ):
                configs.append(compose(config_name="config", overrides=[f"task={task}"]))
        original, forward = configs
        self.assertEqual(
            OmegaConf.to_container(original.env.observations),
            OmegaConf.to_container(forward.env.observations),
        )
        self.assertEqual(
            OmegaConf.to_container(original.env.actions),
            OmegaConf.to_container(forward.env.actions),
        )
        self.assertEqual(
            OmegaConf.to_container(original.env.events), OmegaConf.to_container(forward.env.events)
        )
        self.assertEqual(list(forward.env.commands.twist.ranges.lin_vel_x), [0.3, 0.6])
        self.assertEqual(list(forward.env.commands.twist.ranges.ang_vel_z), [0.0, 0.0])
        self.assertEqual(original.reward.tracking_lin_vel.weight, 1.0)
        self.assertEqual(forward.reward.tracking_lin_vel.weight, 2.0)
        self.assertEqual(original.reward.swing_feet_z.weight, 4.0)
        self.assertEqual(forward.reward.swing_feet_z.weight, 0.25)


if __name__ == "__main__":
    unittest.main()
