"""Trace alignment, reset boundaries and disabled-path behavior."""

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rsl_rl"))
from unilab_rma.evaluation_trace import EvaluationTrace


def trace_raw(num_envs=2):
    joint_names = [
        f"{leg}_{joint}_joint"
        for leg in ("FL", "FR", "RL", "RR")
        for joint in ("hip", "thigh", "calf")
    ]
    ids = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
    action = SimpleNamespace(
        cfg=SimpleNamespace(entity_name="robot", clip=None),
        target_names=[joint_names[i] for i in ids],
        target_ids=ids,
        scale=0.25,
        offset=torch.zeros(num_envs, 12),
    )
    rewards = SimpleNamespace(
        step_reward_names=("tracking_lin_vel", "penalty"),
        get_active_iterable_terms=lambda i: [
            ("tracking_lin_vel", [float(i + 1)]),
            ("penalty", [-0.2]),
        ],
    )
    return SimpleNamespace(
        step_dt=0.02,
        reward_manager=rewards,
        action_manager=SimpleNamespace(get_term=lambda name: action),
        scene={"robot": SimpleNamespace(joint_names=joint_names)},
    )


class EvaluationTraceTests(unittest.TestCase):
    def test_terminal_state_reset_ids_and_units_are_not_mixed(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "weights.pt"
            checkpoint.write_bytes(b"test checkpoint")
            raw = trace_raw()
            trace = EvaluationTrace(directory, raw, {"num_envs": 2}, checkpoint, warmup_steps=1)
            before = torch.zeros(2, 52)
            before[:, 42] = 0.5
            before[:, 49] = 0.1
            after = before.clone()
            after[:, 49] = 0.25
            action = torch.zeros(2, 12)
            action[:, 0] = 0.96
            original = before.clone(), after.clone(), action.clone()
            trace.record(
                before,
                action,
                after,
                torch.tensor([0.016, 0.036]),
                torch.tensor([True, False]),
                torch.zeros(2, dtype=torch.bool),
                raw.reward_manager,
            )
            trace.record(
                before,
                action,
                after,
                torch.tensor([0.016, 0.036]),
                torch.zeros(2, dtype=torch.bool),
                torch.zeros(2, dtype=torch.bool),
                raw.reward_manager,
            )
            report = trace.finish({"total_transitions": 4})
            trace.close()
            with (trace.directory / "transitions.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            metadata = json.loads((trace.directory / "run_config.json").read_text())
            self.assertEqual(len(rows), 4)
            self.assertAlmostEqual(float(rows[0]["before_vx_m_s"]), 0.1)
            self.assertAlmostEqual(float(rows[0]["after_vx_m_s"]), 0.25)
            self.assertEqual(rows[2]["episode_id"], "1")
            self.assertEqual(rows[2]["episode_step"], "1")
            self.assertEqual(rows[3]["episode_id"], "0")
            self.assertEqual(rows[3]["episode_step"], "2")
            self.assertAlmostEqual(float(rows[1]["reward_rate/tracking_lin_vel"]), 2)
            self.assertEqual(report["trace"]["full"]["transitions"], 4)
            self.assertEqual(report["trace"]["after_startup_window"]["transitions"], 1)
            self.assertEqual(report["trace"]["normalized_action_near_bound_fraction"][0], 1)
            self.assertEqual(
                metadata["control_contract"]["action_target_joint_order"][0], "FR_hip_joint"
            )
            self.assertEqual(
                metadata["control_contract"]["joint_observation_order"][0], "FL_hip_joint"
            )
            for left, right in zip(original, (before, after, action), strict=True):
                self.assertTrue(torch.equal(left, right))

    def test_repeated_invocations_do_not_overwrite_and_empty_window_is_null(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "weights.pt"
            checkpoint.write_bytes(b"checkpoint")
            first = EvaluationTrace(directory, trace_raw(), {"num_envs": 2}, checkpoint)
            second = EvaluationTrace(directory, trace_raw(), {"num_envs": 2}, checkpoint)
            try:
                self.assertNotEqual(first.directory, second.directory)
                self.assertIsNone(first.report()["after_startup_window"]["tracking_rmse_by_axis"])
                self.assertIsNone(first.report()["normalized_action_min"])
            finally:
                first.close()
                second.close()
