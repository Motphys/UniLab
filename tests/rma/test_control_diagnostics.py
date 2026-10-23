"""Position-control telemetry must resolve storage order and reset boundaries."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rsl_rl"))
from unilab_rma.control_diagnostics import ControlDiagnostics


def fixture():
    names = [
        f"{leg}_{joint}_joint"
        for leg in ("FL", "FR", "RL", "RR")
        for joint in ("hip", "thigh", "calf")
    ]
    permutation = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8])
    target = torch.arange(12, dtype=torch.float32).reshape(1, 12) / 10
    bias = np.full((1, 12), 0.02)
    buffer = np.zeros((1, 12))
    buffer[:, permutation] = target.numpy() - bias
    actuators = [f"motor_{i}" for i in range(12)]
    data = SimpleNamespace(
        joint_pos=np.zeros((1, 12)),
        joint_vel=np.ones((1, 12)),
        encoder_bias=bias,
        control_buffer=buffer,
    )
    term = SimpleNamespace(
        cfg=SimpleNamespace(entity_name="robot"), target_names=names, processed_action=target
    )
    robot = SimpleNamespace(joint_names=names, actuator_names=actuators, data=data)
    backend = SimpleNamespace(
        get_actuator_names=lambda: actuators,
        get_actuator_joint_names=lambda: [names[i] for i in permutation],
        get_actuator_gains=lambda: (np.full(12, 35), np.full(12, 0.5)),
    )
    raw = SimpleNamespace(
        num_envs=1,
        step_dt=0.02,
        scene={"robot": robot},
        action_manager=SimpleNamespace(get_term=lambda _: term),
        backend=backend,
    )
    return raw, target, permutation


class ControlDiagnosticsTests(unittest.TestCase):
    def test_permuted_targets_bias_and_before_snapshot(self):
        raw, action, _ = fixture()
        with tempfile.TemporaryDirectory() as directory:
            recorder = ControlDiagnostics(raw, directory)
            before = recorder.before()
            raw.scene["robot"].data.joint_pos[:] = 0.2
            recorder.record(before, action, False, False)
            np.testing.assert_array_equal(before[0], np.zeros(12))
            self.assertLess(max(abs(row["mapping_residual_rad"]) for row in recorder.rows), 1e-7)
            self.assertAlmostEqual(recorder.rows[0]["pd_demand_estimate_nm"], -1.2)
            self.assertAlmostEqual(recorder.rows[0]["q_after_rad"], 0.2)
            torch.testing.assert_close(
                raw.action_manager.get_term("joint_pos").processed_action, action
            )

    def test_direct_copy_bug_is_detected(self):
        raw, action, _ = fixture()
        raw.scene["robot"].data.control_buffer[:] = action.numpy() - 0.02
        with tempfile.TemporaryDirectory() as directory:
            recorder = ControlDiagnostics(raw, directory)
            recorder.record(recorder.before(), action, False, False)
            self.assertGreater(max(abs(row["mapping_residual_rad"]) for row in recorder.rows), 0.2)

    def test_reset_boundary_has_new_episode_and_copied_terminal_state(self):
        raw, action, _ = fixture()
        with tempfile.TemporaryDirectory() as directory:
            recorder = ControlDiagnostics(raw, directory)
            recorder.record(recorder.before(), action, True, False)
            raw.scene["robot"].data.joint_pos[:] = 9
            recorder.record(recorder.before(), action, False, False)
            self.assertEqual(recorder.rows[0]["q_after_rad"], 0)
            self.assertEqual(recorder.rows[12]["episode"], 1)
            self.assertEqual(recorder.rows[12]["episode_step"], 1)
            self.assertEqual(recorder.rows[12]["q_before_rad"], 9)

    def test_multiple_worlds_rejected(self):
        raw, _, _ = fixture()
        raw.num_envs = 2
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            ControlDiagnostics(raw, directory)


if __name__ == "__main__":
    unittest.main()
