"""Rendering isolation/cleanup regressions; synthetic states, not Go2 benchmarks."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_rma
from unilab_rma import ActorCriticRMA, AdaptationModule, RMAConfig

try:
    import mujoco
except ImportError:
    mujoco = None


class SyntheticEnv:
    def __init__(self):
        self.actions = []

    def observations(self):
        return torch.randn(1, 49), torch.randn(1, 52), torch.ones(1, 2)

    def reset(self):
        return self.observations()

    def step(self, action):
        self.actions.append(action.clone())
        obs = self.observations()
        done = torch.zeros(1, dtype=torch.bool)
        return obs, -action.square().mean(-1), done, done, obs


class PlaybackEnv:
    step_dt = 0.02
    play_capabilities = SimpleNamespace(supports_physics_state_playback=True)

    def __init__(self):
        self.closed = False
        if mujoco is not None:
            self.model = mujoco.MjModel.from_xml_string(
                '<mujoco><worldbody><body name="base"><joint type="hinge"/>'
                '<geom type="sphere" size="0.1" mass="1"/></body></worldbody></mujoco>'
            )

    def close(self):
        self.closed = True

    def get_playback_model(self, env_index=0):
        return self.model

    def get_physics_state_layout(self):
        return SimpleNamespace(
            nq=1,
            nv=1,
            nmocap=0,
            split_state=lambda state: SimpleNamespace(
                time=state[0], qpos=state[1:2], qvel=state[2:3]
            ),
        )

    def get_physics_state_snapshot(self):
        import numpy as np

        return np.array([[0.0, 0.2, 0.1]])


class FakeViewer:
    def __init__(self, open_checks=None):
        self.cam = SimpleNamespace()
        self.open_checks = open_checks
        self.closed = False
        self.frames = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def lock(self):
        return contextlib.nullcontext()

    def is_running(self):
        if self.open_checks is None:
            return True
        result = self.open_checks > 0
        self.open_checks -= 1
        return result

    def sync(self):
        self.frames += 1


class EvaluationRenderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.directory = tempfile.TemporaryDirectory()
        cls.checkpoint = Path(cls.directory.name) / "student.pt"
        cfg = RMAConfig()
        teacher, adaptation = ActorCriticRMA(cfg), AdaptationModule(cfg)
        torch.save(
            {
                "format": "unilab-rma-v1",
                "phase": "student",
                "config": teacher.metadata(),
                "teacher": teacher.state_dict(),
                "adaptation": adaptation.state_dict(),
            },
            cls.checkpoint,
        )

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def evaluate(self, raw, env, *extra):
        output = io.StringIO()
        argv = [
            "evaluate_rma.py",
            "--checkpoint",
            str(self.checkpoint),
            "--num-envs",
            "1",
            "--steps",
            "3",
            *extra,
        ]
        with (
            patch.object(sys, "argv", argv),
            patch.object(evaluate_rma, "build_environment", return_value=(raw, env)),
            contextlib.redirect_stdout(output),
        ):
            evaluate_rma.main()
        return json.loads(output.getvalue())

    def test_default_does_not_import_rendering_and_keeps_transition_count(self):
        raw, env = PlaybackEnv(), SyntheticEnv()
        with patch.dict(
            sys.modules,
            {
                "unilab_rma.rendering": None,
                "unilab_rma.tensorrt_runtime": None,
                "unilab_rma.evaluation_trace": None,
            },
        ):
            metrics = self.evaluate(raw, env)
        self.assertEqual(metrics["total_transitions"], 3)
        self.assertTrue(raw.closed)

    def test_diagnostics_preserve_default_metrics_and_actions(self):
        raw, env = PlaybackEnv(), SyntheticEnv()
        expected = self.evaluate(raw, env)
        raw2, env2 = PlaybackEnv(), SyntheticEnv()
        raw2.reward_manager = SimpleNamespace(
            step_reward_names=("tracking_lin_vel",), step_reward_means=torch.tensor([0.5])
        )
        actual = self.evaluate(raw2, env2, "--diagnostics")
        diagnostics = actual.pop("diagnostics")
        self.assertEqual(actual, expected)
        self.assertAlmostEqual(
            diagnostics["weighted_reward_terms_before_dt"]["tracking_lin_vel"], 0.5
        )
        for left, right in zip(env.actions, env2.actions):
            self.assertTrue(torch.equal(left, right))

    def test_fixed_command_and_task_are_forwarded_to_environment_builder(self):
        raw, env = PlaybackEnv(), SyntheticEnv()
        argv = [
            "evaluate_rma.py",
            "--checkpoint",
            str(self.checkpoint),
            "--num-envs",
            "1",
            "--steps",
            "3",
            "--command",
            "0.5",
            "0",
            "0",
            "--task",
            "go2_joystick_flat/rma_forward",
        ]
        with (
            patch.object(sys, "argv", argv),
            patch.object(evaluate_rma, "build_environment", return_value=(raw, env)) as build,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            evaluate_rma.main()
        self.assertEqual(
            build.call_args.kwargs,
            {"task": "go2_joystick_flat/rma_forward", "command": [0.5, 0.0, 0.0]},
        )

    def test_trace_preserves_actions_and_metrics_and_writes_separate_json(self):
        raw, env = PlaybackEnv(), SyntheticEnv()
        expected = self.evaluate(raw, env)
        raw2, env2 = PlaybackEnv(), SyntheticEnv()
        names = [f"joint_{i}" for i in range(12)]
        term = SimpleNamespace(
            cfg=SimpleNamespace(entity_name="robot", clip=None),
            target_names=names,
            target_ids=list(range(12)),
            scale=0.25,
            offset=torch.zeros(1, 12),
        )
        raw2.action_manager = SimpleNamespace(get_term=lambda name: term)
        raw2.scene = {"robot": SimpleNamespace(joint_names=names)}
        raw2.reward_manager = SimpleNamespace(
            step_reward_names=("tracking_lin_vel",),
            get_active_iterable_terms=lambda index: [("tracking_lin_vel", [0.5])],
        )
        with tempfile.TemporaryDirectory() as directory:
            actual = self.evaluate(raw2, env2, "--trace-dir", directory)
            trace = actual.pop("trace")
            self.assertEqual(actual, expected)
            self.assertEqual(trace["full"]["transitions"], 3)
            saved = json.loads((Path(trace["directory"]) / "summary.json").read_text())
            self.assertEqual(saved["total_transitions"], 3)
            self.assertTrue(raw2.closed)
            for left, right in zip(env.actions, env2.actions, strict=True):
                self.assertTrue(torch.equal(left, right))

    def test_reset_failure_still_closes_environment(self):
        raw, env = PlaybackEnv(), SyntheticEnv()
        with patch.object(env, "reset", side_effect=RuntimeError("reset failed")):
            with self.assertRaisesRegex(RuntimeError, "reset failed"):
                self.evaluate(raw, env, "--render")
        self.assertTrue(raw.closed)

    def test_tensorrt_actions_drive_environment_and_runtime_is_closed(self):
        raw, env = PlaybackEnv(), SyntheticEnv()

        class Runtime:
            closed = False

            def infer(self, obs, history):
                self.history_matches_current = torch.equal(history[:, -1, :], obs)
                return torch.full((1, 12), 0.125)

            def close(self):
                self.closed = True

        runtime = Runtime()
        module = SimpleNamespace(TensorRTWorker=lambda *args: runtime)
        with patch.dict(sys.modules, {"unilab_rma.tensorrt_runtime": module}):
            metrics = self.evaluate(raw, env, "--backend", "tensorrt", "--engine", "unused.engine")
        self.assertTrue(runtime.closed and raw.closed and runtime.history_matches_current)
        self.assertEqual(metrics["inference_backend"], "tensorrt_cpp")
        self.assertEqual(metrics["total_transitions"], 3)
        self.assertTrue(
            all(torch.equal(action, torch.full((1, 12), 0.125)) for action in env.actions)
        )

    def test_tensorrt_failure_closes_runtime_and_environment(self):
        raw, env = PlaybackEnv(), SyntheticEnv()
        closed = []

        def fail(*args):
            raise RuntimeError("worker failed")

        runtime = SimpleNamespace(infer=fail, close=lambda: closed.append(True))
        module = SimpleNamespace(TensorRTWorker=lambda *args: runtime)
        with patch.dict(sys.modules, {"unilab_rma.tensorrt_runtime": module}):
            with self.assertRaisesRegex(RuntimeError, "worker failed"):
                self.evaluate(raw, env, "--backend", "tensorrt", "--engine", "unused.engine")
        self.assertEqual(closed, [True])
        self.assertTrue(raw.closed)

    @unittest.skipIf(mujoco is None, "MuJoCo is required for playback contract tests")
    def test_render_keeps_actions_and_metrics_and_uses_separate_model(self):
        import mujoco.viewer

        raw, env = PlaybackEnv(), SyntheticEnv()
        expected = self.evaluate(raw, env)
        raw2, env2, viewer = PlaybackEnv(), SyntheticEnv(), FakeViewer()

        def launch(model, data):
            self.assertIsNot(model, raw2.model)
            model.opt.gravity[:] = 0  # A viewer edit must not change physics.
            return viewer

        with (
            patch("glfw.init", return_value=True),
            patch("glfw.terminate"),
            patch("mujoco.viewer.launch_passive", side_effect=launch),
            patch("unilab_rma.rendering.time.perf_counter", return_value=0),
            patch("unilab_rma.rendering.time.sleep") as sleep,
        ):
            actual = self.evaluate(raw2, env2, "--render", "--render-speed", "0.5")
        self.assertEqual(actual, expected)
        for a, b in zip(env.actions, env2.actions):
            self.assertTrue(torch.equal(a, b))
        self.assertNotEqual(float(raw2.model.opt.gravity[2]), 0)
        self.assertEqual(viewer.frames, 4)
        self.assertTrue(viewer.closed and raw2.closed)
        self.assertEqual(sleep.call_count, 3)
        self.assertAlmostEqual(sleep.call_args.args[0], 0.04)

    @unittest.skipIf(mujoco is None, "MuJoCo is required for playback contract tests")
    def test_window_close_reports_actual_transition_count(self):
        import mujoco.viewer

        raw, env, viewer = PlaybackEnv(), SyntheticEnv(), FakeViewer(open_checks=1)
        with (
            patch("glfw.init", return_value=True),
            patch("glfw.terminate"),
            patch("mujoco.viewer.launch_passive", return_value=viewer),
        ):
            metrics = self.evaluate(raw, env, "--render")
        self.assertEqual(metrics["total_transitions"], 1)
        self.assertTrue(viewer.closed and raw.closed)

    @unittest.skipIf(mujoco is None, "MuJoCo is required for playback contract tests")
    def test_inference_failure_closes_viewer(self):
        import mujoco.viewer

        raw, env, viewer = PlaybackEnv(), SyntheticEnv(), FakeViewer()
        with (
            patch("glfw.init", return_value=True),
            patch("glfw.terminate"),
            patch("mujoco.viewer.launch_passive", return_value=viewer),
            patch.object(env, "step", side_effect=RuntimeError("step failed")),
        ):
            with self.assertRaisesRegex(RuntimeError, "step failed"):
                self.evaluate(raw, env, "--render")
        self.assertTrue(viewer.closed and raw.closed)

    @unittest.skipIf(mujoco is None, "MuJoCo is required for playback contract tests")
    def test_unavailable_display_fails_cleanly(self):
        raw, env = PlaybackEnv(), SyntheticEnv()
        with patch("glfw.init", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "no usable GLFW display"):
                self.evaluate(raw, env, "--render")
        self.assertTrue(raw.closed)


if __name__ == "__main__":
    unittest.main()
