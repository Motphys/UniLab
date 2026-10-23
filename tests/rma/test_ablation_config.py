"""Protect the single-variable experiment and existing checkpoints."""

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "rsl_rl"))
import train_rma
from inspect_rma_task import differences, resolved
from unilab_rma.evaluation_diagnostics import (
    FORWARD_TASK,
    NO_POSE_TASK,
    RMA_TASKS,
    TIGHT_TRACKING_TASK,
)


class AblationConfigTests(unittest.TestCase):
    def test_legacy_133_overlay_changes_only_std(self):
        import yaml

        path = ROOT / "experiments/rma/runtime_1_3_3/rma_forward_tight_tracking.yaml"
        overlay = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.assertEqual(overlay["defaults"], ["/task/go2_joystick_flat/rma_forward", "_self_"])
        self.assertEqual(overlay["reward"]["tracking_lin_vel"], {"params": {"std": 0.35}})

    def test_native_tight_tracking_changes_only_denominator(self):
        baseline, experiment = resolved(FORWARD_TASK), resolved(TIGHT_TRACKING_TASK)
        self.assertIn(TIGHT_TRACKING_TASK, RMA_TASKS)
        self.assertEqual(
            differences(baseline, experiment),
            [dict(path="reward.tracking_lin_vel.params.tracking_sigma", before=0.5, after=0.1225)],
        )
        self.assertNotIn("std", experiment["reward"]["tracking_lin_vel"]["params"])
        self.assertEqual(experiment["reward"]["similar_to_default"]["weight"], -0.1)

    def test_inspector_catches_extra_changes_and_is_read_only(self):
        before = {"reward": {"width": 0.5, "weight": 2}, "other": [1, 2]}
        after = {"reward": {"width": 0.35, "weight": 3}, "other": [1, 2]}
        changes = differences(before, after)
        self.assertEqual([row["path"] for row in changes], ["reward.weight", "reward.width"])
        self.assertEqual(before["reward"], {"width": 0.5, "weight": 2})

    def test_composed_experiment_changes_exactly_one_weight(self):
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf

        configs = []
        for task in (FORWARD_TASK, NO_POSE_TASK):
            with initialize_config_dir(
                config_dir=str(ROOT / "src/unilab/conf/ppo"), version_base=None
            ):
                config = compose(config_name="config", overrides=[f"task={task}"])
            configs.append(OmegaConf.to_container(config, resolve=True))
        baseline, experiment = configs
        self.assertIn(NO_POSE_TASK, RMA_TASKS)
        self.assertEqual(baseline["reward"]["similar_to_default"]["weight"], -0.1)
        self.assertEqual(experiment["reward"]["similar_to_default"]["weight"], 0.0)
        experiment["reward"]["similar_to_default"]["weight"] = -0.1
        self.assertEqual(experiment, baseline)

    def test_existing_output_is_rejected_before_model_or_environment_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing.pt"
            output.write_bytes(b"existing checkpoint must be preserved")
            argv = [
                "train_rma.py",
                "--phase",
                "teacher",
                "--task",
                NO_POSE_TASK,
                "--output",
                str(output),
            ]
            stderr = io.StringIO()
            with (
                patch.object(sys, "argv", argv),
                patch.object(train_rma, "ActorCriticRMA") as model,
                patch.object(train_rma, "build_environment") as build,
                contextlib.redirect_stderr(stderr),
                self.assertRaises(SystemExit) as error,
            ):
                train_rma.main()
            self.assertEqual(error.exception.code, 2)
            self.assertIn("Output already exists", stderr.getvalue())
            self.assertEqual(output.read_bytes(), b"existing checkpoint must be preserved")
            model.assert_not_called()
            build.assert_not_called()


if __name__ == "__main__":
    unittest.main()
