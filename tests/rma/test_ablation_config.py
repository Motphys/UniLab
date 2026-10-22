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
from unilab_rma.evaluation_diagnostics import FORWARD_TASK, NO_POSE_TASK, RMA_TASKS


class AblationConfigTests(unittest.TestCase):
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
