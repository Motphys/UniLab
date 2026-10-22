import contextlib
import csv
import io
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rsl_rl"))
from unilab_rma import ActorCriticRMA, AdaptationModule, RMAConfig
from unilab_rma.runner import train_teacher
from unilab_rma.training_log import EpisodeLog, TrainingLog


class LoggingTests(unittest.TestCase):
    def test_csv_events_and_unique_runs(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "teacher.pt"
            with TrainingLog(output, "teacher", {"seed": 42}) as log:
                directory = log.directory
                log.write(
                    1,
                    48,
                    loss=0.5,
                    mean_step_reward=0.1,
                    completed_episodes=0,
                    mean_episode_return=None,
                    mean_episode_length_steps=None,
                )
                self.assertEqual(
                    len(list(csv.DictReader((directory / "metrics.csv").read_text().splitlines()))),
                    1,
                )
                log.write(
                    2,
                    96,
                    loss=0.4,
                    mean_step_reward=0.2,
                    completed_episodes=2,
                    mean_episode_return=1.5,
                    mean_episode_length_steps=7,
                )
            rows = list(csv.DictReader((directory / "metrics.csv").read_text().splitlines()))
            self.assertEqual(rows[0]["mean_episode_return"], "")
            self.assertEqual(rows[1]["total_transitions"], "96")
            events = EventAccumulator(str(directory / "tensorboard")).Reload()
            self.assertEqual([event.step for event in events.Scalars("teacher/loss")], [1, 2])
            self.assertEqual(len(events.Scalars("teacher/mean_episode_return")), 1)
            with TrainingLog(output, "teacher", {}) as second:
                self.assertNotEqual(second.directory, directory)

    def test_episode_accumulators_cross_rollouts_and_reset_individually(self):
        episodes = EpisodeLog(2)
        first = episodes.consume(
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[False, True], [False, False]])
        )
        self.assertEqual(first["mean_episode_return"], 2.0)
        second = episodes.consume(torch.tensor([[5.0, 6.0]]), torch.tensor([[True, True]]))
        self.assertEqual(second["completed_episodes"], 2)
        self.assertEqual(second["mean_episode_return"], 9.5)
        self.assertEqual(second["mean_episode_length_steps"], 2.5)

    def test_logging_does_not_change_policy_updates_or_rng(self):
        class Env:
            num_envs = 2

            def reset(self):
                return self.obs()

            def obs(self):
                return torch.randn(2, 49), torch.randn(2, 52), torch.ones(2, 2)

            def step(self, action):
                done = torch.zeros(2, dtype=torch.bool)
                return self.obs(), -action.square().mean(-1), done, done, self.obs()

        def run(output, logging):
            torch.manual_seed(123)
            teacher, adaptation = ActorCriticRMA(RMAConfig()), AdaptationModule(RMAConfig())
            scope = TrainingLog(output, "teacher", {}) if logging else contextlib.nullcontext(None)
            with contextlib.redirect_stdout(io.StringIO()), scope as logger:
                train_teacher(Env(), teacher, adaptation, 2, output, horizon=3, logger=logger)
            return teacher.state_dict(), torch.get_rng_state()

        with tempfile.TemporaryDirectory() as folder:
            expected, rng1 = run(Path(folder) / "off.pt", False)
            actual, rng2 = run(Path(folder) / "on.pt", True)
        self.assertTrue(torch.equal(rng1, rng2))
        self.assertTrue(all(torch.equal(expected[name], actual[name]) for name in expected))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
