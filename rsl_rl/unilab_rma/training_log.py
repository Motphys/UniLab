"""Per-iteration CSV and TensorBoard records, independent of policy updates."""

import csv
import json
import tempfile
import time
from datetime import datetime
from pathlib import Path


class TrainingLog:
    FIELDS = (
        "iteration",
        "total_transitions",
        "wall_seconds",
        "loss",
        "latent_mse",
        "mean_step_reward",
        "completed_episodes",
        "mean_episode_return",
        "mean_episode_length_steps",
    )

    def __init__(self, output, phase, metadata, log_dir=None):
        output = Path(output).resolve()
        parent = Path(log_dir).resolve() if log_dir else output.parent / (output.stem + "_logs")
        parent.mkdir(parents=True, exist_ok=True)
        self.directory = Path(
            tempfile.mkdtemp(prefix=datetime.now().strftime("%Y%m%d_%H%M%S_"), dir=parent)
        )
        self._writer = None
        self._stream = None
        self._started = time.monotonic()
        self._phase = phase
        try:
            from torch.utils.tensorboard import SummaryWriter

            self._writer = SummaryWriter(log_dir=str(self.directory / "tensorboard"), flush_secs=10)
            self._stream = (self.directory / "metrics.csv").open("w", newline="", encoding="utf-8")
            self._csv = csv.DictWriter(self._stream, fieldnames=self.FIELDS)
            self._csv.writeheader()
            self._stream.flush()
            record = dict(
                metadata,
                output=str(output),
                phase=phase,
                log_directory=str(self.directory),
                iteration_numbering="one-based completed updates",
            )
            (self.directory / "run_config.json").write_text(
                json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8"
            )
        except BaseException:
            self.close()
            raise

    def write(self, iteration, total_transitions, **metrics):
        row = dict(
            iteration=iteration,
            total_transitions=total_transitions,
            wall_seconds=time.monotonic() - self._started,
            **metrics,
        )
        self._csv.writerow(row)
        self._stream.flush()  # Keep completed iterations available after interruption.
        for name, value in row.items():
            if value is not None and name != "iteration":
                self._writer.add_scalar(self._phase + "/" + name, value, iteration)

    def close(self):
        try:
            if self._writer is not None:
                self._writer.close()
                self._writer = None
        finally:
            if self._stream is not None:
                self._stream.close()
                self._stream = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class EpisodeLog:
    """CPU-only bookkeeping; consumes detached rollout copies once per update."""

    def __init__(self, count):
        self.returns = [0.0] * count
        self.lengths = [0] * count

    def consume(self, rewards, done):
        completed_returns, completed_lengths = [], []
        for reward_row, done_row in zip(
            rewards.detach().cpu().tolist(), done.detach().cpu().tolist()
        ):
            for index, (reward, finished) in enumerate(zip(reward_row, done_row)):
                self.returns[index] += reward
                self.lengths[index] += 1
                if finished:
                    completed_returns.append(self.returns[index])
                    completed_lengths.append(self.lengths[index])
                    self.returns[index] = 0.0
                    self.lengths[index] = 0
        return dict(
            completed_episodes=len(completed_returns),
            mean_episode_return=(
                sum(completed_returns) / len(completed_returns) if completed_returns else None
            ),
            mean_episode_length_steps=(
                sum(completed_lengths) / len(completed_lengths) if completed_lengths else None
            ),
        )
