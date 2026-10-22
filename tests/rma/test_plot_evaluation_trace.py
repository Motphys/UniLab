import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from plot_evaluation_trace import plot_trace


class PlotEvaluationTraceTests(unittest.TestCase):
    def test_generates_command_action_and_reward_figures(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "eval_sample"
            trace.mkdir()
            (trace / "run_config.json").write_text(
                json.dumps(
                    {
                        "control_period_s": 0.02,
                        "control_contract": {
                            "action_target_joint_order": [f"joint_{i}" for i in range(12)]
                        },
                    }
                )
            )
            fields = ["step", "before_vx_m_s", "command_vx_m_s", "reward_rate/track"]
            fields += [f"action_{i}" for i in range(12)]
            with (trace / "transitions.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for step in range(2):
                    writer.writerow(
                        {
                            "step": step,
                            "before_vx_m_s": 0.1 * step,
                            "command_vx_m_s": 0.5,
                            "reward_rate/track": 1.0,
                            **{f"action_{i}": 0.01 * i for i in range(12)},
                        }
                    )
            output = plot_trace(trace)
            self.assertEqual(len(output), 3)
            self.assertTrue(all(Path(path).stat().st_size > 1000 for path in output))


if __name__ == "__main__":
    unittest.main()
