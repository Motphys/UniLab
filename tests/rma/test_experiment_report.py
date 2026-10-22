import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rsl_rl"))

from unilab_rma.experiment_report import load_evaluations, rolling_mean


class ExperimentReportTests(unittest.TestCase):
    def test_published_reports_have_same_protocol(self):
        rows = load_evaluations(ROOT / "results/rma")
        self.assertEqual(len(rows), 6)
        self.assertEqual({row["seed"] for row in rows}, {123, 456, 789})
        self.assertTrue(all(row["total_transitions"] == 8000 for row in rows))
        self.assertTrue(all(row["vx_rmse_m_s"] > 0.48 for row in rows))

    def test_trailing_mean_retains_startup(self):
        self.assertEqual(rolling_mean([1.0, 3.0, 5.0], window=2), [1.0, 2.0, 4.0])


if __name__ == "__main__":
    unittest.main()
