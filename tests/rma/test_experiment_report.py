import json
import sys
import tempfile
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

    def test_rejects_student_report_in_teacher_comparison(self):
        report = json.loads((ROOT / "results/rma/baseline/seed123.json").read_text())
        report["policy"] = "student"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "baseline").mkdir()
            (root / "baseline/seed123.json").write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "Unexpected evaluation protocol"):
                load_evaluations(root)


if __name__ == "__main__":
    unittest.main()
