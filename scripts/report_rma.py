"""Rebuild the checked-in RMA result tables and figures."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rsl_rl"))

from unilab_rma.experiment_report import build_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results/rma")
    args = parser.parse_args()
    print(json.dumps(build_report(args.results), indent=2))


if __name__ == "__main__":
    main()
