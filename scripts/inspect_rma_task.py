"""Show fully resolved task differences without building an environment."""

import argparse
import json

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from train_rma import ROOT
from unilab_rma.evaluation_diagnostics import FORWARD_TASK, RMA_TASKS


def differences(before, after, prefix=""):
    if isinstance(before, dict) and isinstance(after, dict):
        result = []
        for key in sorted(before.keys() | after.keys()):
            path = f"{prefix}.{key}" if prefix else key
            if key not in before or key not in after:
                result.append(dict(path=path, before=before.get(key), after=after.get(key)))
            else:
                result.extend(differences(before[key], after[key], path))
        return result
    if before != after:
        return [dict(path=prefix, before=before, after=after)]
    return []


def resolved(task):
    with initialize_config_dir(config_dir=str(ROOT / "src/unilab/conf/ppo"), version_base=None):
        config = compose(config_name="config", overrides=[f"task={task}"])
    return OmegaConf.to_container(config, resolve=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=RMA_TASKS, required=True)
    parser.add_argument("--compare-with", choices=RMA_TASKS, default=FORWARD_TASK)
    parser.add_argument("--expect-only", help="Fail unless exactly this resolved field changes")
    args = parser.parse_args()
    baseline, candidate = resolved(args.compare_with), resolved(args.task)
    changes = differences(baseline, candidate)
    print(
        json.dumps(
            dict(
                root=str(ROOT),
                baseline=args.compare_with,
                candidate=args.task,
                baseline_tracking=baseline["reward"]["tracking_lin_vel"],
                candidate_tracking=candidate["reward"]["tracking_lin_vel"],
                changes=changes,
            ),
            indent=2,
        )
    )
    if args.expect_only and [row["path"] for row in changes] != [args.expect_only]:
        parser.exit(1, "Single-variable check FAILED; inspect changes above.\n")


if __name__ == "__main__":
    main()
