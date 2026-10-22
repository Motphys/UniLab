"""Create quick-look plots from an opt-in RMA evaluation trace."""

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path


def plot_trace(trace_dir, output_dir=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trace_dir = Path(trace_dir)
    output_dir = Path(output_dir) if output_dir else trace_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = trace_dir / "run_config.json"
    if not metadata_path.exists():
        metadata_path = trace_dir / "run_configuration.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    csv_path = trace_dir / "transitions.csv"
    if not csv_path.exists():
        csv_path = trace_dir / "transitions.csv.gz"
    opener = gzip.open if csv_path.suffix == ".gz" else open
    with opener(csv_path, "rt", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No transitions in {trace_dir / 'transitions.csv'}")
    steps = sorted({int(row["step"]) for row in rows})
    velocity_by_step = defaultdict(list)
    command_by_step = defaultdict(list)
    action_rows = []
    reward_columns = [name for name in rows[0] if name.startswith("reward_rate/")]
    reward_by_step = {name: defaultdict(list) for name in reward_columns}
    for row in rows:
        step = int(row["step"])
        velocity_by_step[step].append(float(row["before_vx_m_s"]))
        command_by_step[step].append(float(row["command_vx_m_s"]))
        action_rows.append([float(row[f"action_{i}"]) for i in range(12)])
        for name in reward_columns:
            reward_by_step[name][step].append(float(row[name]))

    dt = float(metadata["control_period_s"])
    elapsed = [step * dt for step in steps]
    velocity_mean = [sum(velocity_by_step[step]) / len(velocity_by_step[step]) for step in steps]
    velocity_std = [
        (sum((value - mean) ** 2 for value in velocity_by_step[step]) / len(velocity_by_step[step]))
        ** 0.5
        for step, mean in zip(steps, velocity_mean, strict=True)
    ]
    commands = [sum(command_by_step[step]) / len(command_by_step[step]) for step in steps]
    fig, axis = plt.subplots(figsize=(10, 4), constrained_layout=True)
    axis.plot(elapsed, commands, "--", color="#C53434", label="Command")
    axis.plot(elapsed, velocity_mean, color="#1D4C79", label="Mean measured vx")
    axis.fill_between(
        elapsed,
        [mean - std for mean, std in zip(velocity_mean, velocity_std, strict=True)],
        [mean + std for mean, std in zip(velocity_mean, velocity_std, strict=True)],
        color="#4679A8",
        alpha=0.2,
        label="Across-env standard deviation",
    )
    axis.set(
        xlabel="Time since evaluation start (s)",
        ylabel="Body-frame vx (m/s)",
        title="Command tracking; each point averages active environments, resets included",
    )
    axis.grid(alpha=0.2)
    axis.legend()
    fig.savefig(output_dir / "velocity_tracking.png", dpi=160)
    plt.close(fig)

    joint_names = metadata["control_contract"]["action_target_joint_order"]
    fig, axis = plt.subplots(figsize=(12, 4), constrained_layout=True)
    axis.boxplot(
        [[row[index] for row in action_rows] for index in range(12)],
        showfliers=False,
    )
    axis.set_xticks(range(1, 13), joint_names)
    axis.axhline(0.95, color="#C53434", linestyle="--", linewidth=1)
    axis.axhline(-0.95, color="#C53434", linestyle="--", linewidth=1)
    axis.set(
        xlabel="Resolved joint target order",
        ylabel="Normalized policy action",
        title="Action distribution; dashed lines flag the |action| ≥ 0.95 indicator",
    )
    axis.tick_params(axis="x", rotation=35)
    axis.grid(axis="y", alpha=0.2)
    fig.savefig(output_dir / "action_distribution.png", dpi=160)
    plt.close(fig)

    if reward_columns:
        fig, axis = plt.subplots(figsize=(11, 5), constrained_layout=True)
        for name in reward_columns:
            values = [
                sum(reward_by_step[name][step]) / len(reward_by_step[name][step]) for step in steps
            ]
            axis.plot(elapsed, values, label=name.removeprefix("reward_rate/"))
        axis.set(
            xlabel="Time since evaluation start (s)",
            ylabel="Weighted reward rate before control-dt scaling",
            title="Reward components averaged across active environments",
        )
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, ncol=2)
        fig.savefig(output_dir / "reward_components.png", dpi=160)
        plt.close(fig)
    return sorted(str(path) for path in output_dir.glob("*.png"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(plot_trace(args.trace_dir, args.output), indent=2))


if __name__ == "__main__":
    main()
