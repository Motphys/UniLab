"""Build a static, auditable report from selected RMA evaluation and CSV data."""

import csv
import json
from pathlib import Path


def load_evaluations(root):
    """Read published reports; fail on mixed command or malformed comparisons."""
    root = Path(root)
    records = []
    for label in ("baseline", "forward_v1"):
        for seed in (123, 456, 789):
            report = json.loads((root / label / f"seed{seed}.json").read_text())
            diagnostics = report["diagnostics"]
            if report["seed"] != seed or diagnostics["mean_command"] != [0.5, 0.0, 0.0]:
                raise ValueError(f"Unexpected evaluation protocol: {label}, seed {seed}")
            if report["total_transitions"] != 8000:
                raise ValueError("Comparison expects 8 environments x 1000 control steps")
            records.append(
                {
                    "model": label,
                    "seed": seed,
                    "target_vx_m_s": 0.5,
                    "mean_vx_m_s": diagnostics["mean_measured_velocity"][0],
                    "vx_rmse_m_s": diagnostics["tracking_rmse_by_axis"][0],
                    "planar_rmse_m_s": report["planar_velocity_rmse_m_per_s"],
                    "terminations": report["terminations"],
                    "total_transitions": report["total_transitions"],
                }
            )
    return records


def rolling_mean(values, window=100):
    """Trailing mean over available numeric rows, retaining startup samples."""
    output, total = [], 0.0
    for index, value in enumerate(values):
        total += value
        if index >= window:
            total -= values[index - window]
        output.append(total / min(index + 1, window))
    return output


def build_report(root):
    """Generate CSV/PNG outputs without importing or running the simulator."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(root)
    records = load_evaluations(root)
    with (root / "evaluation_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    colors = ("#5C6F91", "#E38B3B")
    for index, label in enumerate(("baseline", "forward_v1")):
        selected = [row for row in records if row["model"] == label]
        for axis, field in zip(axes, ("mean_vx_m_s", "vx_rmse_m_s"), strict=True):
            values = [row[field] for row in selected]
            axis.bar(index, sum(values) / len(values), color=colors[index], alpha=0.65)
            axis.scatter([index - 0.07, index, index + 0.07], values, color="#222222", zorder=3)
    axes[0].axhline(0.5, color="#C53434", linestyle="--", label="Command: 0.5 m/s")
    axes[0].set(
        ylabel="Mean body-frame vx (m/s)", title="Measured forward speed", ylim=(-0.04, 0.56)
    )
    axes[0].legend(fontsize=8)
    axes[1].set(ylabel="Forward tracking RMSE (m/s)", title="Forward speed error", ylim=(0, 0.56))
    for axis in axes:
        axis.set_xticks([0, 1], ["Original teacher", "Forward v1"])
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle(
        "Same checkpoint per group; dots are evaluation seeds, not training replicates", fontsize=10
    )
    fig.savefig(figures / "velocity_comparison.png", dpi=170)
    plt.close(fig)

    with (root / "forward_v1/training_metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    iterations = [int(row["iteration"]) for row in rows]
    rewards = [float(row["mean_step_reward"]) for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
    axes[0].plot(iterations, rewards, alpha=0.18, color="#4679A8", label="Per-update reward")
    axes[0].plot(
        iterations, rolling_mean(rewards), color="#1D4C79", label="Trailing 100-update mean"
    )
    axes[0].set(
        ylabel="Mean step reward", title="Forward v1: 64 environments, rollout 24, training seed 42"
    )
    completed = [
        (int(row["iteration"]), float(row["mean_episode_return"]))
        for row in rows
        if row["mean_episode_return"]
    ]
    axes[1].scatter(
        [x for x, _ in completed], [y for _, y in completed], s=5, alpha=0.3, color="#E38B3B"
    )
    axes[1].set(
        xlabel="Completed PPO updates",
        ylabel="Completed episode return",
        title="Only updates containing completed episodes are shown",
    )
    for axis in axes:
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.savefig(figures / "forward_v1_training.png", dpi=170)
    plt.close(fig)
    return {"evaluation_reports": len(records), "training_rows": len(rows), "figures": str(figures)}
