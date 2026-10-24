"""Compare saved MuJoCo and Genesis runs; never loads or trains a policy."""

import argparse
import csv
import json
import math
from pathlib import Path


def load_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Empty trace: {path}")
    return rows


def series(rows, genesis=False):
    names = ("vx_m_s", "vy_m_s", "yaw_rate_rad_s") if genesis else (
        "after_vx_m_s", "after_vy_m_s", "after_wz_rad_s"
    )
    return [[float(row[name]) for row in rows] for name in names]


def metrics(rows, command, genesis=False):
    values = series(rows, genesis)
    rmse = [math.sqrt(sum((x - target) ** 2 for x in axis) / len(axis))
            for axis, target in zip(values, command, strict=True)]
    falls = sum(row["reset_reason"] == "bad_orientation" for row in rows) if genesis else sum(
        int(row["terminated"]) for row in rows
    )
    return dict(
        samples=len(rows), mean_velocity=[sum(axis) / len(axis) for axis in values],
        rmse_by_axis=rmse, planar_rmse_m_s=math.hypot(rmse[0], rmse[1]),
        fall_or_termination_count=falls,
        near_bound_fraction=[sum(abs(float(row[f"action_{i}"])) >= 0.95 for row in rows) / len(rows)
                             for i in range(12)],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mujoco-trace", type=Path, required=True)
    p.add_argument("--genesis-csv", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--command", type=float, nargs=3, default=[0.5, 0.0, 0.0])
    args = p.parse_args()
    candidates = list(args.mujoco_trace.rglob("transitions.csv"))
    if len(candidates) != 1:
        p.error(f"Expected one MuJoCo trace, got {len(candidates)}; select its exact directory")
    mj, gs = load_rows(candidates[0]), load_rows(args.genesis_csv)
    if {row["env_id"] for row in mj} != {"0"}:
        p.error("Use a single-environment MuJoCo trace")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = dict(
        command=args.command, mujoco=metrics(mj, args.command), genesis=metrics(gs, args.command, True),
        measurement="Both use post-action IMU-site velocity; yaw rate is rad/s",
        limitation="Nominal interface-aligned comparison; contact solvers remain different. No universal pass threshold.",
        sources=dict(mujoco=str(candidates[0].resolve()), genesis=str(args.genesis_csv.resolve())),
    )
    (args.output_dir / "comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t_mj = [float(row["next_time_s"]) for row in mj]
    t_gs = [float(row["sim_time_s"]) for row in gs]
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    for i, (ax, label) in enumerate(zip(axes, ("vx (m/s)", "vy (m/s)", "yaw rate (rad/s)"), strict=True)):
        ax.plot(t_mj, series(mj)[i], label="MuJoCo", linewidth=1, alpha=0.8)
        ax.plot(t_gs, series(gs, True)[i], label="Genesis", linewidth=1, alpha=0.8)
        ax.axhline(args.command[i], linestyle="--", color="black", label="Command")
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
    axes[0].legend(ncol=3)
    axes[0].set_title("Frozen student, nominal evaluation; post-step IMU measurements")
    axes[-1].set_xlabel("Simulation time (s)")
    fig.savefig(args.output_dir / "velocity_comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    x = list(range(12))
    ax.bar([i - 0.2 for i in x], summary["mujoco"]["near_bound_fraction"], width=0.4, label="MuJoCo")
    ax.bar([i + 0.2 for i in x], summary["genesis"]["near_bound_fraction"], width=0.4, label="Genesis")
    names = [f"{leg}_{joint}" for leg in ("FL", "FR", "RL", "RR") for joint in ("hip", "thigh", "calf")]
    ax.set_xticks(x, names, rotation=35)
    ax.set_ylabel("Fraction with |action| >= 0.95")
    ax.set_title("Normalized action proximity to bounds; not joint-limit or torque saturation")
    ax.legend()
    fig.savefig(args.output_dir / "action_near_bounds.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True, constrained_layout=True)
    axes[0].plot(t_gs, [float(row["tilt_deg"]) for row in gs])
    axes[0].set_ylabel("Tilt (deg)")
    axes[1].plot(t_gs, [float(row["base_z_m"]) for row in gs])
    axes[1].set_ylabel("Base height (m)")
    axes[1].set_xlabel("Simulation time (s)")
    axes[0].set_title("Genesis stability trace")
    fig.savefig(args.output_dir / "genesis_stability.png", dpi=160)
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    print(f"Saved three plots in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
