#!/usr/bin/env python3
"""Visualize collection-time decomposition from MLX PPO train logs."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

ITER_RE = re.compile(
    r"\[iter\s+(\d+)/(\d+)\].*?collect=([0-9.]+)s.*?"
    r"prof\(act/step/core/post/reset/buf/fin/ep\)="
    r"([0-9.]+)/([0-9.]+)/([0-9.]+)/([0-9.]+)/([0-9.]+)/([0-9.]+)/([0-9.]+)/([0-9.]+)"
    r"(?:\s+reset_sub\(idx/call/scatter/info\)=([0-9.]+)/([0-9.]+)/([0-9.]+)/([0-9.]+))?"
)


def _latest_run_dir(base: Path) -> Path:
    runs = [p for p in base.iterdir() if p.is_dir()]
    if not runs:
        raise FileNotFoundError(f"No run dirs under {base}")
    return sorted(runs, key=lambda p: p.name)[-1]


def _mean_std(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    return {
        "mean": float(statistics.mean(values)),
        "std": float(statistics.pstdev(values)),
    }


def parse_rows(log_text: str) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for m in ITER_RE.finditer(log_text):
        rows.append(
            {
                "iter": float(m.group(1)),
                "collect": float(m.group(3)),
                "act": float(m.group(4)),
                "step": float(m.group(5)),
                "core": float(m.group(6)),
                "post": float(m.group(7)),
                "reset": float(m.group(8)),
                "buffer": float(m.group(9)),
                "finite": float(m.group(10)),
                "episode": float(m.group(11)),
                "reset_idx": float(m.group(12) or 0.0),
                "reset_call": float(m.group(13) or 0.0),
                "reset_scatter": float(m.group(14) or 0.0),
                "reset_info": float(m.group(15) or 0.0),
            }
        )
    return rows


def plot(rows: list[dict[str, float]], out_dir: Path) -> None:
    iters = [int(r["iter"]) for r in rows]
    collect = [r["collect"] for r in rows]
    core = [r["core"] for r in rows]
    post = [r["post"] for r in rows]
    reset = [r["reset"] for r in rows]
    act = [r["act"] for r in rows]
    episode = [r["episode"] for r in rows]
    residual = [
        max(c - (a + co + po + re + ep), 0.0)
        for c, a, co, po, re, ep in zip(collect, act, core, post, reset, episode)
    ]

    plt.figure(figsize=(10, 5))
    plt.plot(iters, collect, marker="o", label="collection")
    plt.plot(iters, core, marker=".", label="env_core")
    plt.plot(iters, reset, marker=".", label="env_reset")
    plt.plot(iters, episode, marker=".", label="episode_stats")
    plt.xlabel("Iteration")
    plt.ylabel("Time (s)")
    plt.title("Collection Timing by Iteration")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "collection_timing_lines.png", dpi=140)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.stackplot(
        iters,
        core,
        reset,
        post,
        episode,
        act,
        residual,
        labels=["env_core", "env_reset", "env_post", "episode_stats", "act", "residual"],
        alpha=0.9,
    )
    plt.plot(iters, collect, color="black", linewidth=1.5, label="collection_total")
    plt.xlabel("Iteration")
    plt.ylabel("Time (s)")
    plt.title("Collection Decomposition (Stacked)")
    plt.legend(loc="upper right", ncol=2)
    plt.tight_layout()
    plt.savefig(out_dir / "collection_timing_stacked.png", dpi=140)
    plt.close()

    reset_idx = [r["reset_idx"] for r in rows]
    reset_call = [r["reset_call"] for r in rows]
    reset_scatter = [r["reset_scatter"] for r in rows]
    reset_info = [r["reset_info"] for r in rows]
    plt.figure(figsize=(10, 5))
    plt.stackplot(
        iters,
        reset_call,
        reset_scatter,
        reset_idx,
        reset_info,
        labels=["reset_call", "reset_scatter", "reset_index", "reset_info"],
        alpha=0.9,
    )
    plt.plot(iters, reset, color="black", linewidth=1.5, label="reset_total")
    plt.xlabel("Iteration")
    plt.ylabel("Time (s)")
    plt.title("Reset Path Decomposition")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_dir / "reset_timing_stacked.png", dpi=140)
    plt.close()


def summarize(rows: list[dict[str, float]]) -> dict[str, object]:
    keys = [
        "collect",
        "act",
        "step",
        "core",
        "post",
        "reset",
        "buffer",
        "finite",
        "episode",
        "reset_idx",
        "reset_call",
        "reset_scatter",
        "reset_info",
    ]
    stats = {k: _mean_std([r[k] for r in rows]) for k in keys}
    collect_mean = max(stats["collect"]["mean"], 1e-9)
    share = {k: float(100.0 * stats[k]["mean"] / collect_mean) for k in keys if k != "collect"}
    return {"n_iters": len(rows), "stats_seconds": stats, "share_vs_collect_percent": share}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("logs/mlx_rl_train/Go2JoystickFlatTerrain"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.log_file is not None:
        log_file = args.log_file
        run_dir = log_file.parent
    else:
        run_dir = _latest_run_dir(args.run_root)
        log_file = run_dir / "train.log"

    if not log_file.exists():
        raise FileNotFoundError(f"Train log not found: {log_file}")

    out_dir = args.output_dir if args.output_dir is not None else run_dir / "timing_viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = parse_rows(log_file.read_text())
    if not rows:
        raise RuntimeError(f"No profile rows parsed from {log_file}")

    plot(rows, out_dir)
    summary = summarize(rows)
    summary["log_file"] = str(log_file)
    (out_dir / "collection_timing_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    print(f"Parsed iterations: {summary['n_iters']}")
    print(f"Output dir: {out_dir}")


if __name__ == "__main__":
    main()
