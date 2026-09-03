"""Compare MicroDuck PPO training metrics between UniLab and upstream microduck_rl runs.

Parses rsl_rl tensorboard event files from one or more run directories per side,
aggregates across seeds (final-window statistics and convergence speed), and
emits a markdown comparison report plus an optional JSON dump of all curves.

    uv run scripts/microduck_alignment_compare.py \
        --unilab logs/rsl_rl_ppo/MicroduckVelocityFlat/<dir1> logs/.../<dir2> \
        --upstream /path/to/microduck_rl/logs/rsl_rl/velocity/<dirA> \
        [--output report.md] [--json out.json]

Run metadata is read from ``run_config.json`` (UniLab) or ``params/agent.yaml``
(upstream); both are optional and fall back to the directory name. Read-only.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import re
import statistics
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# rsl_rl scalar tags shared by both sides.
OVERVIEW_TAGS = ("Train/mean_reward", "Train/mean_episode_length")
REWARD_PREFIX = "Episode_Reward/"
TERMINATION_PREFIX = "Episode_Termination/"

# Reward/termination term names differ between the two codebases for a handful
# of entries. Keys are UniLab term names, values the upstream equivalents;
# terms absent here are assumed to share the same name on both sides.
REWARD_TERM_ALIASES = {
    "tracking_lin_vel": "track_linear_velocity",
    "tracking_ang_vel": "track_angular_velocity",
    "leg_pose": "pose",
    "action_rate": "action_rate_l2",
}
TERMINATION_ALIASES = {
    "tilt": "fell_over",
}

# Fraction of trailing iterations used for final-window statistics.
FINAL_WINDOW_FRAC = 0.2
# Convergence threshold: first iteration where mean_reward reaches this share
# of its own final-window mean.
CONVERGENCE_FRAC = 0.8


@dataclasses.dataclass
class RunData:
    """Scalar curves and metadata for a single training run."""

    run_dir: str
    seed: int | None
    max_iterations: int | None
    curves: dict[str, list[tuple[int, float]]]  # tag -> [(iteration, value)]


def _parse_agent_scalar(text: str, key: str) -> int | None:
    """Extract a top-level integer field from rsl_rl's agent.yaml.

    The file carries ``!!python/tuple`` tags that SafeLoader cannot parse, and
    only two flat fields are needed, so a regex keeps this dependency-free.
    """
    match = re.search(rf"^{re.escape(key)}:\s*(\d+)\s*$", text, re.MULTILINE)
    return int(match.group(1)) if match else None


def _load_curves(run_dir: Path) -> dict[str, list[tuple[int, float]]]:
    """Load the scalar tags of interest from every tfevents file in a run dir."""
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run directory not found: {run_dir}")
    accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    curves: dict[str, list[tuple[int, float]]] = {}
    for tag in accumulator.Tags()["scalars"]:
        if tag in OVERVIEW_TAGS or tag.startswith((REWARD_PREFIX, TERMINATION_PREFIX)):
            curves[tag] = [(e.step, e.value) for e in accumulator.Scalars(tag)]
    return curves


def _unilab_meta(run_dir: Path) -> tuple[int | None, int | None]:
    config_path = run_dir / "run_config.json"
    if not config_path.is_file():
        return None, None
    payload = json.loads(config_path.read_text())
    algo = payload.get("config", {}).get("algo", {})
    seed = payload.get("run", {}).get("effective_seed", algo.get("seed"))
    max_iterations = algo.get("max_iterations")
    return seed, max_iterations


def _upstream_meta(run_dir: Path) -> tuple[int | None, int | None]:
    agent_path = run_dir / "params" / "agent.yaml"
    if not agent_path.is_file():
        return None, None
    text = agent_path.read_text()
    return _parse_agent_scalar(text, "seed"), _parse_agent_scalar(text, "max_iterations")


def load_runs(run_dirs: list[str], side: str) -> list[RunData]:
    """Load runs; directories without rsl_rl scalars (e.g. aborted runs) are skipped."""
    meta_fn = _unilab_meta if side == "unilab" else _upstream_meta
    runs = []
    for raw in run_dirs:
        run_dir = Path(raw)
        curves = _load_curves(run_dir)
        if not curves:
            print(f"warning: no rsl_rl scalar tags under {run_dir}, skipped", file=sys.stderr)
            continue
        seed, max_iterations = meta_fn(run_dir)
        runs.append(
            RunData(
                run_dir=str(run_dir),
                seed=seed,
                max_iterations=max_iterations,
                curves=curves,
            )
        )
    if not runs:
        raise ValueError(f"no usable {side} run directories (all empty or missing)")
    return runs


def _final_window(values: list[float]) -> list[float]:
    n_tail = max(1, math.ceil(len(values) * FINAL_WINDOW_FRAC))
    return values[-n_tail:]


def _final_mean(points: list[tuple[int, float]]) -> float | None:
    if not points:
        return None
    return statistics.mean(_final_window([v for _, v in points]))


def _convergence_iteration(points: list[tuple[int, float]]) -> int | None:
    """First iteration where mean_reward reaches CONVERGENCE_FRAC of its final mean."""
    if not points:
        return None
    final = _final_mean(points)
    if final is None or final <= 0:
        return None
    threshold = CONVERGENCE_FRAC * final
    for step, value in points:
        if value >= threshold:
            return step
    return None


def _aggregate(per_run: list[float | None]) -> tuple[float | None, float | None]:
    values = [v for v in per_run if v is not None]
    if not values:
        return None, None
    std = statistics.stdev(values) if len(values) > 1 else None
    return statistics.mean(values), std


def summarize_side(runs: list[RunData]) -> dict[str, dict[str, float | None]]:
    """Per-tag cross-seed statistics: final-window mean/std of per-run means."""
    tags = sorted({tag for run in runs for tag in run.curves})
    summary: dict[str, dict[str, float | None]] = {}
    for tag in tags:
        per_run = [_final_mean(run.curves.get(tag, [])) for run in runs]
        mean, std = _aggregate(per_run)
        summary[tag] = {"mean": mean, "std": std, "n_runs": len(runs)}
    convergence = [_convergence_iteration(run.curves.get("Train/mean_reward", [])) for run in runs]
    conv_mean, conv_std = _aggregate([float(it) if it is not None else None for it in convergence])
    summary["Train/mean_reward@convergence_iter"] = {
        "mean": conv_mean,
        "std": conv_std,
        "n_runs": len(runs),
    }
    return summary


def _fmt_stat(entry: dict[str, float | None]) -> str:
    mean, std = entry["mean"], entry["std"]
    if mean is None:
        return "n/a"
    if std is None:
        return f"{mean:.3f} (n=1)"
    return f"{mean:.3f} ± {std:.3f}"


def _fmt_rel_diff(unilab: dict[str, float | None], upstream: dict[str, float | None]) -> str:
    u_mean, up_mean = unilab["mean"], upstream["mean"]
    if u_mean is None or up_mean is None or math.isclose(up_mean, 0.0):
        return "n/a"
    return f"{(u_mean - up_mean) / abs(up_mean) * 100:+.1f}%"


def _canonical_terms(
    unilab_tags: list[str], upstream_tags: list[str], prefix: str, aliases: dict[str, str]
) -> tuple[list[str], list[str], list[str]]:
    """Pair terms across sides via aliases; return (paired, unilab-only, upstream-only)."""
    unilab_terms = {t.removeprefix(prefix) for t in unilab_tags}
    upstream_terms = {t.removeprefix(prefix) for t in upstream_tags}
    upstream_reverse = {up: uni for uni, up in aliases.items()}
    paired, unilab_only = [], []
    for term in sorted(unilab_terms):
        upstream_name = aliases.get(term, term)
        if upstream_name in upstream_terms:
            paired.append(term)
        else:
            unilab_only.append(term)
    upstream_only = sorted(
        t for t in upstream_terms if upstream_reverse.get(t, t) not in unilab_terms
    )
    return paired, unilab_only, upstream_only


def _render_run_table(side: str, runs: list[RunData]) -> list[str]:
    lines = ["| run | seed | max_iterations |", "| --- | --- | --- |"]
    for run in runs:
        lines.append(
            f"| `{run.run_dir}` | {run.seed if run.seed is not None else '?'} "
            f"| {run.max_iterations if run.max_iterations is not None else '?'} |"
        )
    if len(runs) < 2:
        lines.append("")
        lines.append(f"> ⚠ {side} 侧只有 {len(runs)} 个 run，std 不可用，统计仅供参考。")
    return lines


def build_report(unilab_runs: list[RunData], upstream_runs: list[RunData]) -> str:
    unilab_summary = summarize_side(unilab_runs)
    upstream_summary = summarize_side(upstream_runs)
    lines = ["# MicroDuck PPO 对比报告", ""]
    lines.append(
        f"终段窗口 = 每 run 末尾 {FINAL_WINDOW_FRAC:.0%} iterations；"
        f"收敛速度 = mean_reward 首次达到自身终段均值 {CONVERGENCE_FRAC:.0%} 的 iteration。"
    )
    lines.append("")
    lines.append("## Runs")
    lines.append("")
    lines.append("**UniLab**")
    lines.extend(_render_run_table("unilab", unilab_runs))
    lines.append("")
    lines.append("**Upstream (microduck_rl)**")
    lines.extend(_render_run_table("upstream", upstream_runs))
    lines.append("")

    lines.append("## 总览指标")
    lines.append("")
    lines.append("| 指标 | UniLab | 上游 | 相对差 |")
    lines.append("| --- | --- | --- | --- |")
    overview_rows = [
        ("Train/mean_reward", "mean_reward 终段均值"),
        ("Train/mean_episode_length", "episode length 终段均值"),
        ("Train/mean_reward@convergence_iter", "收敛 iteration (80% final reward)"),
    ]
    for tag, label in overview_rows:
        u = unilab_summary.get(tag, {"mean": None, "std": None})
        up = upstream_summary.get(tag, {"mean": None, "std": None})
        lines.append(f"| {label} | {_fmt_stat(u)} | {_fmt_stat(up)} | {_fmt_rel_diff(u, up)} |")
    lines.append("")

    def _term_section(title: str, prefix: str, aliases: dict[str, str]) -> None:
        lines.append(f"## {title}")
        lines.append("")
        paired, unilab_only, upstream_only = _canonical_terms(
            [t for t in unilab_summary if t.startswith(prefix)],
            [t for t in upstream_summary if t.startswith(prefix)],
            prefix,
            aliases,
        )
        lines.append("| term (UniLab / 上游) | UniLab 终段 | 上游终段 | 相对差 |")
        lines.append("| --- | --- | --- | --- |")
        for term in paired:
            u = unilab_summary[prefix + term]
            up = upstream_summary[prefix + aliases.get(term, term)]
            alias_note = f" / {aliases[term]}" if term in aliases else ""
            lines.append(
                f"| {term}{alias_note} | {_fmt_stat(u)} | {_fmt_stat(up)} | {_fmt_rel_diff(u, up)} |"
            )
        if unilab_only:
            lines.append("")
            lines.append(f"UniLab 独有 term: {', '.join(f'`{t}`' for t in unilab_only)}")
        if upstream_only:
            lines.append(f"上游独有 term: {', '.join(f'`{t}`' for t in upstream_only)}")
        lines.append("")

    _term_section("Reward term 终段对比", REWARD_PREFIX, REWARD_TERM_ALIASES)
    _term_section("Termination 构成对比", TERMINATION_PREFIX, TERMINATION_ALIASES)

    # Termination composition shares (UniLab side naming for paired terms).
    lines.append("## Termination 终段占比")
    lines.append("")
    lines.append("| term | UniLab 占比 | 上游占比 |")
    lines.append("| --- | --- | --- |")

    def _shares(summary: dict[str, dict[str, float | None]]) -> dict[str, float]:
        means = {
            t.removeprefix(TERMINATION_PREFIX): e["mean"] or 0.0
            for t, e in summary.items()
            if t.startswith(TERMINATION_PREFIX)
        }
        total = sum(means.values())
        return {k: v / total for k, v in means.items()} if total > 0 else {}

    u_shares, up_shares = _shares(unilab_summary), _shares(upstream_summary)
    paired_terms, u_only, up_only = _canonical_terms(
        [t for t in unilab_summary if t.startswith(TERMINATION_PREFIX)],
        [t for t in upstream_summary if t.startswith(TERMINATION_PREFIX)],
        TERMINATION_PREFIX,
        TERMINATION_ALIASES,
    )
    for term in paired_terms + u_only:
        up_name = TERMINATION_ALIASES.get(term, term)
        lines.append(
            f"| {term} | {u_shares.get(term, 0.0):.1%} | {up_shares.get(up_name, 0.0):.1%} |"
        )
    for term in up_only:
        lines.append(f"| {term} (上游独有) | — | {up_shares.get(term, 0.0):.1%} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--unilab", nargs="+", required=True, help="UniLab run directories.")
    parser.add_argument("--upstream", nargs="+", required=True, help="Upstream run directories.")
    parser.add_argument("--output", help="Write the markdown report to this path.")
    parser.add_argument("--json", dest="json_path", help="Dump full curve summaries as JSON.")
    args = parser.parse_args()

    unilab_runs = load_runs(args.unilab, "unilab")
    upstream_runs = load_runs(args.upstream, "upstream")

    report = build_report(unilab_runs, upstream_runs)
    print(report)
    if args.output:
        Path(args.output).write_text(report + "\n")

    if args.json_path:
        payload = {
            "unilab": {
                "runs": [dataclasses.asdict(run) for run in unilab_runs],
                "summary": summarize_side(unilab_runs),
            },
            "upstream": {
                "runs": [dataclasses.asdict(run) for run in upstream_runs],
                "summary": summarize_side(upstream_runs),
            },
        }
        Path(args.json_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
