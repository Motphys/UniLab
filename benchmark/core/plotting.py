"""Plotting utilities for benchmark results."""

from pathlib import Path
from typing import Any, Dict, List

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def save_line_plot(
    records: List[Dict[str, Any]],
    x_key: str,
    y_key: str,
    group_key: str,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
    xscale: str = "log",
    yscale: str = "log",
    device_info: str = "",
) -> bool:
    if plt is None or not records:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    groups = sorted({r[group_key] for r in records})

    fig, ax = plt.subplots(figsize=(10, 6))
    for group in groups:
        subset = sorted([r for r in records if r[group_key] == group], key=lambda x: x[x_key])
        if not subset:
            continue
        x = [r[x_key] for r in subset]
        y = [r[y_key] for r in subset]
        ax.plot(x, y, marker="o", label=group)

    full_title = f"{title}\n{device_info}" if device_info else title
    ax.set_title(full_title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xscale:
        ax.set_xscale(xscale, base=2 if xscale == "log" else 10)
    if yscale:
        ax.set_yscale(yscale)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return True
