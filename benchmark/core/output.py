"""Output formatting utilities."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def print_table(records: List[Dict[str, Any]], headers: List[str]) -> None:
    if not records:
        print("No records.")
        return
    rows = [[str(r.get(h, "")) for h in headers] for r in records]
    col_w = [len(h) for h in headers]
    for row in rows:
        for i, v in enumerate(row):
            col_w[i] = max(col_w[i], len(v))
    fmt = lambda vals: " | ".join(v.ljust(col_w[i]) for i, v in enumerate(vals))
    print(fmt(headers))
    print("-+-".join("-" * w for w in col_w))
    for row in rows:
        print(fmt(row))


def save_json(path: Path, results: List[Dict], meta: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            **meta,
        },
        "results": results,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved: {path.resolve()}")
