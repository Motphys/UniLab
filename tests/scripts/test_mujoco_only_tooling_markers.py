from __future__ import annotations

from pathlib import Path


def test_mujoco_only_tooling_is_marked_explicitly():
    root = Path(__file__).resolve().parents[2]
    target_files = [
        root / "scripts" / "play_interactive.py",
        root / "scripts" / "motion" / "csv_to_npz.py",
        root / "scripts" / "motion" / "replay_npz.py",
        root / "scripts" / "motion" / "bones_seed_csv_to_npz.py",
        root / "scripts" / "motion" / "replay_bones_seed_csv.py",
        root / "src" / "unilab" / "visualization" / "render_many.py",
        root / "src" / "unilab" / "envs" / "locomotion" / "g1" / "symmetry.py",
    ]

    missing = [
        str(path.relative_to(root))
        for path in target_files
        if "MuJoCo-only" not in path.read_text(encoding="utf-8")
    ]

    assert missing == []
