"""Owner-YAML config contract (Phase 1 safety net) + CLI route guard.

Scope note: the full task/backend compose matrix (reward populated, ``training.sim_backend``
matches the owner path) is already covered dynamically by tests/config/test_config_system.py
(``_supported_task_cases`` -> ``test_supported_task_composes``). This file deliberately does
NOT duplicate that; it adds only the contract checks that file does not cover:
- No orphan top-level ``algorithm:`` in any ppo/appo/offpolicy owner YAML (ADR-0003:
  algorithm hyperparams live under ``algo:``; train_appo reads ``cfg.algo``, so a top-level
  ``algorithm:`` is silently dropped to defaults).
- CLI rejects ``training.sim_backend`` as a route-defining override (cli.py
  RESERVED_OVERRIDE_KEYS), so backend switching must go through ``task=<task>/<backend>``.

The one known live violation (APPO go2/motrix puts ``algorithm:`` at top level) is excluded
from the scan and captured by the strict xfail below — which flips to a hard suite failure
once Phase 1 Step 1.1 moves the block under ``algo:``, forcing the xfail to be removed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[2]

# The single known live orphan-``algorithm:`` violation, excluded from the scan and pinned
# by the strict xfail below (it flips to a suite failure the moment Phase 1 fixes the YAML).
KNOWN_ORPHAN_REL = "appo/task/go2_joystick_flat/motrix.yaml"


def _owner_task_yamls() -> list[Path]:
    """Every ppo/appo/offpolicy owner task YAML, minus the one known-orphan exception."""
    paths: list[Path] = []
    for algo_dir in ("ppo", "appo", "offpolicy"):
        root = REPO / "conf" / algo_dir / "task"
        if not root.is_dir():
            continue
        for path in sorted(root.glob("**/*.yaml")):
            if path.relative_to(REPO / "conf").as_posix() == KNOWN_ORPHAN_REL:
                continue
            paths.append(path)
    return paths


def _compose(algo_dir: str, task: str, overrides: tuple[str, ...] = ()):
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(REPO / "conf" / algo_dir), version_base="1.3"):
        return compose("config", overrides=[f"task={task}", *overrides])


@pytest.mark.parametrize(
    "path", _owner_task_yamls(), ids=lambda p: p.relative_to(REPO / "conf").as_posix()
)
def test_owner_yaml_has_no_orphan_top_level_algorithm(path: Path):
    """Raw-file scan: no owner YAML may carry a top-level ``algorithm:`` (ADR-0003).

    Complements (does not duplicate) test_config_system's compose matrix — that proves the
    composed cfg resolves; this proves owner-file hygiene at the source. The one known
    violation is excluded (see KNOWN_ORPHAN_REL) and covered by the strict xfail below.
    """
    container = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
    assert isinstance(container, dict)
    assert "algorithm" not in container, (
        f"{path.relative_to(REPO / 'conf')}: orphan top-level 'algorithm:' — algorithm "
        "hyperparams belong under 'algo:' (ADR-0003); a top-level block is read by nothing."
    )


def test_cli_rejects_sim_backend_passthrough_override():
    """b.md b5: backend switch must go through task=<task>/<backend>, not a bare override.

    cli.py puts training.sim_backend in RESERVED_OVERRIDE_KEYS, so build_command must reject
    it — bare Hydra compose alone can't prove the CLI guard.
    """
    from unilab.cli import build_command

    with pytest.raises(SystemExit):
        build_command(
            mode="train",
            algo="ppo",
            task="go2_joystick_flat",
            sim="mujoco",
            overrides=["training.sim_backend=motrix"],
        )


@pytest.mark.xfail(
    reason="APPO go2/motrix YAML puts algorithm: at top level (orphan); train_appo reads "
    "cfg.algo, so num_learning_epochs falls back to the default 5. Phase 1 Step 1.1 moves "
    "the block under algo: — this test then XPASSes. strict=True turns that XPASS into a "
    "suite failure, forcing this xfail (and KNOWN_ORPHAN_REL) to be removed once fixed.",
    strict=True,
)
def test_appo_go2_motrix_algorithm_under_algo():
    cfg = _compose("appo", "go2_joystick_flat/motrix")
    assert OmegaConf.select(cfg, "algorithm") is None  # no orphan top-level algorithm
    assert cfg.algo.algorithm.num_learning_epochs == 25  # YAML intent (not default 5)
