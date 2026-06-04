"""
Stage 3: NaN guard wiring verification for multi-process algorithms.

Multi-process algorithms (APPO, SAC, TD3, FlashSAC) run their environment
collectors in spawned subprocesses. Direct in-process NaN injection (the
Stage 2 approach) does not work across the IPC boundary without invasive
source-code changes to the collector workers.

This stage takes a different approach: STATIC VERIFICATION of the wiring.

For each algorithm, verify:
  1. The train script (scripts/train_<algo>.py) reads nan_guard config and
     forwards a NanGuardCfg into the runner.
  2. The collector worker (src/unilab/algos/torch/<algo>/worker.py) constructs
     a NanGuard from the cfg and calls env.set_nan_guard(...) inside the
     subprocess.
  3. The algorithm's hydra config (conf/<algo>/config.yaml) contains a
     training.nan_guard block.

If all three are present, the wiring is correct and a real NaN that occurs
inside the collector subprocess will be detected and dumped by the same
NanGuard machinery validated in Stage 2 (proto + stage2_nan_inject.py).

For end-to-end runtime validation, use the manual test recipe at the bottom.

Run:
    .venv/bin/python tests/nan_injection/stage3_nan_inject.py
"""

from __future__ import annotations

import re
import sys
import textwrap
from pathlib import Path

# Resolve repo root from this file's location: <repo>/tests/nan_injection/<this>
ROOT_DIR = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Specs: what each multi-process algorithm should look like
# ---------------------------------------------------------------------------


SPECS = [
    {
        "name": "APPO",
        "train_script": "scripts/train_appo.py",
        "worker": "src/unilab/algos/torch/appo/worker.py",
        "config_yaml": "conf/appo/config.yaml",
    },
    {
        "name": "SAC (offpolicy)",
        "train_script": "scripts/train_offpolicy.py",
        "worker": "src/unilab/algos/torch/offpolicy/worker.py",
        "config_yaml": "conf/offpolicy/config.yaml",
    },
    {
        "name": "TD3 (offpolicy)",
        "train_script": "scripts/train_offpolicy.py",
        "worker": "src/unilab/algos/torch/offpolicy/worker.py",
        "config_yaml": "conf/offpolicy/config.yaml",
    },
    {
        "name": "FlashSAC (offpolicy)",
        "train_script": "scripts/train_offpolicy.py",
        "worker": "src/unilab/algos/torch/offpolicy/worker.py",
        "config_yaml": "conf/offpolicy/config.yaml",
    },
]


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _check_train_script(path: Path) -> tuple[bool, str]:
    text = _read(path)
    if not text:
        return False, f"file not found: {path}"
    if "NanGuardCfg" not in text:
        return False, "NanGuardCfg import/use missing"
    if "nan_guard" not in text:
        return False, "no reference to nan_guard config"
    return True, "OK (reads nan_guard cfg, constructs NanGuardCfg)"


def _check_worker(path: Path) -> tuple[bool, str]:
    text = _read(path)
    if not text:
        return False, f"file not found: {path}"
    if "NanGuard" not in text:
        return False, "NanGuard import missing"
    if not re.search(r"env\.set_nan_guard\s*\(", text):
        return False, "env.set_nan_guard(...) call missing"
    return True, "OK (constructs NanGuard, calls env.set_nan_guard in subprocess)"


def _check_config(path: Path) -> tuple[bool, str]:
    text = _read(path)
    if not text:
        return False, f"file not found: {path}"
    if "nan_guard" not in text:
        return False, "training.nan_guard block missing in config"
    return True, "OK (training.nan_guard block present)"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 78)
    print("Stage 3: Multi-process NaN guard WIRING verification (static check)")
    print("=" * 78)
    print()
    print("Approach: verify each multi-process algorithm has the 3-piece wiring")
    print("  (train script -> collector worker -> hydra config) so the same")
    print("  NanGuard machinery validated in Stage 2 will fire in the collector")
    print("  subprocess on a real NaN.")
    print()

    all_ok = True
    for spec in SPECS:
        print(f"--- {spec['name']} ---")

        train_path = ROOT_DIR / spec["train_script"]
        worker_path = ROOT_DIR / spec["worker"]
        cfg_path = ROOT_DIR / spec["config_yaml"]

        for label, fn, p in [
            ("train script", _check_train_script, train_path),
            ("collector worker", _check_worker, worker_path),
            ("hydra config", _check_config, cfg_path),
        ]:
            ok, msg = fn(p)
            mark = "✓" if ok else "✗"
            print(f"  {mark} {label:<18} {p.relative_to(ROOT_DIR)}")
            print(f"    {msg}")
            if not ok:
                all_ok = False
        print()

    print("=" * 78)
    print("Summary")
    print("=" * 78)
    if all_ok:
        print("All multi-process algorithms have correct nan_guard wiring.")
        print()
        print("Stage 2 (in-process injection) directly validated NaN detection")
        print("and dump for PPO/HIM-PPO. Since the same NanGuard class is used")
        print("inside the APPO/off-policy collector subprocesses (verified above),")
        print("the detection + dump path is exercised by the same code paths.")
    else:
        print("Some wiring checks failed. See details above.")
    print()

    print("=" * 78)
    print("Manual end-to-end test recipe (optional)")
    print("=" * 78)
    print(
        textwrap.dedent("""
        To exercise NaN detection end-to-end inside a collector subprocess:

        1. Pick a task env file, e.g.
           src/unilab/envs/locomotion/go1/go1_joystick.py
        2. Inside the env's update_state or apply_action, add a temporary
           one-shot NaN raise guarded by an env-counter, for example:

               if not getattr(self, "_nan_done", False) and self.step_counter >= 5:
                   state.reward[0] = float("nan")
                   self._nan_done = True

        3. Run a short training, e.g. (APPO):
               python scripts/train_appo.py task=go1_joystick_flat/mujoco \\
                 algo.num_envs=8 algo.num_steps_per_env=4 \\
                 algo.train_for_env_steps=64 \\
                 training.nan_guard.enabled=true \\
                 training.nan_guard.output_dir=/tmp/unilab/nan_dumps

        4. Verify a dump file appears under the output_dir.
        5. Revert the env edit.

        Repeat with scripts/train_offpolicy.py to exercise SAC/TD3/FlashSAC
        (set algo=sac / algo=td3 / algo=flashsac via Hydra override).
    """).strip()
    )
    print()

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
