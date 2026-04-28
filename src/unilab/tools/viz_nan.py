"""Interactive viewer for NaN guard state dumps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np


def load_dump(dump_path: str) -> dict:
    data = np.load(dump_path, allow_pickle=True)
    states = data["states"]
    metadata = {}
    for key in data.files:
        if key.startswith("meta_"):
            val = data[key]
            metadata[key[5:]] = val.item() if val.ndim == 0 else val
    return {"states": states, "metadata": metadata}


def replay_dump(dump_path: str, env_index: int = 0) -> None:
    import mujoco as _mujoco
    import mujoco.viewer as _viewer  # noqa: F401

    mujoco: Any = _mujoco

    dump = load_dump(dump_path)
    states = dump["states"]
    meta = dump["metadata"]

    if states.size == 0:
        print("No physics states in dump (backend may not support state playback).")
        print(f"Metadata: {meta}")
        return

    model_file = str(meta.get("model_file", ""))
    dump_dir = Path(dump_path).parent
    model_path = None
    if model_file and Path(model_file).is_file():
        model_path = model_file
    else:
        for f in sorted(dump_dir.glob("*_model.*")):
            model_path = str(f)
            break

    if model_path is None:
        print(f"Cannot find model file. model_file in metadata: {model_file}")
        return

    model = (
        mujoco.MjModel.from_xml_path(model_path)
        if model_path.endswith(".xml")
        else mujoco.MjModel.from_binary_path(model_path)
    )
    d = mujoco.MjData(model)

    num_steps = states.shape[0]
    num_envs = states.shape[1] if states.ndim >= 2 else 1
    nan_env_ids = meta.get("nan_env_ids", np.array([]))
    step_detected = meta.get("detection_step", "?")

    print(f"Dump: {dump_path}")
    print(f"Steps in buffer: {num_steps}, Envs: {num_envs}")
    print(f"NaN detected at step {step_detected}, env ids: {nan_env_ids}")
    print(f"Viewing env index: {env_index}")
    print("Press Ctrl+C to exit.")

    state_size = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_PHYSICS)

    with mujoco.viewer.launch_passive(model, d) as viewer:
        step_idx = 0
        while viewer.is_running():
            if states.ndim >= 3:
                flat = states[step_idx, env_index]
            elif states.ndim == 2:
                flat = states[step_idx]
            else:
                break

            if flat.shape[0] >= state_size:
                mujoco.mj_setState(model, d, flat[:state_size], mujoco.mjtState.mjSTATE_PHYSICS)
                mujoco.mj_forward(model, d)

            viewer.sync()

            import time

            time.sleep(1.0 / 30.0)
            step_idx = (step_idx + 1) % num_steps


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="unilab-viz-nan",
        description="Replay a NaN guard state dump in MuJoCo viewer.",
    )
    parser.add_argument("dump_path", help="Path to the .npz dump file")
    parser.add_argument("--env-index", type=int, default=0, help="Environment index to visualize")
    args = parser.parse_args(argv)

    replay_dump(args.dump_path, env_index=args.env_index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
