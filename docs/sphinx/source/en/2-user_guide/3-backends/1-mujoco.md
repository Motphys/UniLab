# MuJoCo Backend

MuJoCo is the default backend path in the committed owner configs. The Python
dependencies are the official `mujoco` package (`~=3.11.0`, with the exact
default version pinned by the committed `uv.lock`) plus the
`mjbatch` native batch engine (currently pinned to the
[integration fork](https://github.com/unilabsim/mjbatch) in
`pyproject.toml`), and the adapter lives under `unisim.backend.mujoco`.

## When To Use It

- You want the default training route for PPO, APPO, off-policy SAC/TD3, or
  FlashSAC.
- The task owner exists only as `src/unilab/conf/.../<task>/mujoco.yaml`.
- You need MuJoCo-specific tooling such as `scripts/play_viser.py` or scene
  export from a MuJoCo XML/MJB model.

## Commands

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco
uv run train --algo appo --task go1_joystick_flat --sim mujoco training.no_play=true
uv run train --algo sac --task g1_walk_flat --sim mujoco
```

Playback mode is resolved by the backend contract in
`unisim.backend.base`. MuJoCo reports physics-state playback support
in `unisim.backend.mujoco.backend`; `auto` playback records video
rather than opening the Motrix native interactive renderer.

## Switching MuJoCo Versions

pyproject constrains `mujoco~=3.11.0`; the committed `uv.lock` pins the exact
default version, and uv's prefer-locked semantics keep ordinary relocks from
drifting. The `mjbatch` engine is built against `mujoco==3.11.0` and records
its build-time `mujoco` version, refusing to import against any other version
(fail-closed, never a silent behavior change). Switching versions therefore
requires an `mjbatch` build against the requested version:

1. bump the `mujoco` bound and the `mjbatch` source pin in
   `pyproject.toml` (and mirror `pyproject.rocm.toml`),
2. re-lock (`uv lock`, plus the ROCm lockfile via `make sync-rocm`) and
   re-sync (`uv sync --extra mujoco`).

The fork's build pins `mujoco==3.11.0` at build time, so the isolated build
always compiles against the matching mujoco. The engine is currently consumed
from the pinned integration fork; its final distribution identity (PyPI
package vs git pin, and prebuilt wheels) is the roadmap's open item — until
then, coordinate version bumps with the
[fork](https://github.com/unilabsim/mjbatch).
