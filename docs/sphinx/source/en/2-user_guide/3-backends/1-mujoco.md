# MuJoCo Backend

MuJoCo is the default backend path in the committed owner configs. On Python
3.11 and newer, the `mujoco` extra installs the official `mujoco` package
(`~=4.0.1`) plus the published `mjbatch-uni==0.3.1` native batch engine, with
exact versions pinned by the committed `uv.lock`. The adapter lives under
`unisim.backend.mujoco`.

## When To Use It

- You want the default training route for PPO, APPO, off-policy SAC, or
  FlashSAC.
- The task owner exists only as `src/unilab/conf/.../<task>/mujoco.yaml`.
- You need MuJoCo-specific tooling such as `scripts/play_viser.py` or scene
  export from a MuJoCo XML/MJB model.

## Commands

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco
uv run train --algo appo --task go2_joystick_flat --sim mujoco training.no_play=true
uv run train --algo sac --task g1_walk_flat --sim mujoco
```

Playback mode is resolved by the backend contract in
`unisim.backend.base`. MuJoCo reports physics-state playback support
in `unisim.backend.mujoco.backend`; `auto` playback records video
rather than opening the Motrix native interactive renderer.

## Switching MuJoCo Versions

On Python 3.11 and newer, pyproject constrains `mujoco~=4.0.1` and pins
`mjbatch-uni==0.3.1`; the committed `uv.lock` pins the exact default versions,
and uv's prefer-locked semantics keep ordinary relocks from drifting. The
`mjbatch` engine records its build-time `mujoco` version and refuses to import
against a different one (fail-closed, never a silent behavior change).
Switching versions therefore requires a compatible `mjbatch` release:

1. bump the `mujoco` bound and the `mjbatch-uni` pin in `pyproject.toml` (and
   mirror `pyproject.rocm.toml`),
2. re-lock (`uv lock`, plus the ROCm lockfile via `make sync-rocm`) and
   re-sync (`uv sync --extra mujoco`).

Prebuilt `mjbatch-uni` wheels are the default installation path. Coordinate
new MuJoCo/engine version pairs with the
[integration fork](https://github.com/unilabsim/mjbatch).
