# MuJoCo Backend

MuJoCo is the default backend path in the committed owner configs. The Python
dependencies are the official `mujoco` package (`>=3.5`, with the default
version pinned by the committed `uv.lock`) plus
`mujoco-uni-runtime` in `pyproject.toml`, and the adapter lives
under `src/unilab/base/backend/mujoco/`.

## When To Use It

- You want the default training route for PPO, APPO, off-policy SAC/TD3, or
  FlashSAC.
- The task owner exists only as `conf/.../<task>/mujoco.yaml`.
- You need MuJoCo-specific tooling such as `scripts/play_viser.py` or scene
  export from a MuJoCo XML/MJB model.

## Commands

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco
uv run train --algo appo --task go1_joystick_flat --sim mujoco training.no_play=true
uv run train --algo sac --task g1_walk_flat --sim mujoco
```

Playback mode is resolved by the backend contract in
`src/unilab/base/backend/base.py`. MuJoCo reports physics-state playback support
in `src/unilab/base/backend/mujoco/backend.py`; `auto` playback records video
rather than opening the Motrix native interactive renderer.

## Switching MuJoCo Versions

pyproject only constrains `mujoco>=3.5`; the default solver version is pinned
by the committed `uv.lock`, and uv's prefer-locked semantics keep ordinary
relocks from drifting. The tested window is `>=3.5,<3.11` — switch versions
within it.
The
`mujoco-uni-runtime` native extension is compiled against the `mujoco`
package in this environment and refuses to load against any other version,
so switching versions means repinning `mujoco` and rebuilding the extension:

```bash
make mujoco MJ=3.8.0
```

The target runs `uv lock --upgrade-package mujoco==3.8.0`, clears uv's build
cache for `mujoco-uni-runtime` (the cache cannot see that the extension
depends on the mujoco version), and re-syncs with a forced in-env rebuild.
Without the Makefile shortcut, the equivalent is:

```bash
uv lock --upgrade-package mujoco==3.8.0
uv cache clean mujoco-uni-runtime
uv sync --extra mujoco --extra motrix --reinstall-package mujoco-uni-runtime
```

Skipping the cache clean or the forced reinstall lets uv reuse a cached
extension built against the previous mujoco version, which then fails to
import with a dynamic-linker error (fail-closed, never a silent behavior
change).
