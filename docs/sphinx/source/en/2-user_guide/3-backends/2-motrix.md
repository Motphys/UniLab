# Motrix Backend

Motrix is an optional backend installed through the `motrix` extra. The pinned
package is `motrixsim-core==0.8.1.dev104665` in `pyproject.toml`, and the adapter
lives under `src/unilab/base/backend/motrix/`.

## Setup

```bash
uv sync --extra motrix
```

`make setup-motrix` runs the same dependency sync and installs shell completion.

## When To Use It

- The task owner exists under `conf/.../<task>/motrix.yaml`.
- You want Motrix native interactive playback; the backend advertises native
  interactive renderer and video-capture capability.
- The generated support matrix marks your entrypoint/task/backend combination as
  configured or tested.

## Commands

```bash
uv run train --algo ppo --task go2_joystick_flat --sim motrix training.no_play=true
uv run eval --algo ppo --task go2_joystick_flat --sim motrix --load-run -1 --render-mode record
```

Use `--render-mode record` for headless video-only playback. Leave backend
selection in `--sim motrix` rather than overriding `training.sim_backend` by
itself.
