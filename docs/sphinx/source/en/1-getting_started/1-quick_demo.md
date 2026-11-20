# Quick Demo

This page mirrors the README Quick Demo and uses the first-level UniLab CLI:
`uv run train`, `uv run eval`, and `uv run demo`.

## Clone And Sync

```bash
# 0. If uv is not installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# 1. Clone the repository
git clone https://github.com/unilabsim/UniLab.git
cd UniLab
```

Choose exactly one dependency setup command for your platform:

```bash
# Linux CUDA or macOS
make setup

# Without shell completion setup:
# uv sync --extra mujoco

# If make is not installed:
# uv sync --extra mujoco && uv run --no-sync unilab-complete install

# Linux AMD / ROCm
# make sync-rocm

# Linux Intel Arc / iGPU
# make sync-xpu
```

## First Success

Run a pre-trained policy before changing any task configuration:

```bash
# Fetches the checkpoint and assets from Hugging Face on first run.
uv run demo dance
```

Available demo names are `teaser` and `dance`. Use `uv run demo --help` for
device and refresh options.

## Train A Task

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco
```

This command routes to the registered `go2_joystick_flat` task with the MuJoCo
backend. The CLI keeps algorithm, task, and backend selection explicit through
`--algo`, `--task`, and `--sim`; internally it composes the matching owner YAML.

## Evaluate And Replay

```bash
uv run eval --algo ppo --task go2_joystick_flat --sim mujoco --load-run -1

# Headless video export for Linux/server runs
uv run eval --algo ppo --task go2_joystick_flat --sim mujoco \
  --load-run -1 --render-mode record

```

Mainland China users: motions, scenes, robot meshes, and demo checkpoints come
from Hugging Face on first run. If `huggingface.co` is unreachable, switch to the
community mirror before running training, eval, or demo commands:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

On macOS, the CLI routes Motrix interactive playback through `mxpython` when
needed. Use `--render-mode record` for headless video export or
`--render-mode none` to skip rendering.

## Short Smoke Variant

For CI-style local checks, keep the same CLI route and add Hydra overrides after
the flags:

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco \
  algo.max_iterations=1 \
  algo.num_envs=16 \
  training.no_play=true
```

## Next Steps

- Platform details: {doc}`2-installation`.
- Replay details: {doc}`3-evaluation_and_playback`.
- Learn the unified CLI in {doc}`../2-user_guide/1-training/1-cli_reference`.
- Read how Hydra owner YAMLs work in {doc}`../2-user_guide/1-training/2-hydra_config`.
