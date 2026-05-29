# CLI Reference

UniLab exposes package commands for common training and playback routes, and
keeps the lower-level scripts available for debugging Hydra composition.

## Unified Commands

| Goal | Command Shape | Routed Script |
| --- | --- | --- |
| PPO | `uv run train --algo ppo --task <task> --sim <backend>` | `scripts/train_rsl_rl.py` |
| MLX PPO | `uv run train --algo mlx_ppo --task <task> --sim <backend>` | `scripts/train_mlx_ppo.py` |
| APPO | `uv run train --algo appo --task <task> --sim <backend>` | `scripts/train_appo.py` |
| SAC | `uv run train --algo sac --task <task> --sim <backend>` | `scripts/train_offpolicy.py` |
| TD3 | `uv run train --algo td3 --task <task> --sim <backend>` | `scripts/train_offpolicy.py` |
| FlashSAC | `uv run train --algo flashsac --task <task> --sim <backend>` | `scripts/train_offpolicy.py` |

Examples:

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco
uv run train --algo appo --task g1_motion_tracking --sim motrix training.no_play=true
uv run train --algo sac --task g1_walk_flat --sim mujoco training.no_play=true
uv run train --algo flashsac --task go2_joystick_flat --sim mujoco
```

The CLI builds the owner YAML path from `--algo`, `--task`, `--sim`, and
optional `--profile`. Route-defining values must use the CLI flags; Hydra
overrides after the command are for fields such as `algo.max_iterations`,
`algo.num_envs`, and `training.no_play`.

## Evaluation

`uv run eval` sets `training.play_only=true` and optionally maps `--load-run` to
`algo.load_run`.

```bash
uv run eval --algo ppo --task go2_joystick_flat --sim mujoco --load-run -1
uv run eval --algo ppo --task go2_joystick_flat --sim motrix --load-run -1 \
  --render-mode record
```

Supported render modes are `auto`, `interactive`, `record`, and `none`.

## Demo

```bash
uv run demo
uv run demo --preset go2_joystick_mujoco_ppo
uv run demo --refresh --device cpu
```

The demo entrypoint is implemented by `src/unilab/demo.py` and routed from
`src/unilab/cli.py`.

## Low-Level Scripts

Use direct scripts when you need to inspect Hydra config groups or reproduce a
script-level issue:

```bash
uv run scripts/train_rsl_rl.py task=go2_joystick_flat/mujoco
uv run scripts/train_mlx_ppo.py task=go2_joystick_flat/mujoco
uv run scripts/train_appo.py task=g1_motion_tracking/mujoco
uv run scripts/train_offpolicy.py algo=sac task=sac/g1_walk_flat/mujoco
uv run scripts/train_offpolicy.py algo=td3 task=td3/g1_walk_flat/mujoco
```

For off-policy scripts, keep `algo=<algo>` aligned with
`task=<algo>/<task>/<backend>`.

## Common Overrides

```bash
training.no_play=true
training.play_only=true
training.play_render_mode=record
algo.max_iterations=10
algo.num_envs=128
algo.load_run=-1
training.logger=wandb
```

Backend selection belongs to the task owner path. Do not use
`training.sim_backend=<backend>` as a standalone switch.
