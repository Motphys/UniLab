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
uv run demo dance
uv run demo wallflip
uv run demo boxtracking
uv run demo locomani
uv run demo inhandgrasp
uv run demo dance --refresh --device cpu
```

Available demos: `dance`, `wallflip`, `boxtracking`, `locomani`, `inhandgrasp`.
Each demo fetches a pre-trained checkpoint from the
`unilabsim/unilab-checkpoints` Hugging Face dataset on first run and caches it
under `src/unilab/assets/checkpoints/<demo>/model_0.pt`. Pass `--refresh` to
re-download.

The demo entrypoint is implemented by `src/unilab/demo.py` and routed from
`src/unilab/cli.py`.

## Low-Level Scripts

The lower-level scripts remain available when you need to inspect Hydra config
groups or reproduce a script-level issue. For normal usage, keep
route-defining values in the unified CLI flags above.

For off-policy routes, keep `--algo` aligned with the owner tree under
`conf/offpolicy/task/<algo>/`; do not include the algorithm name in `--task`.

## Common Overrides

```bash
training.no_play=true
algo.max_iterations=10
algo.num_envs=128
algo.load_run=-1
training.logger=wandb
```

Use `uv run eval ... --load-run ...` for playback and `--render-mode record`
or `--render-mode none` for render behavior.

Backend selection belongs to the CLI `--sim` choice and the resulting owner
YAML. Do not use `training.sim_backend=<backend>` as a standalone switch.
