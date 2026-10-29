# CLI Reference

UniLab exposes package commands for common training and playback routes, and
keeps the lower-level scripts available for debugging Hydra composition.

## Unified Commands

| Goal | Command Shape | Routed Script |
| --- | --- | --- |
| PPO | `uv run train --algo ppo --task <task> --sim <backend>` | `src/unilab/scripts/train_rsl_rl.py` |
| APPO | `uv run train --algo appo --task <task> --sim <backend>` | `src/unilab/scripts/train_appo.py` |
| SAC | `uv run train --algo sac --task <task> --sim <backend>` | `src/unilab/scripts/train_sac.py` |
| FlashSAC | `uv run train --algo flashsac --task <task> --sim <backend>` | `src/unilab/scripts/train_flashsac.py` |

Examples:

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco
uv run train --algo appo --task g1_motion_tracking --sim mujoco training.no_play=true
uv run train --algo sac --task g1_walk_flat --sim mujoco training.no_play=true
uv run train --algo flashsac --task go2_joystick_flat --sim mujoco
```

The CLI builds the owner YAML path from `--algo`, `--task`, `--sim`, and
optional `--profile`. Route-defining values must use the CLI flags; Hydra
overrides after the command are for fields such as `algo.max_iterations`,
`algo.num_envs`, and `training.no_play`.

## Interrupting Training

Send `Ctrl+C` once to request cooperative shutdown. Training entrypoints,
collectors, sibling data-parallel ranks, and torch elastic workers unwind their
existing cleanup paths before the process exits with the conventional status
`130`. Send `Ctrl+C` a second time to force the CLI child to terminate; forced
termination does not guarantee graceful cleanup. Normal shutdown notices from
torch elastic may still appear, but a user cancellation is not reported as a
training failure.

The `uni-cumps` command owns the explicit user-space daemon lifecycle for
CUDA MPS. It reports read-only host status and topology diagnostics, starts and
stops only UniLab-recorded user-owned daemons, and prints launcher environment
values without mutating the caller's shell:

```bash
# Single-GPU host; all defaults are complete.
uv run uni-cumps start
eval "$(uv run uni-cumps env)"
uv run uni-cumps doctor
uv run uni-cumps stop

# Multi-GPU or explicitly named deployments.
uv run uni-cumps doctor --gpus <gpu-index-or-uuid>
uv run uni-cumps start --gpus <gpu-index-or-uuid> --name <name>
uv run uni-cumps env --name <name>
uv run uni-cumps stop --name <name>
```

See the tensor runtime production guide for ownership, concurrency, and topology
limits. Training still requests MPS explicitly with
`training.cuda_process_sharing=mps`.

### Per-Environment Invocation

The CLI prefix depends on the install flavor:

- ROCm: run `make sync-rocm` once, then use `uv run ...`.
- Intel XPU: use `uv run --no-sync ...`.

## Tab Completion

Shell completion is optional. It only completes the `uv run train` / `uv run eval`
entrypoints, their flags, and some choices; it never changes command behavior.
On a fresh checkout, one setup command syncs the environment and installs the
completion:

```bash
# Full default environment:
make setup
```

`make setup` syncs the default environment (`uv sync --extra mujoco`, with
the built-in RL stack) followed by
`uv run --no-sync unilab-complete install`.
Choose one setup path for an environment. The install command picks Bash or Zsh
from `$SHELL` / platform and only writes user-level rc files. The current shell
is not auto-activated; reopen the terminal or source the rc file to apply.

If `make` is unavailable, run the steps directly:

```bash
uv sync --extra mujoco
uv run --no-sync unilab-complete install
```

Bash users (Linux / WSL) can instead add this to `~/.bashrc`:

```bash
source scripts/completions/unilab.bash
```

Zsh users (default on macOS) can add this to `~/.zshrc`:

```zsh
autoload -Uz compinit
compinit
source scripts/completions/unilab.zsh
```

After reopening the terminal or sourcing the rc file, candidates appear for
`uv run <TAB>`, `uv run train --algo <TAB>`, and `uv run train --sim <TAB>`.

## Evaluation

`uv run eval` sets `training.play_only=true` and optionally maps `--load-run` to
`algo.load_run`.

```bash
uv run eval --algo ppo --task go2_joystick_flat --sim mujoco --load-run -1
uv run eval --algo ppo --task go2_joystick_flat --sim mujoco --load-run -1 \
  --render-mode record
```

Supported render modes are `auto`, `interactive`, `record`, `viser`, and
`none`.

The MuJoCo interactive mode (`--sim mujoco --render-mode interactive`) routes
directly to `play_interactive.py` and always rolls out one environment.

For backends with physics-state playback (`mujoco`, `mjwarp`, `newton`,
`drake`, `superdex`), `--render-mode viser` serves the rollout through the
browser-based viser viewer: no local display is required, and macOS runs it
without `mjpython`. Eval routes to the standalone `play_viser.py` viewer;
train runs the same viewer as its post-training playback. viser is a required
dependency; the `viser.*` config group (`viser.port`, `viser.max_envs`,
`viser.display_mode`, `viser.env_idx`) tunes the server and the scene.

## Demo

```bash
uv run demo dance
uv run demo dance --refresh --device cpu
```

Available demos: `teaser`, `dance`.
Each demo fetches a pre-trained checkpoint from the
`unilabsim/unilab-checkpoints` Hugging Face dataset on first run and caches it
under `src/unilab/assets/checkpoints/<demo>/model_0.pt`. Pass `--refresh` to
re-download.

Mainland China users: when `huggingface.co` is unreachable, switch to the
community mirror before running the demo:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

The demo entrypoint is implemented by `src/unilab/demo.py` and routed from
`src/unilab/cli.py`.

## Low-Level Scripts

The lower-level scripts remain available when you need to inspect Hydra config
groups or reproduce a script-level issue. For normal usage, keep
route-defining values in the unified CLI flags above.

For off-policy routes, `--algo` selects the per-algorithm owner tree
`src/unilab/conf/<algo>/`; do not include the algorithm name in `--task`.

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
