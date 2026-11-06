# Hydra Config

UniLab uses Hydra composition with task owner YAMLs. The owner YAML is the
identity of the task, backend, reward, scene, and task-specific runtime fields.

## Owner Paths

| Stack | Owner YAML Shape |
| --- | --- |
| PPO | `src/unilab/conf/ppo/task/<task>/<backend>.yaml` |
| APPO | `src/unilab/conf/appo/task/<task>/<backend>.yaml` |
| SAC / FlashSAC | `src/unilab/conf/<algo>/task/<task>/<backend>.yaml` |

Examples:

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco
uv run train --algo ppo --task go2_joystick_flat --sim mujoco
uv run train --algo sac --task g1_walk_flat --sim mujoco
```

For off-policy, `--algo` selects the per-algorithm config tree `src/unilab/conf/<algo>/`;
do not include the algorithm name in `--task`.

## Safe Overrides

Hydra overrides can tune fields inside the selected owner path:

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco \
  algo.max_iterations=10 \
  algo.num_envs=128 \
  training.no_play=true
```

Common fields:

- `algo.max_iterations`
- `algo.num_envs`
- `algo.load_run`
- `algo.seed`
- `training.no_play`
- `training.tf32_matmul` (PPO and APPO owner configs only)
- `training.play_only`
- `training.play_render_mode`
- `training.logger`

`training.tf32_matmul` controls the backend used for float32 matrix multiplies
on the learner. `ieee` (default) keeps full-precision FP32; `tf32` opts into
TF32 tensor-core kernels and automatically falls back to IEEE on macOS/MPS,
ROCm/HIP, pre-Ampere CUDA GPUs (compute capability below 7.0), and CPU. The
resolved value and fallback reason are recorded in `run_config.json` and
`run_summary.json`. The knob exists only on the PPO and APPO owner configs
because the win is on their FP32 update path; SAC-family learners already cast
their main matmuls through bf16 AMP, where #2477 measured TF32 as neutral.

## Inspect the Composed Config

To debug composition, append `--cfg job` to print the fully composed config
without running training:

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco --cfg job
```

## Backend Identity

`training.sim_backend` is an identity field set by the selected owner YAML. It
is not an independent backend switch. Use the unified CLI `--sim` flag to
select the backend.

See the developer contract in {doc}`../../4-developer_guide/2-contracts/3-task_owner`.

## Per-Algorithm Reference

Field-by-field tables — training efficiency, policy performance (GAE lambda, entropy
coefficient, learning rate, network shape), training runtime, and reward term weights —
live on each algorithm page:

- {doc}`../2-algorithms/1-ppo`
- {doc}`../2-algorithms/2-appo`
- {doc}`../2-algorithms/3-sac`
- {doc}`../2-algorithms/4-flash_sac`
