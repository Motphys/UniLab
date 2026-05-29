# Simulation Backends

UniLab currently uses two backend names in registry/config paths: `mujoco` and
`motrix`. Backend selection is made through the task owner YAML, not by
overriding `training.sim_backend` alone.

## Runtime Prerequisites

- Install Motrix support with `uv sync --extra motrix`.
- Any `task=.../mujoco` run, MuJoCo playback, or MuJoCo-only debugging tool
  still requires a working MuJoCo runtime.
- On macOS, the package CLI routes Motrix interactive playback through
  `mxpython` when needed. Direct script calls that open the native Motrix
  renderer should use `uv run mxpython`.

## Select A Backend

```bash
uv run scripts/train_rsl_rl.py task=go1_joystick_flat/mujoco
uv run scripts/train_rsl_rl.py task=go1_joystick_flat/motrix
uv run train --algo ppo --task go1_joystick_flat --sim motrix
```

Owner YAML locations:

- PPO / APPO: `conf/{ppo,appo}/task/<task>/<backend>.yaml`
- Off-policy: `conf/offpolicy/task/<algo>/<task>/<backend>.yaml`

The selected owner YAML sets `training.sim_backend` as an identity field.

## Playback Differences

- `training.play_render_mode=auto` exports `play_video.mp4` on MuJoCo paths.
- `training.play_render_mode=auto` opens Motrix native interactive rendering
  on Motrix paths.
- `training.play_render_mode=record` records without opening an interactive
  window.
- `training.play_render_mode=none` disables playback.

```bash
uv run eval --algo ppo --task go1_joystick_flat --sim mujoco --load-run -1
uv run eval --algo ppo --task go1_joystick_flat --sim motrix --load-run -1 \
  --render-mode record
```

## Support Evidence

Task/backend/entrypoint support is evidence-graded. See
{doc}`../../reference/support_matrix` for the support matrix entry and links to
the generated source data.

## Related Contracts

- {doc}`Backend contract </en/developer_guide/contracts/backend_contract>`
- {doc}`Task owner contract </en/developer_guide/contracts/task_owner>`
- {doc}`Backend capability boundary ADR </adr/ADR-0002-backend-capability-boundary-for-play-and-snapshot>`
- {doc}`Registry bootstrap ADR </adr/ADR-0004-registry-bootstrap-contract>`
