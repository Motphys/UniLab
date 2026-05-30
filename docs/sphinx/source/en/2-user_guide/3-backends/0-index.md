# Simulation Backends

UniLab currently uses two backend names in registry/config paths: `1-mujoco` and
`2-motrix`. User commands select them with `--sim`, which routes to the matching
task owner YAML; do not switch a run by overriding `training.sim_backend` alone.

## Runtime Prerequisites

- Install Motrix support with `uv sync --extra motrix`.
- Any run using `--sim mujoco`, MuJoCo playback, or MuJoCo-only debugging tool
  still requires a working MuJoCo runtime.
- On macOS, the package CLI routes Motrix interactive playback through
  `mxpython` when needed. Direct script calls that open the native Motrix
  renderer should use `uv run mxpython`.

## Select A Backend

```bash
uv run train --algo ppo --task go1_joystick_flat --sim mujoco
uv run train --algo ppo --task go1_joystick_flat --sim motrix
```

Owner YAML locations:

- PPO / APPO: `conf/{ppo,appo}/task/<task>/<backend>.yaml`
- Off-policy: `conf/offpolicy/task/<algo>/<task>/<backend>.yaml`

The selected owner YAML sets `training.sim_backend` as an identity field.

## Playback Differences

- `--render-mode auto` exports `play_video.mp4` on MuJoCo paths.
- `--render-mode auto` opens Motrix native interactive rendering on Motrix
  paths.
- `--render-mode record` records without opening an interactive window.
- `--render-mode none` disables playback.

```bash
uv run eval --algo ppo --task go1_joystick_flat --sim mujoco --load-run -1
uv run eval --algo ppo --task go1_joystick_flat --sim motrix --load-run -1 \
  --render-mode record
```

## Support Evidence

Task/backend/entrypoint support is evidence-graded. See
{doc}`../../5-reference/5-support_matrix` for the support matrix entry and links to
the generated source data.

## Related Contracts

- {doc}`Backend contract </en/4-developer_guide/2-contracts/2-backend_contract>`
- {doc}`Task owner contract </en/4-developer_guide/2-contracts/3-task_owner>`
- {doc}`Backend capability boundary ADR </adr/ADR-0002-backend-capability-boundary-for-play-and-snapshot>`
- {doc}`Registry bootstrap ADR </adr/ADR-0004-registry-bootstrap-contract>`

```{toctree}
:hidden:

1-mujoco
2-motrix
3-choosing_a_backend
```
