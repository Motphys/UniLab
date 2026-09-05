# Simulation Backends

UniLab exposes backend names through registry/config paths, including `mujoco`,
`motrix`, `mjwarp`, `drake`, `isaacgym`, `genesis`, `isaacsim`, and `newton`
where an owner is registered.
User commands select them with `--sim`, which routes to the matching task owner
YAML; do not switch a run by overriding `training.sim_backend` alone.

## Runtime Prerequisites

- Install Motrix support with `uv sync --extra motrix`.
- IsaacGym and IsaacSim use dedicated external worker runtimes; see their
  backend pages for installation and runtime requirements.
- Any run using `--sim mujoco`, MuJoCo playback, or MuJoCo-only debugging tool
  still requires a working MuJoCo runtime.
- Drake uses the external `drake-uni` package plus a locally built C++ batch
  extension; see {doc}`6-drake` before selecting `--sim drake`.
- Newton uses the isolated `newton` extra (`uv sync --extra newton`), which
  cannot share an environment with the `mujoco` / `mjwarp` extras; see
  {doc}`7-newton` before selecting `--sim newton`.
- On macOS, the package CLI routes Motrix interactive playback through
  `mxpython` when needed. Direct script calls that open the native Motrix
  renderer should use `uv run mxpython`.

## Select A Backend

UniLab selects the simulator through the task owner config. For normal usage,
choose the task and backend with `--task` and `--sim`; off-policy commands keep
the algorithm in `--algo`, not in `--task`. Do not switch a run by overriding
`training.sim_backend` alone; that field is set by the owner YAML and identifies
the composed backend.

### Quick Choice

| Need | Prefer |
| --- | --- |
| Default path or broadest owner coverage | MuJoCo |
| Native interactive playback through the backend | Motrix |
| MuJoCo-only tools such as `scripts/play_viser.py` | MuJoCo |
| Task owner exists only under `src/unilab/conf/.../<task>/mujoco.yaml` | MuJoCo |
| Task owner exists under `src/unilab/conf/.../<task>/motrix.yaml` and the support matrix marks the combination as tested or configured | Motrix |

The support matrix is generated from registry, owner YAML, and tests; use it as
the current evidence source: {doc}`../../5-reference/5-support_matrix`.

```bash
uv run train --algo ppo --task go1_joystick_flat --sim mujoco
uv run train --algo ppo --task go1_joystick_flat --sim motrix
uv run train --algo ppo --task g1_walk_flat --sim isaacsim
```

More combinations:

```bash
uv run train --algo ppo --task stewart_balance --sim drake \
  algo.max_iterations=1 algo.num_envs=8 training.no_play=true
uv run train --algo sac --task g1_walk_flat --sim mujoco
```

Owner YAML locations:

- PPO / APPO: `src/unilab/conf/{ppo,appo}/task/<task>/<backend>.yaml`
- Off-policy (SAC / TD3 / FlashSAC): `src/unilab/conf/<algo>/task/<task>/<backend>.yaml`

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

## The unisim-core boundary

UniLab's physics backends are provided by the independent `unisim-core`
distribution, with `unisim` as the Python namespace. For example:

```bash
uv sync --extra mujoco
uv run python -c "import unisim; print(unisim.ADAPTER_SPECS)"
```

`unisim` has no dependency on UniLab, Hydra, or training components. MuJoCo,
Motrix, Drake, MJWarp, Genesis, IsaacGym, IsaacSim, and Newton use one public
contract.
Missing proprietary SDKs or GPU workers produce an explicit cold-path diagnostic;
no backend silently falls back to another engine.

Backend physics is owned exclusively by `unisim-core`. UniLab keeps only the
owner-layer assembly entry point `unilab.base.backend_factory`; contracts and
adapters are imported from `unisim`. The former `unilab.base.backend`
implementation and compatibility layer have been removed; do not add backend
APIs to UniLab.

Benchmark v1 reserves only `BenchmarkCase`, `BenchmarkResult`, and provenance
schema. Workloads, timing, comparisons, and performance claims require a
separately authorized issue.

```{toctree}
:hidden:

1-mujoco
2-motrix
3-isaacgym
4-isaacsim
5-genesis
6-drake
7-newton
```
