# Simulation Backends

The tensor-only Manager runtime currently exposes `mujoco`, `mjwarp`,
`genesis`, and `newton`. User commands select one with `--sim`, which routes to
the matching task owner YAML; do not switch a run by overriding
`training.sim_backend` alone.

The `motrix`, `drake`, `isaacgym`, `isaacsim`, and `superdex` adapters remain
temporarily shelved by `unisim-core` during issue #1811. Their historical pages
are retained for adapter context only and are not production support claims.
The train/eval CLI rejects these names until new capability, parity, and
support-matrix evidence is provided.

## Runtime Prerequisites

- Any run using `--sim mujoco`, MuJoCo playback, or a MuJoCo-only debugging
  tool requires the `mujoco` extra and its pinned `mjbatch` runtime.
- MJWarp requires the `mjwarp` extra, NVIDIA CUDA, and on multi-GPU hosts an
  explicit process-device topology.
- Genesis requires the `genesis` extra and the validated Linux x86_64 GPU path.
- Newton requires the `newton` extra and an NVIDIA CUDA device; the current
  tensor-native support scope is SAC `g1_walk_flat` and FlashSAC
  `g1_motion_tracking`.

## OS and GPU Support

| Backend | Operating system | GPU |
| --- | --- | --- |
| MuJoCo | Linux / macOS / Windows | Not required: CPU physics; offline playback can render on CPU |
| MJWarp | Linux (validated path) | Required: NVIDIA CUDA; a single-GPU host uses the current CUDA device by default |
| Genesis | Linux x86_64 | Required: NVIDIA GPU and driver; only the `gs.gpu` channel is validated |
| Newton | Linux | Required: NVIDIA CUDA; the selected-reset lane is device-resident |

Backend device requirements are independent of the learner device: MuJoCo can
still train with its learner on CUDA, ROCm, MPS, or XPU. See the platform
profiles in {doc}`../../1-getting_started/2-installation`.

## Select A Backend

UniLab selects the simulator through the task owner config. For normal usage,
choose the task and backend with `--task` and `--sim`; off-policy commands keep
the algorithm in `--algo`, not in `--task`.

### Quick Choice

| Need | Prefer |
| --- | --- |
| Default path or broadest owner coverage | MuJoCo |
| MuJoCo-only tools such as `scripts/play_viser.py` | MuJoCo |
| Device-resident tensor owner | MJWarp, Genesis, or Newton, when the task support matrix marks the combination supported |

The support matrix is generated from registry, owner YAML, and tests; use it as
the current evidence source: {doc}`../../5-reference/5-support_matrix`.

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco
uv run train --algo sac --task g1_motion_tracking --sim mjwarp
```

Owner YAML locations:

- PPO / APPO: `src/unilab/conf/{ppo,appo}/task/<task>/<backend>.yaml`
- Off-policy (SAC / FlashSAC): `src/unilab/conf/<algo>/task/<task>/<backend>.yaml`

The selected owner YAML sets `training.sim_backend` as an identity field.

## Playback Differences

- `--render-mode auto` exports `play_video.mp4` on MuJoCo paths.
- `--render-mode record` records without opening an interactive window.
- `--render-mode viser` serves the rollout in a browser-based viser viewer on
  backends with physics-state playback (MuJoCo and MJWarp).
- `--render-mode none` disables playback.

```bash
uv run eval --algo ppo --task go2_joystick_flat --sim mujoco --load-run -1
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

`unisim` has no dependency on UniLab, Hydra, or training components. The scoped
MuJoCo, MJWarp, Genesis, and Newton adapters and the temporarily shelved
adapters use one public contract. Missing proprietary SDKs or GPU workers
produce an explicit cold-path diagnostic; no backend silently falls back to
another engine.

```{toctree}
:hidden:

1-mujoco
5-genesis
7-newton
```
