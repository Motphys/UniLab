# Support Matrix

Each generated block is assembled from registry entries, owner-YAML backend
identity, validation inventories, and UniSim's static platform profiles. The
generator implementation is `scripts/tools/support_matrix.py`. Do not infer
support beyond the evidence grade shown below.

## Backend Selection Rules

- The tensor-only Manager runtime currently supports `mujoco`, `mjwarp`,
  `genesis`, `newton`, and `motrix`.
- The default backend is `mujoco`.
- `--sim mjwarp` requires the `mjwarp` extra. Validated combinations are shown
  in the generated matrix below; all other entrypoints retain their matrix
  evidence grade.
- `--sim genesis` requires the `genesis` extra; real CUDA/platform requirements
  are shown in the generated matrix.
- `--sim newton` requires the `newton` extra; real CUDA/platform requirements
  are shown in the generated matrix.
- `--sim motrix` requires the `motrix` extra; the CPU-authoritative packed
  HOST_BRIDGE profile is shown in the generated matrix.
- `--algo`, `--task`, and `--sim` jointly select the owner YAML.
- Do not treat `training.sim_backend` as a standalone backend switch.
- `drake`, `isaacgym`, `isaacsim`, and `superdex` are temporarily outside this
  runtime scope. Their adapters remain in UniSim, but they are not UniLab
  production support claims; re-enabling requires capability, parity, and
  support-matrix evidence (#1811).

## Playback Differences

- `mujoco`: `--render-mode auto` exports `play_video.mp4`; `--render-mode
  viser` serves the rollout in a browser-based viser viewer.
- `mjwarp`: supports explicit, finite-step `record` by default, rendered offline
  through the task owner's MuJoCo visual model; `--render-mode interactive`
  routes to the MuJoCo interactive viewer (mjwarp runs the physics while
  MuJoCo renders env[0], forced to a single env); `--render-mode viser`
  routes to the browser-based viser viewer with per-env MuJoCo playback
  models; `auto` and native renderers are not supported.
- `genesis`: `viser` is unsupported, as is the MuJoCo interactive/offline
  playback contract described here.
- `newton`: the canonical owners use `record` through native Newton rendering,
  with the MuJoCo snapshot as the incomplete-install fallback.
- `--render-mode record`: MuJoCo and mjwarp record a video only.
- `--render-mode none`: no playback.

## Support Matrix

The matrix below is generated from registry entries, owner YAMLs, validation
inventories, and UniSim platform profiles. Do not edit its tables by hand.
Refresh it with:

```bash
uv run scripts/generate_support_matrix.py --write
```

<!-- BEGIN GENERATED SUPPORT MATRIX -->
### Evidence Grades

| Grade | Repository evidence |
|---|---|
| `Registered` | The env/backend exists in `registry.list_registered_envs()` after `ensure_registries()`. |
| `Configured` | The owner YAML sets `training.sim_backend` to the backend. |
| `Tested` | Automated coverage or explicit maintainer validation; it is not a default recommendation. |
| `Benchmarked` | A checked-in benchmark manifest is bound to the combination. |
| `Recommended` | Explicit recommendation metadata exists in the repository. |

`Tested` describes repository evidence only; it does not imply every DR, rendering, or production capability of the MuJoCo owner. No benchmark or recommendation metadata is currently checked in, so rows do not auto-promote to `Benchmarked` or `Recommended`.

### Tensor Backend Platform Matrix

This table is derived from UniSim's SDK-free public static inventory. It describes reviewed tensor-lifecycle boundaries, not optional-SDK installation and not automatic task-owner support.

| Backend | Execution / process / data plane | Torch devices | CUDA runtime | Linux+CUDA | macOS | ROCm | Worker | Reset randomization | Fixed variants | Host callbacks | Packed bridge |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `mujoco` | Host bridge / in-process / host bridge | CPU / CUDA | Required only when the learner requests CUDA state/control buffers | Supported: CPU-authoritative physics with optional CUDA Torch buffers | CPU-authoritative host bridge only; no CUDA physics claim | CPU-authoritative host bridge only; no ROCm CUDA-only fallback | In-process; no external Python worker | unknown | unknown | Unsupported | Exact |
| `motrix` | Host bridge / in-process / host bridge | CPU / CUDA | Required only when the learner requests CUDA state/control buffers | Supported: CPU-authoritative physics with optional CUDA Torch buffers | CPU-authoritative host bridge only; no CUDA physics claim | CPU-authoritative host bridge only; no ROCm CUDA-only fallback | In-process; no external Python worker | Unsupported | Unsupported | Unsupported | Exact |
| `mjwarp` | Device-resident / in-process / direct | CUDA | Required for the entire tensor lifecycle | Supported: Linux CUDA only | Unsupported; no CPU, MPS, or ROCm fallback | Unsupported; no CPU, MPS, or ROCm fallback | In-process; no external Python worker | Unsupported | Unsupported | Unsupported | Unsupported |
| `newton` | Device-resident / in-process / direct | CUDA | Required for the entire tensor lifecycle | Supported: Linux CUDA only | Unsupported; no CPU, MPS, or ROCm fallback | Unsupported; no CPU, MPS, or ROCm fallback | In-process; no external Python worker | Unsupported | Unsupported | Unsupported | Unsupported |
| `genesis` | Device-resident / in-process / direct | CUDA | Required for the entire tensor lifecycle | Supported: Linux CUDA only | Unsupported; no CPU, MPS, or ROCm fallback | Unsupported; no CPU, MPS, or ROCm fallback | In-process; no external Python worker | Unsupported | Unsupported | Unsupported | Unsupported |

### Entrypoint x Task Owner

| Entrypoint | Task owner | MuJoCo | Motrix | mjwarp | Newton | Genesis |
|------------|------------|---|---|---|---|---|
| PPO (torch) | `go2_joystick_flat` (Go2 joystick) | Tested | - | - | - | - |
| PPO (torch) | `g1_walk_flat` (G1 walk flat) | Tested | Registered | Tested | Registered | Configured |
| PPO (torch) | `g1_motion_tracking` (G1 motion tracking) | Tested | - | - | - | - |
| PPO (torch) | `g1_flip_tracking` (G1 flip tracking) | Tested | - | - | - | - |
| PPO (torch) | `x2_wall_flip_tracking` (X2 wall flip tracking) | Tested | - | - | - | - |
| PPO (torch) | `allegro_inhand` (Allegro in-hand) | Tested | - | - | - | - |
| PPO (torch) | `allegro_inhand_grasp` (allegro inhand grasp) | Tested | - | - | - | - |
| PPO (torch) | `g1_box_tracking` (g1 box tracking) | Tested | - | - | - | - |
| PPO (torch) | `stewart_balance` (stewart balance) | Tested | - | - | - | - |
| APPO (torch) | `go2_joystick_flat` (Go2 joystick) | Tested | - | - | - | - |
| APPO (torch) | `g1_walk_flat` (G1 walk flat) | Tested | Registered | Registered | Registered | Registered |
| APPO (torch) | `g1_motion_tracking` (G1 motion tracking) | Tested | - | - | - | - |
| APPO (torch) | `g1_flip_tracking` (G1 flip tracking) | Tested | - | - | - | - |
| APPO (torch) | `allegro_inhand` (Allegro in-hand) | Tested | - | - | - | - |
| SAC (torch) | `g1_walk_flat` (G1 walk flat) | Tested | Tested | Tested | Tested | Tested |
| SAC (torch) | `g1_motion_tracking` (G1 motion tracking) | Tested | Registered | Configured | Registered | Configured |
| SAC (torch) | `g1_flip_tracking` (G1 flip tracking) | Tested | - | Configured | - | - |
| SAC (torch) | `g1_wbt_obs` (g1 wbt obs) | Tested | - | - | - | - |
| FlashSAC (torch) | `go2_joystick_flat` (Go2 joystick) | Tested | - | - | - | - |
| FlashSAC (torch) | `g1_walk_flat` (G1 walk flat) | Tested | Registered | Configured | Registered | Registered |
| FlashSAC (torch) | `g1_motion_tracking` (G1 motion tracking) | Tested | Tested | Configured | Configured | Configured |
| WarpSAC (torch) | `g1_walk_flat` (G1 walk flat) | Tested | Registered | Tested | Registered | Registered |
| WarpSAC (torch) | `g1_motion_tracking` (G1 motion tracking) | Tested | Registered | Tested | Registered | Registered |

### Source Index

- Registry bootstrap: `src/unilab/envs/**` decorators via `unilab.base.registry.ensure_registries()`.
- Owner backend identity: `training.sim_backend` in `src/unilab/conf/{ppo,appo,sac,flashsac,warpsac}/task/**`.
- Platform/capability source: `unisim.support.get_tensor_platform_profiles()`.
- Unsupported platform/device requests are guarded before backend construction in `src/unilab/base/backend_factory.py`.
- Generic compose coverage: `tests/config/test_config_system.py::test_supported_task_composes`.
<!-- END GENERATED SUPPORT MATRIX -->

## Platform Troubleshooting

CUDA-only tensor backends are rejected before backend construction on CPU, MPS,
macOS, ROCm/HIP, unavailable CUDA, malformed CUDA ordinals, and out-of-range
device requests. They do not fall back to NumPy or a host bridge.

Check the Torch runtime and visible ordinals in the same process environment:

```bash
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.version.hip, torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.current_device())"
```

`torch.version.hip` must be `None` for the CUDA-only matrix. If CUDA is
unavailable, check the NVIDIA driver/container mismatch before changing task
configuration. If `CUDA_VISIBLE_DEVICES` is set, backend ordinals address that
remapped namespace, not host-global physical indices.

On macOS and ROCm, use a CPU-authoritative host-bridge backend. A CUDA Torch
buffer on a host-bridge backend is not a CUDA physics or device-resident
lifecycle claim.
