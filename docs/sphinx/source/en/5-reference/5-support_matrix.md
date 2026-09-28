# Support Matrix

Each generated block is assembled from registry entries, owner-YAML backend
identity, validation inventories, and UniSim's static platform profiles. The
generator implementation is `scripts/tools/support_matrix.py`. Do not infer
support beyond the evidence grade shown below.

## Backend Selection Rules

- The default backend is `mujoco`.
- Switch to Motrix with `--sim motrix` on the unified CLI.
- Switch to IsaacSim with `--sim isaacsim` on the unified CLI. IsaacSim uses
  an external Python 3.11 worker; its owner scope is currently limited to the
  configured G1 walk-flat PPO/SAC paths.
- `--sim mjwarp` requires the `mjwarp` extra. Validated combinations are shown
  in the generated matrix below; all other entrypoints retain their matrix
  evidence grade.
- `--algo`, `--task`, and `--sim` jointly select the owner YAML.
- Do not treat `training.sim_backend` as a standalone backend switch.

## Playback Differences

- `mujoco`: `--render-mode auto` exports `play_video.mp4`; `--render-mode
  viser` serves the rollout in a browser-based viser viewer.
- `motrix`: `--render-mode auto` opens an interactive renderer window; it does
  not record a video and is not bound by `play_steps`. `--render-mode viser`
  routes to the browser-based viser viewer with per-env MuJoCo playback
  models driven by physics-state snapshots.
- `mjwarp`: supports explicit, finite-step `record` by default, rendered offline
  through the task owner's MuJoCo visual model; `--render-mode interactive`
  routes to the MuJoCo interactive viewer (mjwarp runs the physics while
  MuJoCo renders env[0], forced to a single env); `--render-mode viser`
  routes to the browser-based viser viewer with per-env MuJoCo playback
  models; `auto` and native renderers are not supported.
- `isaacsim`: `auto` selects the Kit viewer when a display is available and
  otherwise selects headless RGB capture. The current real host still has an
  RTX renderer initialization blocker, so owner support remains `Configured`.
- `--render-mode record`: MuJoCo, mjwarp, and Motrix record a video only.
  IsaacSim routes to its offline RGB protocol, whose real-host playback claim
  remains `Configured`.
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
| `drake` | Host bridge / in-process / host bridge | CPU / CUDA | Required only when the learner requests CUDA state/control buffers | Supported: CPU-authoritative physics with optional CUDA Torch buffers | CPU-authoritative host bridge only; no CUDA physics claim | CPU-authoritative host bridge only; no ROCm CUDA-only fallback | In-process; no external Python worker | Unsupported | Unsupported | Unsupported | Exact |
| `mjwarp` | Device-resident / in-process / direct | CUDA | Required for the entire tensor lifecycle | Supported: Linux CUDA only | Unsupported; no CPU, MPS, or ROCm fallback | Unsupported; no CPU, MPS, or ROCm fallback | In-process; no external Python worker | Unsupported | Unsupported | Unsupported | Unsupported |
| `newton` | Device-resident / in-process / direct | CUDA | Required for the entire tensor lifecycle | Supported: Linux CUDA only | Unsupported; no CPU, MPS, or ROCm fallback | Unsupported; no CPU, MPS, or ROCm fallback | In-process; no external Python worker | Unsupported | Unsupported | Unsupported | Unsupported |
| `superdex` | Host bridge / in-process / host bridge | CPU / CUDA | Required only when the learner requests CUDA state/control buffers | Supported: CPU-authoritative physics with optional CUDA Torch buffers | CPU-authoritative host bridge only; no CUDA physics claim | CPU-authoritative host bridge only; no ROCm CUDA-only fallback | In-process; no external Python worker | Unsupported | Unsupported | Unsupported | Exact |
| `genesis` | Device-resident / in-process / direct | CUDA | Required for the entire tensor lifecycle | Supported: Linux CUDA only | Unsupported; no CPU, MPS, or ROCm fallback | Unsupported; no CPU, MPS, or ROCm fallback | In-process; no external Python worker | Unsupported | Unsupported | Unsupported | Unsupported |
| `isaacgym` | Device-resident / external worker / cuda ipc | CUDA | Required for the entire tensor lifecycle | Supported: Linux CUDA only | Unsupported; no CPU, MPS, or ROCm fallback | Unsupported; no CPU, MPS, or ROCm fallback | Dedicated external Python 3.8 worker; host Python paths are not inherited | Unsupported | Unsupported | Unsupported | Unsupported |
| `isaacsim` | Device-resident / external worker / cuda ipc | CUDA | Required for the entire tensor lifecycle | Supported: Linux CUDA only | Unsupported; no CPU, MPS, or ROCm fallback | Unsupported; no CPU, MPS, or ROCm fallback | Dedicated external Python 3.11 worker; host Python paths are not inherited | Unsupported | Unsupported | Unsupported | Unsupported |

### Entrypoint x Task Owner

| Entrypoint | Task owner | MuJoCo | Motrix | Drake | mjwarp | Newton | SuperDex | Genesis | IsaacGym | IsaacSim |
|------------|------------|---|---|---|---|---|---|---|---|---|
| PPO (torch) | `go2_joystick_flat` (Go2 joystick) | Tested | Tested | Tested | - | - | Configured | - | - | - |
| PPO (torch) | `g1_walk_flat` (G1 walk flat) | Tested | Tested | - | Tested | Configured | - | Configured | Configured | Configured |
| PPO (torch) | `g1_motion_tracking` (G1 motion tracking) | Tested | Tested | - | - | - | - | - | - | - |
| PPO (torch) | `g1_flip_tracking` (G1 flip tracking) | Tested | Tested | - | - | - | - | - | - | - |
| PPO (torch) | `x2_wall_flip_tracking` (X2 wall flip tracking) | Tested | Tested | - | - | - | - | - | - | - |
| PPO (torch) | `allegro_inhand` (Allegro in-hand) | Tested | Tested | Tested | - | - | - | - | - | - |
| PPO (torch) | `allegro_inhand_grasp` (allegro inhand grasp) | Tested | Tested | - | - | - | - | - | - | - |
| PPO (torch) | `fr3_joint_target` (fr3 joint target) | - | - | - | - | - | Configured | - | - | - |
| PPO (torch) | `g1_box_tracking` (g1 box tracking) | Tested | Tested | - | - | - | - | - | - | - |
| PPO (torch) | `stewart_balance` (stewart balance) | Tested | Tested | Tested | - | - | - | - | - | - |
| APPO (torch) | `go2_joystick_flat` (Go2 joystick) | Tested | Tested | Registered | - | - | Registered | - | - | - |
| APPO (torch) | `g1_walk_flat` (G1 walk flat) | Tested | Registered | - | Registered | Registered | - | Registered | Registered | Registered |
| APPO (torch) | `g1_motion_tracking` (G1 motion tracking) | Tested | Tested | - | - | - | - | - | - | - |
| APPO (torch) | `g1_flip_tracking` (G1 flip tracking) | Tested | Tested | - | - | - | - | - | - | - |
| APPO (torch) | `allegro_inhand` (Allegro in-hand) | Tested | Tested | Tested | - | - | - | - | - | - |
| SAC (torch) | `go2_joystick_flat` (Go2 joystick) | Registered | Registered | Tested | - | - | Registered | - | - | - |
| SAC (torch) | `g1_walk_flat` (G1 walk flat) | Tested | Tested | - | Tested | Tested | - | Tested | Tested | Configured |
| SAC (torch) | `g1_motion_tracking` (G1 motion tracking) | Tested | Tested | - | Configured | Configured | - | Configured | Configured | Configured |
| SAC (torch) | `g1_flip_tracking` (G1 flip tracking) | Tested | Registered | - | Configured | - | - | - | - | - |
| SAC (torch) | `g1_wbt_obs` (g1 wbt obs) | Tested | Registered | - | - | - | - | - | - | - |
| SAC (torch) | `stewart_balance` (stewart balance) | Registered | Registered | Tested | - | - | - | - | - | - |
| FlashSAC (torch) | `go2_joystick_flat` (Go2 joystick) | Tested | Registered | Registered | - | - | Registered | - | - | - |
| FlashSAC (torch) | `g1_walk_flat` (G1 walk flat) | Tested | Tested | - | Configured | Registered | - | Registered | Registered | Registered |
| FlashSAC (torch) | `g1_motion_tracking` (G1 motion tracking) | Tested | Tested | - | Configured | Configured | - | Configured | Registered | Registered |
| WarpSAC (torch) | `g1_walk_flat` (G1 walk flat) | Tested | Registered | - | Tested | Registered | - | Registered | Registered | Registered |
| WarpSAC (torch) | `g1_motion_tracking` (G1 motion tracking) | Tested | Registered | - | Tested | Registered | - | Registered | Registered | Registered |

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

IsaacGym and IsaacSim use their dedicated Python 3.8 and Python 3.11 workers.
They inherit the CUDA visibility namespace but not host `PYTHONPATH` or
`PYTHONHOME`. A mismatch between the learner's current Torch CUDA ordinal and
the integer Isaac payload ordinal fails before construction; same-physical-GPU
CUDA IPC is checked again by the worker handshake.

On macOS and ROCm, use a CPU-authoritative host-bridge backend. A CUDA Torch
buffer on a host-bridge backend is not a CUDA physics or device-resident
lifecycle claim.
