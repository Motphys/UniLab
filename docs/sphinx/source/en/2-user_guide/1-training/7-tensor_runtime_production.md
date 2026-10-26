# Tensor Runtime Production Guide

This guide is the reproduction and operations entry point for the scoped
single-GPU G1 Motion Tracking / FlashSAC / MJWarp tensor runtime. It covers the
canonical owner-routed long soak, runtime artifacts, and failure triage. The
settings contract is described separately in {doc}`4-tensor_runtime`; the
generated {doc}`../../5-reference/5-support_matrix` is the authority for every
backend, task, entrypoint, and platform support claim.

The production path is deliberately narrow:

- Linux with one NVIDIA CUDA GPU;
- FlashSAC;
- `g1_motion_tracking`;
- the `mjwarp` owner selected through the unified task-owner route;
- a tensor-native Manager runtime.

It is not a multi-GPU guide, does not publish packages, and does not generalize
to every MJWarp task. macOS and ROCm/HIP do not provide a fallback for this
device-resident lifecycle; invalid requests fail before backend construction.

## Owner Selection and Effective Combinations

Use `--algo`, `--task`, and `--sim` together. They compose the owner YAML under
`src/unilab/conf/<algo>/task/<task>/<sim>.yaml`. That owner, rather than the CLI
spelling alone, sets `training.sim_backend`; tensor execution is a Manager invariant.

For the canonical combination:

```bash
uv run train --algo flashsac --task g1_motion_tracking --sim mjwarp
```

the MJWarp owner inherits the MuJoCo FlashSAC owner and resolves to:

| Effective setting | Value |
| --- | --- |
| Task semantic name | `G1MotionTracking` |
| Backend identity | `mjwarp` |
| Tensor runtime | invariant of the Manager lifecycle |
| Inference slots | `training.inference_slot_capacity=1` |
| Replay-ingress depth | `training.replay_ingress_depth=2` |
| Replay-ingress slot rows | `null`, resolving to `algo.num_envs` |
| Collector metric interval | `training.collector_metrics_interval=100` |
| CUDA process sharing | `training.cuda_process_sharing=null` |
| Scene | `src/unilab/assets/robots/g1/scene_flat.xml` |
| Motion | `motions/g1/dance1_subject2_part.npz` |

The MuJoCo owner uses the same 2048-environment training budget, but publishes
collector metrics every step. The MJWarp task is also a DR-free lifecycle
baseline: its selected-row tensor reset does not negotiate reset randomization
or interval wrenches.

At the current generated evidence grade, FlashSAC /
`g1_motion_tracking` / MJWarp is `Configured`. A successful local soak does not
silently promote it to `Benchmarked` or `Recommended`; support promotion is a
separate repository decision made through the support-matrix generator and its
validation inventory.

## Reproducible Runtime

The gated profile resolves UniSim and MJBatch from the committed
registry lock files (the RL stack ships in-tree under `unilab.rl`):

```bash
UV_FROZEN=1 uv sync --extra mujoco --extra mjwarp
```

`make setup` is useful for broad repository development, but it does not install
the `mjwarp` extra. The command above matches the gated CUDA CI profile. CI
uses Python 3.11 as the reference version even though the package supports a
wider Python range.

On a host with one visible GPU, no `CUDA_VISIBLE_DEVICES` mask is required.
The automatic single-device topology binds the learner, collector, MJWarp
physics, inference ring, and replay ingress to the same current CUDA device.
Set the variable only to select one physical GPU on a multi-GPU host or to
preserve a launcher/debug/MPS mask:

```bash
export CUDA_VISIBLE_DEVICES=<single-host-cuda-ordinal>
```

All CUDA ordinals inside trainer processes are relative to that mask. Do not
add more devices: M11 is single-GPU only.

## CUDA MPS Execution Sharing

`training.cuda_process_sharing` is an explicit execution-sharing mode for the
already-required rank-local topology. It is not a backend switch and does not
replace `--sim mjwarp` or owner YAML selection.

The default remains `null`: learner and collector keep independent CUDA contexts.
The supported initial topology is Linux, NVIDIA CUDA, one physical GPU, MJWarp,
SAC/FlashSAC, and `world_size=1`. Request shared GPU execution with:

```bash
training.cuda_process_sharing=mps
```

### Managed daemon lifecycle

UniLab now provides an explicit user-owned lifecycle command, while training
itself still never starts or stops a host service implicitly:

On a single-GPU host, the defaults are already complete:

```bash
uv run uni-cumps start
uv run uni-cumps doctor
uv run uni-cumps stop
```

After `start`, a trainer request using `training.cuda_process_sharing=mps`
selects the sole live UniLab-recorded daemon covering its GPU. Shell
integration is optional:

```bash
uv run uni-cumps start
uv run --extra mjwarp train --algo flashsac \
  --task g1_motion_tracking --sim mjwarp \
  training.cuda_process_sharing=mps
```

An explicit `CUDA_MPS_PIPE_DIRECTORY` in the caller environment still takes
precedence. If no matching live daemon exists, training fails closed before
environment/learner/collector construction and points to `uni-cumps start`.

Default values:

| Setting | Default |
| --- | --- |
| GPU | the sole visible physical GPU, resolved to its canonical UUID |
| daemon name | `gpu-<full-uuid>` |
| pipe directory | `/tmp/uni-cumps/<name>/pipe` (mode `0700`) |
| log directory | `/tmp/uni-cumps/<name>/log` (mode `0700`) |
| `env`/`stop` target | the sole live UniLab-recorded daemon |

If more than one GPU is visible, `start` and `doctor` require
`--gpus <index-or-uuid>`. Explicit overrides remain available for deployments
that need stable labels or service-owned paths:

```bash
uv run uni-cumps start \
  --gpus <gpu-index-or-uuid> \
  --name <daemon-name> \
  --pipe-dir /absolute/path/mps/pipe \
  --log-dir /absolute/path/mps/log
eval "$(uv run uni-cumps env --name <daemon-name>)"
uv run uni-cumps stop --name <daemon-name>
```

Read-only status works without arguments:

```bash
uv run uni-cumps status
```

`status` and `doctor` are read-only. `start` creates a user-owned daemon and a
record under the current user's UniLab cache. `stop` uses only that recorded
ownership evidence: UID, host, control PID, and Linux process start-time. It
never sends `quit` to an unmanaged pipe. `env` prints exports for a launcher or
an explicit shell integration; it does not mutate UniLab's parent process.

A daemon is not exclusively owned by a training run and may serve multiple
clients. A stale record is quarantined before reuse, and logs are retained after
stop.

### Topology modes and current limits

The command reports three topology-mode names so future launch contracts remain
additive:

| Mode | Meaning | Initial status |
| --- | --- | --- |
| `single_gpu` | one task uses one physical GPU | Implemented |
| `single_task_multi_gpu` | one task has one rank per GPU | Parsed, fail-closed |
| `task_per_gpu` | independent tasks are packed one per GPU | Parsed, fail-closed |

Selectors resolve through canonical physical GPU UUIDs. MIG UUIDs are rejected
explicitly. A comma-separated multi-GPU request fails closed and references the
multi-GPU gate rather than silently choosing DP or task packing. `all` is useful
in read-only diagnostics but cannot start a daemon in this release.

Training still validates before environment probing, learner construction, and
collector spawn; there is no silent multi-context fallback. The error names the
first unmet prerequisite and points to the CLI lifecycle commands.

`run_config.json` records the configured owner value. A valid run's
`run_summary.json` embeds `runtime_manifest.cuda_process_sharing` with the
configured/effective mode, learner and collector devices, physical UUID
evidence, control pipe, server PID, and validation state. An unconfigured
(`null`) run still records ambient host state on CUDA ranks — the raw
`CUDA_MPS_PIPE_DIRECTORY` value, the effective pipe directory and control-pipe
kind, the answering server PID, and best-effort learner/collector GPU UUIDs —
with `ambient_server=true` when an environment-level daemon answered at the
effective control pipe, so a transparent-MPS run is identifiable from its own
manifest. Ambient observation never fails closed; unresolvable facts record
`null`. The section is a runtime-manifest v1 producer diagnostic, not a
stable scalar contract.

MPS changes only GPU execution sharing. It does not change
`env_steps_per_sync`, learner/collector placement, or training semantics.
Multi-GPU DP remains unsupported until its separate gate and daemon topology
decision are completed. CUDA MPS remains host- and deployment-dependent: in
shared containers, multi-user hosts, or restricted runners, control may be
unavailable, and an explicit request fails closed rather than silently
degrading.

Restricted deployments that cannot use the UniLab CLI may still operate an
existing daemon manually with `nvidia-cuda-mps-control`; training validation
does not depend on which explicit deployment tool started it.

Check the runtime that will execute the benchmark:

```bash
uv run --no-sync python -c "import torch; print(torch.__version__, torch.version.cuda, torch.version.hip, torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.current_device())"
```

`torch.version.hip` must be `None`, CUDA must be available, and the visible
device count must be one.

## Backend CUDA Graph Diagnostics

Training artifacts record the physics backend's tensor runtime diagnostics.
`TorchEnv.get_backend_runtime_diagnostics()` (`src/unilab/base/torch_env.py`)
is the collection boundary: a public cold-path method that serializes the
backend's `SimBackend.get_tensor_runtime_diagnostics()` (the unisim public
contract) into plain JSON. The off-policy collector
(`unilab.rl.runtime.offpolicy`) calls it at startup
and publishes the payload into `runtime_manifest.backend_diagnostics`; the PPO
training script (`src/unilab/scripts/train_rsl_rl.py`) writes the same payload
as a top-level `backend_diagnostics` key in `run_summary.json` because PPO has
no runtime manifest.

Payload shape:

```json
"backend_diagnostics": {
  "available": true,
  "backend_type": "mjwarp",
  "diagnostics": {
    "cuda_graph": {"requested": true, "enabled": false, "disable_reason": "..."},
    "selected_reset_sensor_refresh": {"requested": true, "enabled": true, "disable_reason": null},
    "device_reset_randomization": {"requested": true, "enabled": true, "disable_reason": null}
  }
}
```

- `requested`: the optimization was requested (MJWarp always requests CUDA
  graph).
- `enabled`: the optimization is actually active (for example, fixed-address
  CUDA graph replay).
- `disable_reason`: the backend-owned fallback cause; always present and
  non-empty when `enabled` is false, always `null` when true.

When the env or backend exposes no diagnostics, the payload is an explicit
unavailable marker, never a bare `null`:

```json
"backend_diagnostics": {"available": false, "backend_type": "...", "unavailable_reason": "..."}
```

Like `runtime_manifest.cuda_process_sharing`, this section is a
runtime-manifest v1 producer diagnostic, not a stable scalar contract; the
runtime manifest schema (`unilab.rl.logging.runtime_manifest_schema`)
validates the section fail-closed when it is
present. The diagnostics are observability only: they do not change CUDA graph
enablement policy and are not an acceptance gate. They are a cold-path snapshot
taken at collector/env startup; they describe initialization, not per-step
behavior.

| Observation | First action |
| --- | --- |
| `enabled: false` for an optimization | Read `disable_reason` for the backend-owned fallback cause (graph capture failure, unsupported topology, and so on); the run is still valid, it just steps eagerly. |
| `available: false` | Check `unavailable_reason`: either the backend declares no tensor runtime diagnostics (expected on non-tensor backends such as CPU MuJoCo) or the diagnostic probe raised (the exception type is included). |

Known limitation: unisim does not yet expose the CUDA graph version or
eager-fallback event counts. The serializer forwards future dataclass fields
automatically, so the section grows when unisim adds them.

## Assets

G1 meshes and textures are downloaded from the asset hub. Pre-fetch them before
an offline benchmark:

```bash
uv run --no-sync unilab-pull-assets --robot g1
```

The motion clip is also external. It is downloaded lazily on first use and must
be present at:

```text
src/unilab/assets/motions/g1/dance1_subject2_part.npz
```

For a completely offline run, first materialize the required motion files, then
set `HF_HUB_OFFLINE=1`. Do not set offline mode before the first
materialization.

## Canonical Long Soak

The long soak is the canonical production benchmark. It launches the real
owner-routed FlashSAC trainer, then monitors schemas, progress, CUDA state,
processes, file descriptors, shared memory, inference flight, replay ingress,
and shutdown.

Choose new absolute paths for the run directory and JSON artifact. Do not reuse
a dirty run directory: the monitor uses file changes in that directory as its
liveness signal.

```bash
uv run --no-sync scripts/benchmark/torch_env/g1_flashsac_soak.py \
  --num-envs 1024 \
  --iterations 65000 \
  --min-duration-seconds 1800 \
  --sample-interval-seconds 5 \
  --startup-timeout-seconds 900 \
  --stale-progress-seconds 300 \
  --post-shutdown-grace-seconds 10 \
  --extra-override algo.save_interval=10000 \
  --log-dir /absolute/path/to/g1-flashsac-mjwarp-run \
  --output /absolute/path/to/g1-flashsac-mjwarp-soak.json
```

The wrapper owns and rejects overrides for `algo`, `task`,
`training.sim_backend`, play mode, log directory, environment count, and
iteration count. It constructs the low-level owner route for
`flashsac` / `g1_motion_tracking` / `mjwarp`; users do not pass
`training.sim_backend=mjwarp` as an independent switch.

The M11 acceptance budget is intentionally distinct from the owner production
default:

| Budget | Environments | Iterations | Use |
| --- | ---: | ---: | --- |
| MJWarp owner default | 2048 | 25000 | Owner training budget |
| M11 long soak | 1024 | 65000 | Lifecycle, resource, replay, and shutdown acceptance |

The wrapper also sets `training.no_play=true`, so a passing soak does not create
playback video or playback-driven ONNX output.

## Required Artifacts

The run directory must contain:

- `run_config.json`;
- `run_summary.json`;
- TensorBoard event files;
- checkpoints at the selected `algo.save_interval`;
- progress files consumed by the soak monitor.

`run_summary.json` must carry metric schema `1` and an embedded runtime-manifest
schema `1`. The manifest records effective runtime limits, inference and tensor
memory budgets when a budget decision is made, the collector/backend device,
inference-flight state, replay-ingress state, shutdown diagnostics, and the
backend tensor runtime diagnostics (`runtime_manifest.backend_diagnostics`; PPO
runs carry the same payload as a top-level `backend_diagnostics`). See
{doc}`3-logging` for the field and compatibility contract.

The soak JSON is self-contained monitor evidence with artifact schema `0.3.0`.
A current-head pass requires:

| Evidence | Passing value |
| --- | --- |
| Monitor status | `passed` |
| Trainer return code | `0` |
| Run status | `completed` |
| Failure reason and classification | both `null` |
| Final inference queue depth and publication lag | both `0` |
| Replay final occupancy | `0` |
| Replay dropped batches and early returns | both `0` |
| Replay release sequence | equal to published sequence |
| Published sequence | `total_env_steps / ingress_slot_rows` |
| Shutdown classification | `normal_completion` |
| Shutdown cleanup errors | `[]` |
| Post-shutdown residual processes | `[]` |

`resource_summary` also reports peak process count, RSS, GPU memory, eventfds,
shared-memory FDs, replay high-water occupancy, inference event count, and
maximum in-flight requests. Peaks are expected; look for unbounded or sustained
growth across post-startup samples, not merely a nonzero maximum.

The artifact `context` records the UniLab commit and dirty state, GPU
UUID/name/driver, Python/Torch metadata, and `CUDA_VISIBLE_DEVICES`. Preserve
the JSON and run directory together. The repository intentionally does not
track these machine-generated artifacts.

Extract the last 20 schema-v1 TensorBoard windows after the run:

```bash
uv run --no-sync scripts/benchmark/rl/extract_offpolicy_metrics.py \
  /absolute/path/to/g1-flashsac-mjwarp-run \
  --last 20 \
  --json
```

The helper fails closed on a missing or unsupported metric schema. Historical
unversioned event files require the explicit `--legacy-unversioned` mode and
are never an implicit fallback.

## Historical Soak Evidence

The M11 lifecycle review reached a passing v6 long soak on a local RTX 5090:
65,000 iterations, 66,660,352 environment steps, 1987.55 seconds, no dropped
replay batch, final replay occupancy zero, final inference queue/lag zero, a
normal shutdown, and no residual process. Resource sampling showed bounded
process, RSS, GPU-memory, eventfd, and shared-memory-FD behavior.

That v6 artifact is historical evidence, not a current-head benchmark claim. It
used soak artifact schema `0.2.0` and older UniLab, unilab-rl, and UniSim
commits before current schema-v1 consumer enforcement. Reproduce from the exact
checked-out head with the command above when current-head evidence is required.

## Optional Phase-Local Probe

`g1_flashsac_backend.py` is a diagnostic probe, not the canonical end-to-end
benchmark. It deliberately excludes inference IPC, replay ingestion, learner
updates, and production Manager-Based dispatch.

```bash
uv run --no-sync scripts/benchmark/torch_env/g1_flashsac_backend.py \
  --backends mjwarp,mujoco \
  --num-envs 2048 \
  --warmup 20 \
  --iters 100 \
  --acceptance \
  --output scripts/benchmark/outputs/g1-flashsac-backend/current-host.json
```

Acceptance mode requires available `nvidia-smi` evidence and an idle GPU before
and after each isolated backend process. It also rejects Isaac worker profiler
environment variables.

The JSON schema is `0.3.0`. Use
`throughput_env_control_steps_per_s` and synchronized `phases.iteration_ms` for
cross-backend comparison. Individual phase boundaries may be stream ordered, so
do not add phases into a new wall-clock total or treat one backend's
`update_state_ms` as directly comparable to another backend family's phase.
MuJoCo results also include the packed H2D/D2H inventory and transfer counters.

## Troubleshooting

| Symptom | First action |
| --- | --- |
| MJWarp import fails | Re-run the explicit `uv sync --extra mujoco --extra mjwarp`; `make setup` alone is insufficient for this profile. |
| CUDA unavailable | Check the driver/container runtime and the Torch sanity command; do not change the task owner to force a fallback. |
| Wrong physical GPU | Set `CUDA_VISIBLE_DEVICES` in the shell that launches `uv run`; backend ordinal `0` is relative to that mask. |
| CUDA IPC or worker mismatch | Keep learner, collector, and payload on the same physical GPU. External Isaac workers inherit visibility but not host Python paths. |
| Asset startup failure | Pre-fetch G1 assets, verify the motion NPZ exists, and remove `HF_HUB_OFFLINE=1` until first download completes. |
| Startup timeout | Inspect `soak-console.log` and the first monitor sample; distinguish asset download, dependency installation, backend startup, and process failure. |
| Stale progress | Open the latest run files and console log; no file change for `stale-progress-seconds` intentionally fails rather than waiting forever. |
| Replay backpressure or drops | Read occupancy, high-water, waits, early returns, and dropped batches; do not shrink ingress slot rows without a workload benchmark. |
| Nonzero final replay occupancy | Keep the artifact and console log; this is a lifecycle/finalize failure even if the trainer otherwise appeared to finish. |
| Eventfd/shared-memory growth | Compare post-startup samples; bounded peaks are normal, sustained growth is a leak signal. |
| Abnormal shutdown | Use `failure_classification` and `shutdown.classification` to identify learner, collector, backend worker, stale tick, cancellation, or unknown failure. |
| Residual process | Keep the artifact, which records known PIDs and start ticks; do not reuse the GPU until those processes are explained and gone. |
| Unsupported capability | Read the exact fail-closed error and generated support matrix; there is no hidden NumPy or host-bridge fallback for a CUDA-only lifecycle. |

For full schema definitions and metric interpretation, see {doc}`3-logging` and
{doc}`4-tensor_runtime`.

## Breaking Tensor Runtime Migration

`env.tensor_runtime` and `env.tensor_runtime_device` were removed without a
compatibility alias. The Manager runtime is tensor-native by construction:
placement is derived from the backend's declared data plane and rank process
device. Owner YAMLs and runner overrides that previously set either field must
delete those keys. Unsupported tensor lifecycles fail during binding rather
than falling back to a NumPy wire.
