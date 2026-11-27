# Motrix Backend

## Motrix Warp (CUDA-resident MotrixSim)

Use `--sim mtxwarp` with the `mtxwarp` extra to run MotrixSim physics and
Manager tensors on the same NVIDIA CUDA device. The profile uses MotrixSim
1.0.0 with Warp (warp-lang >= 1.15, resolved to 1.17.0 by the CUDA runtime)
on Linux x86_64, and is available from `unisim-core` 2.9.0.

```bash
uv sync --locked --extra mtxwarp
uv run train --algo sac --task g1_walk_flat --sim mtxwarp
```

The only owner today is SAC `g1_walk_flat`: it declares the per-world contact
and constraint capacities (256 contacts and 400 constraint rows) and a higher
solver iteration budget than the CPU owner (`env.motrix_max_iterations: 10`)
for contact stability. The owner keeps eager selected reset so host curriculum
updates stay live; set `env.autoreset_cuda_graph=false` to pin the eager path
explicitly when debugging. See the UniSim support matrix for engine capability
boundaries; `mtxwarp` and `mjwarp` install into the same environment.

## MotrixSim CPU

Motrix is an optional backend installed through the `motrix` extra, which
delegates to `unisim-core[motrix]`: the runtime pin lives in UniSim's
`pyproject.toml`, and the adapter lives under `unisim.backend.motrix`.

Motrix is a CPU-authoritative packed HOST_BRIDGE backend. Issue #2054 restored
it to the tensor-only Manager runtime with these canonical training workloads:

```bash
uv run --extra motrix train --algo appo --task go2_joystick_flat --sim motrix
uv run --extra motrix train --algo sac --task g1_walk_flat --sim motrix
uv run --extra motrix train --algo flashsac --task g1_motion_tracking --sim motrix
```

Other Motrix owners remain out of scope and do not constitute production
support claims. The Go2 APPO owner is restored as a configured training path;
its current evidence is bounded smoke training, not a performance benchmark.

The public tensor lifecycle exposes:

- persistent packed control D2H;
- persistent packed selected-reset D2H;
- persistent packed full state/sensor H2D;
- persistent packed selected-row reset publication H2D.

Each phase is countable in the transfer plan's diagnostic counters. There is no
hidden task-side NumPy reset composer, and unsupported reset randomization and
fixed variants fail closed.

## Setup

```bash
uv sync --extra motrix
```

`make setup` runs the same dependency sync and installs shell completion.

## When To Use It

- The workload is one of the canonical owners above.
- You need a CPU-authoritative packed HOST_BRIDGE comparison/reference path.
- The generated support matrix marks the entrypoint/task/backend combination as
  configured or tested.

## Commands

Use `--render-mode record` for headless video-only playback. Leave backend
selection in `--sim motrix` rather than overriding `training.sim_backend` by
itself.

## Performance boundary

MotrixSim's native CPU physics is slower than mjbatch and is not required to
match it. Issue #2054's measured canonical control cycle showed Motrix at about
53% of MuJoCo/mjbatch, with explicit D2H/H2D totaling roughly 0.1 ms and native
`step_n` accounting for nearly all of the remaining gap. Avoidable selected-read
full-batch work was removed in unisim #340.

## CPU Affinity

`training.dp_collector_cpu_ids` may provide one explicit CPU-id list per
collector rank. The block reaches the env as `EnvCfg.cpu_ids` and is
validated on the cold path: entries must be non-empty, unique, non-negative
CPU ids available to the owning process. The Motrix adapter then pins
MotrixSim's shared worker pool before the first model load — worker `i`
takes `cpu_ids[i % len(cpu_ids)]` — and env construction also confines the
process itself to the same block, so host-side post-step compute stays
inside the rank's partition. `null` (the default) leaves MotrixSim's default
thread-pool policy and OS scheduling untouched. The validated block is
exposed through the backend's read-only `cpu_ids` property.
