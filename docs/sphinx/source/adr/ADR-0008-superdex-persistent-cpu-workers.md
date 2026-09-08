---
orphan: true
---

# ADR-0008 SuperDex Native C++ Scene Batch Executor

- Status: Accepted
- Date: 2026-09-07
- Owners: SuperDex fork / UniSim backend / UniLab task-config maintainers
- Supersedes: None
- Superseded by: None

## Context

The initial SuperDex adapter creates one independent native scene for every
environment but advances them serially from Python. `Scene.step()` releases the
GIL, yet a Python thread pool would still leave generalized-force writes and
articulated state reads as separate Python-to-C++ calls. A subprocess/shared
memory design was considered and rejected before implementation because it
would introduce a new runtime protocol beyond the requested integration scope.

The maintained fork [unilabsim/project_superdex](https://github.com/unilabsim/project_superdex)
can build an extension against the exact Physics and Robotics sources. This
removes the wheel/header ABI mismatch that prevents a downstream native shim.
The roadmap is [UniLab#1533](https://github.com/Motphys/UniLab/issues/1533).

## Decision

1. Add `superdex.physics.SceneBatchExecutor` to the fork's `mochi_physics`
   pybind extension. It owns persistent C++ worker threads, but does not own
   scenes or actors.
2. One executor invocation receives contiguous generalized-force, articulated
   pose/velocity, link-state, contact and solver-status arrays. It writes each
   actor's forces, advances each distinct scene, and refreshes every runtime
   cache before its completion barrier opens.
3. UniSim owns the executor and creates it after cold-path materialization. It
   keeps reset, asset conversion, cache-frame conversion and `SimBackend`
   ownership in the adapter. `close()` joins the executor
   before destroying bots, scenes, and the process-global runtime.
4. `superdex_num_workers=0` selects `min(CPU affinity, num_envs)`. Explicit
   worker counts are capped at `num_envs`; the SDK stays single-threaded so the
   executor is the only physics parallelism layer.

## Consequences

The local integration requires Physics and Robotics bindings built from this
fork at the same commit. Older qpos/qvel-only executor builds are rejected. It
neither changes package versions nor publishes a wheel. The executor provides
CPU scene parallelism; it does not claim GPU physics, native rendering, or
dynamics equivalence with MuJoCo.

Validation must compare serial and parallel trajectories, selected reset, and
complete backend throughput using the same scene, actions, batch, and substep
count. Training evidence must report the actual outer worker count and SDK
thread count separately.

## Alternatives Considered

- Python `ThreadPoolExecutor`: useful for a narrow step probe, but does not
  fuse force/state binding calls or give the adapter a durable native barrier.
- Subprocess IPC and shared memory: rejected because SuperDex does not require
  process isolation and the protocol is outside this roadmap's approved scope.
- Linking a downstream extension to the released wheel: rejected because the
  wheel does not publish a stable extension ABI or matching headers.

## Related Documents

- {doc}`SuperDex Backend </en/2-user_guide/3-backends/8-superdex>`
- {doc}`UniSim Extraction Boundary </adr/ADR-0007-unisim-extraction-boundary>`
- [UniLab#1533](https://github.com/Motphys/UniLab/issues/1533)
