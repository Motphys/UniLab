---
orphan: true
---

# ADR-0015 No Numba Runtime And Explicit NumPy Leaf Boundaries

语言: 简体中文

- Status: Accepted
- Date: 2026-10-10
- Owners: Env / Manager / Training / Backend maintainers
- Supersedes: None
- Superseded by: None
- Roadmap: [Issue #2124](https://github.com/Motphys/UniLab/issues/2124)

## Context

ADR-0011 removed the NumPy environment lifecycle and ADR-0012 made the Manager
runtime tensor-only. Two migration-era remnants nevertheless remain in UniLab:

- G1 motion tracking still contains CPU/NumPy command kernels implemented with
  Numba. Importing the tensor motion terms also imports and configures those
  kernels, so even a tensor owner is not Numba-free.
- Generic Manager, MDP, reset/event, entity-facade, and locomotion task paths
  still expose NumPy carriers and hidden Torch/NumPy conversions.

The current `unisim-core` base wheel has no Numba implementation or dependency.
Its `body_state` helper is intentionally NumPy-only. The only remaining source
of a Numba installation, after UniLab removes its direct requirement, is the
optional Genesis engine dependency. That transitive package dependency does not
make Numba part of UniLab's or UniSim's runtime contract.

Repository evidence at decision time:

- Numba production imports: `src/unilab/tasks/motion_tracking/common/kernels.py`
  and `src/unilab/base/cpu_runtime.py`.
- Direct Numba requirements: `pyproject.toml` and `pyproject.rocm.toml`.
- NumPy is imported by 62 production Python files across Manager, env, task,
  utility, terrain, training, and visualization owners.
- ADR-0012 requires Torch carriers for the Manager public lifecycle and allows
  NumPy only behind a host-bridge backend adapter boundary.

## Decision

### UniLab has no Numba runtime

UniLab production code must not import, compile, configure, tune, or
transitively require Numba. In particular:

- delete the G1 motion-tracking Numba kernels and their CPU command owner;
- all G1 motion tracking production owners build the tensor command;
- `EnvCfg.cpu_ids` owns Linux process and existing-thread affinity only;
- no generic runtime component calls `numba.set_num_threads` or interprets
  `NUMBA_NUM_THREADS`.

The default and ROCm dependency profiles remove UniLab's direct Numba
requirement. A Genesis installation may still transitively contain Numba, but
UniLab neither imports it nor takes responsibility for tuning its engine-private
thread pools.

### The Manager training runtime is Torch-authoritative

Public and internal Manager execution carriers are Torch tensors: state,
observations, rewards, terminations, commands, events, metrics, recorder state,
and row selectors. Generic NumPy execution, dual-carrier term branches, hidden
host fallbacks, and the environment-owned NumPy RNG stream are removed without a
compatibility layer. `TorchManagerRng`, seeded from the resolved owner seed, is
the sole Manager RNG. Legacy NumPy RNG bitstream equality is not a compatibility
requirement.

### Legacy NumPy backend APIs are not UniLab runtime APIs

UniLab task, Manager, env, and training code must not consume legacy NumPy
`SimBackend` methods such as NumPy stepping/reset, base/body getters, or named
sensor data as normal training hot paths. CPU-authoritative exchange is
negotiated through public tensor lifecycle operations and packed host-bridge
plans. The complete public replacement and removal of UniSim legacy NumPy APIs
is an upstream contract change; UniLab consumes that replacement without a
dual-path compatibility layer.

### NumPy is allowed only in explicit leaf owners

Outside the concrete CPU backend adapter, NumPy may appear only in:

1. cold asset/materialization work, such as NPZ motion ingestion and terrain
   generation;
2. serialization and external interop, such as ONNX Runtime and diagnostic NPZ;
3. playback/rendering/visualization host formats.

Those leaves own their imports and conversions internally. They must not return
mutable NumPy authoritative carriers to the Manager runtime or training hot
path. Cold-path NumPy is not authorization for step/reset parsing, asset
metadata branching, terrain regeneration, or host fallback dispatch.

## Stable Contracts

| Contract | Evidence owner |
| --- | --- |
| Production source contains no Numba import or tuning | architecture boundary tests |
| Manager execution, terms, RNG, and row selectors are Torch-only | Manager/env suites |
| G1 motion production owners build the tensor command | owner config tests |
| `EnvCfg.cpu_ids` confines process/existing threads without engine-private pool tuning | CPU runtime tests |
| Default and ROCm profiles have no direct UniLab Numba requirement | dependency-profile tests and locks |
| UniLab runtime does not call legacy NumPy backend APIs | architecture/backend contract tests |
| Remaining NumPy imports appear only in the explicit leaf allowlist | architecture boundary tests |
| Manager seed/reset reproducibility is Torch-RNG based | reproducibility tests |

## Alternatives Considered

- Keep the CPU/NumPy motion command as a hidden fallback. Rejected: it preserves
  the NpEnv-era dual contract and forces every tensor owner to import Numba.
- Let `EnvCfg.cpu_ids` tune any engine-private pool installed by an optional
  extra. Rejected: generic runtime code would depend on an unrelated engine
  implementation and could not define a stable failure policy.
- Allow NumPy throughout task and Manager code while merely deleting Numba.
  Rejected: hidden host conversions and dual term branches would continue to
  spread and violate ADR-0012.
- Preserve a compatibility layer for legacy NumPy backend APIs. Rejected by the
  roadmap decision: the replacement is a deliberate breaking boundary.

## Consequences

- CPU `MotionCommand` and its Numba kernels disappear without aliases.
- Existing NumPy-stream runs are not bit-compatible with Torch-RNG runs.
- Backend work that still lacks a public packed tensor replacement fails closed
  rather than falling back to a NumPy path.
- Genesis is not considered production runtime support merely because its extra
  transitively installs Numba.
- Review rejects new Manager/task NumPy imports, `.numpy()` hot-path
  conversions, and `torch.from_numpy` compatibility bridges outside the
  allowlisted leaf owners.

## Evidence In Repo

- `src/unilab/tasks/motion_tracking/common/kernels.py`
- `src/unilab/tasks/motion_tracking/common/manager_terms.py`
- `src/unilab/base/cpu_runtime.py`
- `src/unilab/envs/manager_based_rl_env.py`
- `src/unilab/managers/`
- `pyproject.toml`
- `pyproject.rocm.toml`
- `tests/tasks/test_package_boundary.py`

## Related Documents

- [ADR-0011 Torch-Only Manager-Based Runtime](ADR-0011-torch-only-manager-based-runtime.md)
- [ADR-0012 Sole Tensor Manager And Scoped Backends](ADR-0012-sole-tensor-manager-and-scoped-backends.md)
- [Issue #2124](https://github.com/Motphys/UniLab/issues/2124)
- {doc}`ADR Index </adr/ADR-0000-index>`
