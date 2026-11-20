---
orphan: true
---

# ADR-0011 Torch-Only Manager-Based Runtime

语言: 简体中文

- Status: Superseded
- Date: 2026-09-29
- Owners: Env / Manager / Training / Backend maintainers
- Supersedes: [ADR-0006](ADR-0006-community-manager-api-on-numpy-runtime.md)
- Superseded by: [ADR-0012](ADR-0012-sole-tensor-manager-and-scoped-backends.md)
- Roadmap: [Issue #1811](https://github.com/Motphys/UniLab/issues/1811)

> 2026-10-16 更新：下文中与 `uni_rl` 相关的两条边界（"direct rsl-rl/PPO 无
> `uni_rl` 可运行、多卡启动懒加载委托 `uni_rl.ipc`" 与 "`uni_rl` 不得 import
> UniLab"）已由 [ADR-0016](ADR-0016-rl-reintegration-into-unilab-rl.md) 替代：
> unilab-rl 已并入本仓库的 `unilab.rl`，跨包边界改为 `unilab.rl` 内部 DAG。
> 其余 contract（Torch-only manager runtime、tensor 生命周期）不变。

## Context

M11 made the G1 FlashSAC tensor lifecycle, IPC, diagnostics, and production guide reviewable,
but it remains task-owned. The general `ManagerBasedRlEnv` still inherits `NpEnv`; update-state,
reset-done, manager terms, the RSL-RL adapter, playback, resume, and most registry fixtures still
use NumPy arrays as their authoritative representation.

At the current `develop/tensor-runtime` baseline, the strict search
`rg '\bNpEnv(?:State)?\b' src tests scripts docs/sphinx/source AGENTS.md` reports 151 matches in
53 files. Deleting `src/unilab/base/np_env.py` is therefore not a rename: autoreset, final
observation, NaN guards, training state, playback, scene cleanup, backend timing, and Manager-Based
term execution must move to a new owner contract first.

This decision fixes the final runtime structure and migration invariants. It does not claim that
every task/backend combination supports the tensor lifecycle, and it does not add M10 multi-GPU
scaling, PyPI publication, or a new support-matrix grade.

## Decision

### Sole public runtime

The only public environment state/lifecycle contract is:

```python
@dataclass
class TorchEnvState:
    obs: dict[str, torch.Tensor]
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    info: dict[str, Any]
    final_observation: dict[str, torch.Tensor] | None = None
```

`TorchEnv` is the sole base for the Manager-Based runtime. `ManagerBasedRlEnv` keeps its
Manager-Based semantics and canonical name, but its implementation must derive from `TorchEnv`.
The final state deletes `NpEnv`, `NpEnvState`, compatibility imports, aliases, the
`tensor_runtime=false` fallback, CPU-env proxies, and hidden NumPy conversion.

`step(actions)` accepts a contiguous `torch.Tensor`. `reset(env_indices)` accepts either a
one-dimensional Torch integer tensor or `None`. Hot paths do not accept NumPy inputs and do not
implicitly convert Torch tensors to NumPy. Cold-path configuration materialization, asset loading,
host-only rendering, diagnostic artifacts, and external trainer adapters may have explicit host
boundaries without changing `TorchEnv`'s authoritative internal type.

### State and lifecycle semantics

`TorchEnvState.obs` remains an observation-group dict, not a flat tensor. The required actor group
is still `obs`; the only optional critic-only group is still `critic`. The env owner continues to
publish `obs_groups_spec`, and wrappers and learners continue to size actor/critic paths from it.

One control step follows this canonical order:

1. validate action dtype, shape, device, and finite state;
2. transform actions into a backend control tensor;
3. advance physics through the public tensor API;
4. read declared state and sensor views;
5. compute termination, truncation, reward, step/interval events, commands, metrics, and
   observations on the runtime device;
6. save terminal pre-reset observations for done rows;
7. when autoreset is enabled, reset only done rows and write back post-reset observations; and
8. return the same `TorchEnvState` and timing/info contract.

`reset()` returns `(obs_dict, info_dict)`. Returned observations are copies for the reset rows; a
full reset returns every row. Manual reset, selected-row reset, and autoreset share one
transaction semantic: build and validate the complete selected-row payload before committing it
once. Validation failure leaves the pending reset uncommitted. If a backend fails after committing
has started, the environment must fault rather than expose a partially reset batch as a successful
step; this preserves the existing public backend rule that rollback after the first upload is not
promised. `final_observation` represents the terminal pre-reset transition; after autoreset,
`state.obs` represents the post-reset observation.

Termination, truncation, done, finite-horizon timeout, reward scaling, and episode-counter ordering
keep the Manager-Based semantics established by ADR-0006. Implementations may preallocate buffers
and use device-side accumulators, but removing NumPy must not change transition semantics.

### Device and backend ownership

`TorchEnv` consumes only the public `unisim.backend.base.SimBackend` tensor contract:

- `tensor_execution()`;
- `get_tensor_capabilities()`;
- `get_state_views()`;
- `get_sensor_view()`;
- `step_tensor()`;
- `set_state_tensor()`;
- `compile_host_bridge_io()`; and
- public device, stream, process-topology, and capability metadata.

Tasks, managers, runners, and benchmarks must not probe backend-private attributes or methods.

For a `DEVICE_RESIDENT` backend, physics control, state, reset, and sensor views remain on the
declared exact CUDA device. `TorchEnv` observation/reward/termination/metric compute uses that
same device namespace. Apart from declared bounded row validation, logging publication, playback,
or abnormal diagnostics, the hot path must not contain a hidden host detour.

For a `HOST_BRIDGE` backend, CPU physics remains authoritative. `TorchEnv` may run Manager-Based
post-step/reset compute on CPU or a selected Torch device, but all accelerator exchange must use a
persistent packed backend plan: shapes and layouts are fixed on the cold path, transfer counts and
bytes are measurable, and selected-row payloads transfer once. Per-term, per-field, or per-Python-loop
scattered H2D/D2H transfers are prohibited.

Manager reads are scene-owned rather than term-owned. `EntityScene.compile_tensor_reads()` freezes
the union of entity joint columns, explicit named sensors, and canonical tracked-body sensor names
for one runtime device. The resulting phase plan publishes one packet per normal read phase and one
selected-row packet after reset; terms only slice validated entity views from that packet. This is
the implementation boundary for the invariant that a `HOST_BRIDGE` Manager read phase performs one
packed H2D transfer.

CPU-authoritative backends may explicitly choose a CPU Torch lifecycle on CPU-only platforms.
CUDA-native backends fail closed without the required CUDA runtime/device, on cross-device tensors,
or when a required capability is absent. macOS and ROCm must not silently degrade to NumPy or CPU
physics. Generated support evidence and backend capabilities, not backend names or
`env.tensor_runtime`, determine support.

### Manager execution and unsupported terms

The Manager-Based API remains the sole task runtime. Moving to Torch does not restore a legacy env
factory; it replaces the execution carrier for observations, actions, rewards, terminations,
commands, events, curriculum, recorders, and metrics with a Torch tensor plan.

Task owners declare executable term contracts. Tensor-compatible terms use device buffers. Terms
that still require a host API appear only at an explicit host-bridge boundary and are declared
during capability negotiation. Unsupported terms, layouts, reset randomization, sensors, devices,
and process capabilities fail closed. The final path must not bypass a failure through a `_cpu_env`
proxy, NumPy fallback, or by disabling `tensor_runtime`.

### Process, training, and replay invariants

- `EnvFactory` and registry factories remain pickleable, and rank-local device binding completes
  before backend and tensor initialization.
- Direct rsl-rl/PPO remains importable and runnable without `uni_rl`; only multi-GPU launch may
  lazily delegate to `uni_rl.ipc`.
- `uni_rl` must not import UniLab. Its public env protocol becomes tensor-first: reset indices,
  observations, rewards, terminated/truncated values, and final observations are no longer declared
  as NumPy contracts.
- Collectors, learners, replay ingress, weight publication, and NaN guards consume `TorchEnvState`
  and public process/device metadata; they do not implicitly return to host for IPC.
- Checkpoint/resume preserves training-state payload version 1: the cumulative control-step counter
  remains a non-negative integer. Torch-internal counters may synchronize explicitly at the
  export/import boundary, and manager-derived counters restore from the authoritative counter.

### Timing, finite checks, and cleanup

Stable M11 runtime-manifest, TensorBoard, and timing semantics retain their owners. Removing a
legacy `NpEnv` name from a metric requires a schema decision; file deletion must not make benchmark
or soak evidence uninterpretable.

Finite validation is fail-closed. Actions, controls, reset qpos/qvel, rewards, observations, and
termination inputs are checked in their declared scopes. Implementations avoid per-Python-row
scalar synchronization; a backend may use only its declared bounded row closure.

NaN-guard dumps, playback snapshots, interactive renderers, mocap playback, video export, and
abnormal-shutdown diagnostics are explicit diagnostic/render boundaries and may convert to host
formats, but they are not normal training hot paths. `close()` continues to release recorders,
pre-step callbacks, workers, CUDA IPC objects, shared buffers, and backend resources.

### Migration policy

`NpEnv` and `TorchEnv` may temporarily coexist during staged engineering work; this is not a
durable compatibility contract. Each PR migrates one owner boundary with focused parity tests.
Before legacy deletion, direct PPO, APPO/off-policy, IPC pickling, NaN guards, checkpoints,
playback, selected-row reset, benchmarks, and support evidence must be migrated and validated.

After the final gate, strict `NpEnv`/`NpEnvState` search must return zero matches in source, tests,
scripts, configs, API references, and current documentation. Only the historical record in
superseded ADR-0006 may remain.

## Stable Contracts

| Contract | Owner / planned evidence |
| --- | --- |
| Tensor state shape and dict observations | `src/unilab/base/torch_env.py` (P1), `tests/base/test_torch_env.py` |
| Autoreset, final observation, selected-row reset | `TorchEnv` lifecycle owner; focused lifecycle and partial-reset tests |
| Manager execution and task semantics | `src/unilab/envs/manager_based_rl_env.py`, `tests/envs/test_manager_based_rl_env.py`, task parity suites |
| Backend capability and tensor execution | `unisim.backend.base.SimBackend`, backend contract suites, generated support matrix |
| Host-bridge packed I/O | UniSim `HostBridgeTransferPlan`, transfer-count/byte tests, M6 benchmark successors |
| Registry and spawn construction | `src/unilab/base/env_factory.py`, `tests/base/test_registry.py`, `tests/rl/runtime/ipc/` |
| Direct PPO / RSL-RL boundary | `src/unilab/rl/adapters/rsl_rl/vec_env.py`, `tests/algos/test_rsl_rl_runner.py` |
| Off-policy/APPO/IPC boundary | `unilab_rl` public protocol and process/device tests; no UniLab import |
| NaN guard, checkpoint, playback | `src/unilab/training/`, `src/unilab/visualization/`, focused lifecycle suites |
| Benchmark and timing schema | `scripts/benchmark/`, M11 production guide, runtime-manifest tests |

## Alternatives Considered

- Keep `NpEnv` and convert Torch/NumPy in runner wrappers. Rejected: CPU update-state/reset-done and
  process boundaries remain bottlenecks, while hidden fallbacks survive indefinitely.
- Rename the G1 task-local runtime as the general `TorchEnv`. Rejected: it is a narrow owner
  contract and does not own general Manager execution, playback, resume, or registry lifecycle.
- Keep a compatibility alias. Rejected: it preserves old imports, implicit conversion, and support
  ambiguity, contrary to the sole-runtime contract.
- Switch every backend in one change. Rejected: host-bridge and CUDA-native capability/performance
  differences require separate owner validation; one change hides parity and dependency risk.

## Consequences

- Manager-Based semantics remain, while Torch tensors become their only execution carrier.
- CPU physics does not move to CUDA; its exchange cost stays explicit, packed, and measurable.
- CUDA-native backends reject hidden host detours; unsupported capabilities/platforms fail closed.
- Registry, runner, benchmark, playback, and documentation owners migrate by stage rather than by
  base-class rename alone.
- New code must not introduce `NpEnv`, `NpEnvState`, legacy fallbacks, or NumPy state adapters after
  final deletion.
- This ADR does not promote any support grade or claim M10 multi-GPU scaling.

## Evidence In Repo

- The reusable tensor component helpers once cited here
  (`src/unilab/tasks/motion_tracking/common/tensor_runtime.py`,
  `src/unilab/tasks/motion_tracking/common/tensor_state_store.py`) had no
  production consumer once the Manager runtime became device-native; they
  were removed in the YAGNI sweep for issue #2629.
- General Manager runtime to migrate: `src/unilab/envs/manager_based_rl_env.py`
- Legacy runtime: removed. The historical path
  `src/unilab/base/np_env.py` no longer exists and is intentionally not cited
  as current evidence; the current lifecycle contract is
  `src/unilab/base/torch_env.py`.
- Registry/process boundaries: `src/unilab/base/env_factory.py`,
  `src/unilab/base/process_device.py`
- Public backend boundary: `unisim.backend.base.SimBackend`
- Production constraints:
  `docs/sphinx/source/en/4-developer_guide/8-tensor_runtime_production.md`
- Migration roadmap: [Issue #1701](https://github.com/Motphys/UniLab/issues/1701)

## Related Documents

- {doc}`ADR Index </adr/ADR-0000-index>`
- {doc}`ADR-0006 Community Manager API On NumPy Runtime </adr/ADR-0006-community-manager-api-on-numpy-runtime>`
- {doc}`ADR-0007 UniSim Extraction Boundary </adr/ADR-0007-unisim-extraction-boundary>`
- {doc}`Tensor Runtime Production </zh_CN/4-developer_guide/8-tensor_runtime_production>`
