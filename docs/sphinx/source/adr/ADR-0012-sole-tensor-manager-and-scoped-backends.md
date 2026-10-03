---
orphan: true
---

# ADR-0012 Sole Tensor Manager And Scoped Backends

语言: 简体中文

- Status: Proposed
- Date: 2026-10-02
- Owners: Env / Manager / Training / Backend maintainers
- Supersedes: [ADR-0011](ADR-0011-torch-only-manager-based-runtime.md)
- Superseded by: None
- Roadmap: [Issue #1811](https://github.com/Motphys/UniLab/issues/1811)

## Context

ADR-0011 removed the legacy `NpEnv` class and established `TorchEnv` as the sole lifecycle base,
but the runtime is not yet the sole tensor-native Manager. At the current
`develop/tensor-runtime` baseline:

- generic Manager execution still has NumPy/host boundaries in event transactions, reset
  composition, row handling, and selected-row reset (#1703);
- `env.tensor_runtime` remains a per-owner switch and a runner-side placement input;
- FlashSAC `g1_motion_tracking` registers a second task-owned direct environment,
  `TorchG1MotionTrackingFlashSACEnv`, rather than entering through the generic Manager factory;
- task/runtime code still contains backend-private `_sensor_map` probing and backend-name
  readiness branches (#1790);
- active backend breadth exceeds the scope that can be validated during this closing migration.

Recent low-clock multi-GPU measurements show that host-boundary count, not MJWarp physics,
dominates collector cost. Reusing stable device views improved G1 walk throughput by roughly
22–25%, while replacing a compact reset composer with many small Torch operations regressed
selected reset. The final contract therefore needs fewer and larger public operations, not merely
“more Torch”.

This ADR records the structural decisions approved in #1811. It is a review baseline, not a
delivery checklist; implementation remains staged in that roadmap.

## Decision

### Sole Manager-Based training entry point

`make_manager_based_rl_env()` is the sole production factory for training environments. A task
owner may register fused observation/reward/reset components for performance, but those components
are Manager-owned: they are declared by Manager config, execute inside the Manager lifecycle,
consume only public `SimBackend` APIs, and remain protected by an owner semantic fingerprint. They
are not a second environment class, not a direct runner mode, and not a backend-specific branch.

When this ADR is implemented, task-owned direct environment classes are not registered as training
entry points.

### Tensor-only Manager carrier

The Manager public lifecycle and its internal execution are tensor-native. Public row selectors,
state, observations, rewards, terminations, commands, events, metrics, and recorder carriers are
Torch tensors. Generic NumPy execution and dual-carrier term branches are deleted.

NumPy remains allowed only inside a backend adapter that declares a `HOST_BRIDGE` data plane and
owns its explicit packed conversion boundary. MuJoCo remains a production backend and is the
canonical in-process host-bridge implementation; it is not a test-only fallback.

`env.tensor_runtime` and `env.tensor_runtime_device` are removed. Tensor execution is an invariant
of the Manager runtime. Placement is derived from declared backend data plane and rank topology.
No compatibility alias, dual-mode field, or hidden fallback is retained.

### Manager-owned RNG

The Manager runtime owns one Torch RNG seeded from the resolved owner seed. Observation noise,
commands, events, and tensor selected reset consume that generator or generators explicitly
derived from it. Seed/reset state and selected-row reproducibility are contract-tested.

Legacy NumPy stream parity is not a compatibility requirement. Changing the stream is acceptable;
nondeterministic owner seed behavior is not.

### Strong selected-reset publication postcondition

For a backend declaring selected reset, once `set_state_tensor` returns, subsequent public state
and sensor views are authoritative for the committed rows. A backend may implement this by
immediate refresh or by lazy refresh at the first public view. Callers must not issue an extra
physics/control step to obtain readiness.

MuJoCo's host-bridge contract remains the paired selected reset/read boundary. Runtime
diagnostics may describe implementation details, but capability semantics—not diagnostic
values—control dispatch.

### Public backend metadata replaces probing

Task/runtime code must not inspect backend-private attributes, dispatch capability by backend
name, or require undeclared `nq`/`nv` attributes. Backend behavior is consumed through public
contract methods and capabilities, including:

- selected-reset publication semantics;
- a complete public sensor namespace/inventory; and
- public qpos/qvel state widths.

A repository architecture gate rejects backend-private attribute access and backend-name
capability dispatch in UniLab task/runtime code.

### Temporary backend scope gate

This migration's active runtime scope is `mujoco`, `mjwarp`, `genesis`, and `newton`. Other adapters remain
present in UniSim but are gated out of this runtime with actionable errors. Re-enabling a backend
requires its own capability, parity, and support-matrix evidence. Permanent adapter deletion is a
separate maintainer decision; this ADR does not delete those adapters.

### Single-GPU default device behavior

A single-GPU MJWarp run does not require `CUDA_VISIBLE_DEVICES`. It uses the current default CUDA
device. Explicit device selection remains required for ambiguous multi-GPU training and remains
useful for launcher/debug/MPS isolation.

## Stable Contracts

| Contract | Owner / planned evidence |
| --- | --- |
| Sole training factory and no direct registration | `src/unilab/tasks/**/__init__.py`, registry tests, direct-entry architecture gate |
| Tensor-only Manager execution | `src/unilab/envs/manager_based_rl_env.py`, Manager suites, no-NumPy-carrier gate |
| Removal of `env.tensor_runtime` | `ManagerBasedRlEnvCfg`, owner YAMLs, run-config schema, config tests |
| Manager-owned Torch RNG | Manager RNG owner, seed/reset and selected-row reproducibility tests |
| Selected-reset publication postcondition | UniSim tensor capability + MuJoCo/MJWarp/Genesis/Newton contract tests |
| Public sensor namespace | UniSim `SimBackend.get_sensor_names()`/inventory contract and adapter tests |
| Public state widths | UniSim state-width API and reset-plan tests |
| No backend private/name probing | architecture gate in UniLab tests |
| Temporary backend gate | backend factory/registry/docs/support matrix tests |
| Single-GPU default device | process-device/train smoke tests |

## Alternatives Considered

- Keep the direct FlashSAC environment as a performance exception. Rejected: it preserves a second
  training entry point and makes backend contract governance conditional on workload performance.
- Make every Manager term execute through the generic per-term loop. Rejected as a forced
  implementation choice: it reintroduces known Python-dispatch overhead. Fused Manager-owned terms
  preserve the sole API without mandating one execution strategy.
- Keep `tensor_runtime` as an owner switch. Rejected: the switch preserves dual contracts and lets
  unsupported workloads silently select a host fallback.
- Negotiate readiness through runtime diagnostics. Rejected: diagnostics describe state; capability
  semantics must own control flow.
- Permanently delete all non-scoped adapters now. Rejected: this mixes an irreversible support
  decision with a runtime migration. The temporary gate keeps the review boundary explicit.
- Set a strict “no regression” migration gate. Rejected: absorbing the fused direct owner can
  transiently regress a narrow workload. The roadmap uses a bounded migration allowance plus a
  mandatory optimization follow-up.

## Consequences

- Manager-Based semantics remain, but the Manager is the only training entry point and its carrier
  is tensor-only.
- Fused owner kernels are valid only as Manager-owned implementation strategies, not as direct
  modes.
- MuJoCo continues to carry the production host-bridge contract.
- MJWarp, Genesis, and Newton must make public views authoritative after selected reset without an extra control step.
- Backend support claims narrow during migration and are restored only with new evidence.
- Owner configurations lose `env.tensor_runtime`; migration documentation must state the breaking
  boundary without offering a compatibility alias.
- RNG streams may change, but reproducibility from an owner seed becomes explicit.
- Performance convergence is staged: architectural closure is separated from a mandatory
  optimization follow-up.

## Evidence In Repo

- Manager runtime: `src/unilab/envs/manager_based_rl_env.py`
- Absorbed FlashSAC owner fingerprint:
  `src/unilab/tasks/motion_tracking/g1/flashsac_owner_contract.py`
- Tensor state store/readiness consumer:
  `src/unilab/tasks/motion_tracking/common/tensor_state_store.py`
- Public backend contract: `unisim.backend.base.SimBackend`
- Scoped adapters: `unisim.backend.mujoco`, `unisim.backend.mjwarp`, `unisim.backend.genesis`, `unisim.backend.newton`
- Backend-name readiness issue: #1790
- Current Manager tensor migration: #1703
- Delivery roadmap and performance budget: #1811
- Superseded runtime decision: ADR-0011

## Related Documents

- {doc}`ADR Index </adr/ADR-0000-index>`
- {doc}`ADR-0011 Torch-Only Manager-Based Runtime </adr/ADR-0011-torch-only-manager-based-runtime>`
- {doc}`ADR-0007 UniSim Extraction Boundary </adr/ADR-0007-unisim-extraction-boundary>`
- {doc}`Tensor Runtime Production </zh_CN/4-developer_guide/8-tensor_runtime_production>`
