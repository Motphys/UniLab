---
orphan: true
---

# ADR-0021 Genesis Device-Resident DR Lifecycle

语言: 简体中文

- Status: Accepted
- Date: 2026-10-26
- Owners: Simulation backend / Manager runtime maintainers
- Supersedes: None
- Superseded by: None
- Roadmap: [Issue #2520](https://github.com/Motphys/UniLab/issues/2520);
  FlashMimic subset extension:
  [Issue #2521](https://github.com/Motphys/UniLab/issues/2521)

## Context

Genesis previously carried two runtime-DR profiles. The whole-MJCF exact-CUDA lane exposed a
narrow device-resident packed reset contract, while portable `entity_assets` scenes retained
host-plane model-field setters and a portable-only wrench path. Capability wording also mixed
payload plane with lifecycle support: portable host setters could look like Manager runtime DR
even though the Manager reset transaction requires a device-resident packed commit.

This crossed the public `SimBackend` DR capability boundary and the Manager reset/interval
execution contract. UniSim issue #2520 therefore approved one adapter-level lifecycle decision
before this integration; the authoritative adapter ADR and implementation live in `unisim-core`
(`docs/en/adr-genesis-dr-lifecycle.md`).

## Decision

UniLab consumes the unified contract as device-only Genesis runtime DR:

- Genesis declares reset terms only in `supported_device_reset_terms`, and interval terms only
  in `supported_device_interval_terms`. The host term sets are empty, including for portable
  scenes and sessions without the exact CUDA/AMDGPU lifecycle.
- Manager capability negotiation selects the plane from the backend-bound tensor device. It never
  treats default host capability wording as a fallback for a device-resident runtime.
- The packed reset device is declared before cold event-term binds, so owner terms negotiate
  against the same plane that will commit after backend materialization.
- Reset DR is limited to the audited device lifecycle (`body_mass`, `base_mass_delta`,
  `body_ipos`, `base_com_offset`, `kp`, `kd`); interval DR is limited to `body_force`,
  `body_torque`, world-frame root linear velocity delta, and root-local angular velocity delta.
  Unsupported, host-plane, mixed-plane, and CPU/host payloads fail closed with the term and plane
  named.
- Genesis owner YAML enables `pd_gains` only because that device-plane packed lane declares
  `kp`/`kd`. It does not re-enable broad portable host setters, geometry, gravity, DOF, or the
  opaque legacy push descriptor. Startup body mass and the six-axis velocity-delta event use
  their audited public device lifecycle; absolute `geom_friction` remains intentionally
  fail-closed because upstream Genesis has no equivalent per-environment absolute friction API.
- Event operands and interval body IDs are lifted to the backend-bound device before entering the
  public interval plan. A host RNG or NumPy input at a Manager leaf is a cold compatibility edge,
  not permission to execute runtime DR on CPU.

## Stable Contracts

- `DomainRandomizationCapabilities` plane sets are authoritative; aggregate projections keep
  `dr.reset.*`/`dr.interval.*` and `dr.device.reset.*`/`dr.device.interval.*` independent.
- The Manager packed reset transaction declares its device before reset event construction.
- Genesis interval dispatch uses public `IntervalTermOp` values only; backend IDs are Torch int64
  on the backend device and wrench payloads are float32 on that same device.
- Genesis owners cannot declare backend capability by comments or backend-name branches; owner
  YAML may only enable terms whose plane the current backend declares.

## Alternatives Considered

- **Default to the historical host plane before packed-device declaration:** rejected. It made a
  portable host setter look valid during cold binds and could accept a term that the eventual
  device commit must reject.
- **Convert host operands inside each backend adapter:** rejected as the Manager contract. It
  reintroduces hidden hot-path H2D detours and plane ambiguity at every adapter boundary.
- **Keep `events.pd_gains: null` until every Genesis reset term is device-capable:** rejected.
  `kp`/`kd` have one audited lifecycle; unrelated unsupported terms already fail closed and need
  no owner-level blanket disable.

## Consequences

- Genesis SAC and PPO walk owners have the same PD-gain DR default as their MuJoCo counterparts.
- The SAC G1 motion-tracking owner carries the FlashMimic-compatible DR subset: startup
  pelvis/torso mass scaling without inertia recomputation, torso COM and full-actuator PD gain
  reset writes, and six-axis velocity kicks. Its foot-friction event stays disabled.
- Portable Genesis scenes have a smaller reset contract, and removed portable host runtime paths
  have no compatibility alias in UniLab.
- Owners must sample device-resident operands on the Manager RNG/backend device. CPU-only Genesis
  sessions do not advertise runtime DR.
- Adding a Genesis reset or interval term requires first extending the audited device lifecycle
  and capability declaration in `unisim-core`, then updating owners/docs here.

## Evidence In Repo

- `src/unilab/base/entity.py` — interval plane negotiation and device-resident interval ops.
- `src/unilab/envs/manager_based_rl_env.py` — packed reset device declaration before event binds.
- `src/unilab/base/reset_state.py` — plane-aware capability checks.
- `src/unilab/conf/{ppo,sac}/task/g1_walk_flat/genesis.yaml` — walk owner parity.
- `src/unilab/conf/sac/task/g1_motion_tracking/genesis.yaml` — FlashMimic DR subset owner.
- `tests/base/test_genesis_backend.py`; `tests/envs/mdp/test_events.py`;
  `tests/envs/locomotion/g1/test_g1_owner_contract.py`.
- Generated backend support evidence:
  `docs/sphinx/source/{en,zh_CN}/5-reference/5-support_matrix.md`.
- Adapter decision: `unisim-core` `docs/en/adr-genesis-dr-lifecycle.md`.

## Related Documents

- {doc}`ADR Index </adr/ADR-0000-index>`
- {doc}`ADR-0012 Sole Tensor Manager And Scoped Backends </adr/ADR-0012-sole-tensor-manager-and-scoped-backends>`
- {doc}`Genesis Backend </en/2-user_guide/3-backends/5-genesis>`
