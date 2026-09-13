# Domain Randomization Contract

Manager-Based event terms are the sole UniLab DR lifecycle. There is no task
provider protocol and `NpEnv` no longer carries a DR manager.

## Lifecycle

- **Construction identity:** `env.fixed_model_variants` materializes a final
  read-only assignment and attaches a UniSim `FixedVariantPlan` to `SceneCfg`.
  Backends realize it before their first forward and before CUDA graph capture.
- **Reset:** event terms write through Entity bindings into
  `ResetStateTransaction`; the transaction calls
  `SimBackend.set_state(..., randomization=...)` once.
- **Interval:** event terms use backend-owned interval plans through the public
  `SimBackend` contract.

## Capability Boundary

Backend differences are explicit capabilities, not task-side branches:

- `DomainRandomizationCapabilities.supported_reset_terms`
- `supported_interval_terms`
- fixed-variant layouts
- per-environment playback support

An unadvertised requested term fails closed with the backend and term named.
Manager code never imports MuJoCo or mjbatch and never accesses a backend model
or pool.

## Reset Payload And Defaults

`ResetRandomizationPayload` is a curated NumPy plan whose first dimension is the
selected row count. Supported terms include the body mass/COM/inertia family,
gravity, geometry friction/size/solver parameters, joint damping/armature/friction,
and actuator gains. Derived fields such as geometry bounds are backend-owned and
are never independently caller-supplied.

During cold-path binding, `ResetStateTransaction` asks UniSim for
`SimBackend.get_reset_term_default(term)`. The returned table is authoritative
and has one of two layouts:

- canonical model table, such as `(nbody,)` for `body_mass`;
- per-environment fixed-variant table, such as `(num_envs, nbody)`.

For a selected reset subset, event terms use the corresponding per-env rows as
their baseline. A write to a subset of model columns fills every unwritten
column from that same env row before the transaction builds one dense payload.
Missing capabilities, unsupported terms, non-floating tables, invalid tails, and
per-env tables whose first dimension is not `num_envs` fail closed. This removes
the former UniLab-side MuJoCo recompilation used to obtain inertia defaults.

## Interval Terms

Interval plans are term-descriptor based: `IntervalRandomizationPlan.ops`
carries `IntervalTermOp` entries from `unisim.dr.interval`. Builtin payload
contracts are enforced by `IntervalTermOp.validate`;
unknown backend-owned custom terms pass through to that backend's handler table.
Ops and plans remain pickle-safe stdlib/NumPy data across spawn collectors.

## Evidence

- Manager lifecycle: `src/unilab/managers/event_manager.py`
- Reset transaction: `src/unilab/base/reset_state.py`
- Entity bindings: `src/unilab/base/entity.py`
- Task-owned fixed variants: `src/unilab/base/variants.py`
- Backend contract/capability types: `unisim.backend.base`, `unisim.dr.types`
- ADR: {doc}`ADR-0010 Fixed Model Variant Ownership Boundary </adr/ADR-0010-fixed-model-variant-ownership-boundary>`
