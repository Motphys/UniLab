# Domain Randomization Contract

Manager-Based event terms are the sole UniLab DR lifecycle. There is no task
provider protocol and the environment does not carry a DR manager.

## Lifecycle

- **Construction identity:** `env.fixed_model_variants` materializes a final
  read-only assignment and attaches a UniSim `FixedVariantPlan` to `SceneCfg`.
  Backends realize it before their first forward and before CUDA graph capture.
- **Startup:** after manager construction, the backend materializes and event
  terms sample one full-width per-environment table. They commit through the
  same backend-owned packed reset transaction
  (`SimBackend.bind_packed_reset()`) as reset DR before read-plan binding,
  the first reset, or a step. When that commit succeeds, the packed composer
  promotes the staged fields of the committed rows into its persistent reset
  baseline, so later selected-row resets preserve unwritten startup values.
  Startup is opt-in and exchanges continuous
  distribution coverage for removing full-model derived-constant refreshes from
  the steady-state reset path.
- **Reset:** event terms write through Entity bindings into
  `ResetStateTransaction`; the transaction composes all staged writes in the
  backend-owned packed transaction and commits through it exactly once.
- **Range spelling:** lifecycle selection changes only the sampling boundary.
  In owner YAML, scalar bounds use the reset-owner array form such as
  `mass_distribution_params: [lo, hi]`; the mass/COM family keeps its axis
  mapping such as `com_range: {x: [lo, hi], y: [lo, hi], z: [lo, hi]}` in both
  startup and reset declarations.
- **Interval:** event terms use backend-owned interval plans through the public
  `SimBackend` contract.

## Capability Boundary

Backend differences are explicit capabilities, not task-side branches:

- `DomainRandomizationCapabilities.supported_reset_terms` — every reset term the
  backend supports through the public DR path
- `supported_interval_terms` — every interval term the backend supports through
  the public DR path
- fixed-variant layouts
- per-environment playback support
- `TensorLifecycleCapabilities.packed_reset` with
  `SimBackend.bind_packed_reset()`: required for the Manager-Based reset and
  startup DR lifecycle; a missing declaration fails closed at env
  construction.

These are single, plane-independent term sets. Manager capability negotiation
does not inspect the payload device or pass a plane. CPU/GPU execution and any
required internal conversion belong to the backend adapter, not to separate
public capability sets. An unadvertised requested term fails closed with the
backend and term named; a backend may also reject payloads that violate its
device or lifecycle contract. Manager code never imports MuJoCo or mjbatch and
never accesses a backend model or pool.

## Reset Payload And Defaults

`TensorResetRandomizationPayload` is the sole public reset payload: a curated
torch plan whose first dimension is the selected row count. Its 16 fields cover
the body mass/COM/inertia family, gravity, geometry friction/size/solver
parameters, joint damping/armature/friction, actuator gains, and the
`base_mass_delta`/`base_com_offset` delta terms. Fields stay `Any`-typed with
lazy torch validation, so `unisim-core` keeps its NumPy-only dependency
footprint. NumPy survives only behind adapter-private host conversion
(`unisim.dr._host`), cold metadata, and serialization/playback leaves; the
former NumPy `ResetRandomizationPayload` no longer exists. Derived fields such
as geometry bounds are backend-owned and are never independently
caller-supplied.

The packed composer merges staged delta terms with the authoritative baseline
into final absolute tables in torch at commit (`final = baseline + delta`), so
reset DR on CUDA environments no longer round-trips values through the host.

During cold-path binding, `ResetStateTransaction` asks UniSim for
`SimBackend.get_reset_term_default(term)`. The returned table is authoritative
and has one of two layouts:

- canonical model table, such as `(nbody,)` for `body_mass`;
- per-environment fixed-variant table, such as `(num_envs, nbody)`.

For a selected reset subset, the packed composer's baseline is authoritative.
A write to a subset of model columns stages only those columns; unwritten
columns keep the composer's per-env baseline, which the backend-owned composer
merges at its single commit.
Missing capabilities, unsupported terms, non-floating tables, invalid tails, and
per-env tables whose first dimension is not `num_envs` fail closed. This removes
the former UniLab-side MuJoCo recompilation used to obtain inertia defaults.

## Interval Terms

Interval plans are term-descriptor based: `IntervalRandomizationPlan.ops`
carries `IntervalTermOp` entries from `unisim.dr.interval`. Payloads are
torch-first: `IntervalTermOp.payload` is a torch tensor or a
`TensorIntervalPayload`; push is producer-sampled on every backend —
backend-internal global-RNG push sampling no longer exists. Builtin payload
contracts are enforced by `IntervalTermOp.validate`; unknown backend-owned
custom terms pass through to
that backend's handler table. Plans that cross spawn collectors carry sampling
ranges rather than tensors; device payloads are process-local.

The structural decision is recorded in UniSim's
[torch-first DR payload ADR](https://github.com/unilabsim/unisim/blob/main/docs/en/adr-dr-torch-first-payloads.md).

## Evidence

- Manager lifecycle: `src/unilab/managers/event_manager.py`
- Reset transaction: `src/unilab/base/reset_state.py`
- Entity bindings: `src/unilab/base/entity.py`
- Task-owned fixed variants: `src/unilab/base/variants.py`
- Backend contract/capability types: `unisim.backend.base`, `unisim.dr.types`
- ADR: {doc}`ADR-0010 Fixed Model Variant Ownership Boundary </adr/ADR-0010-fixed-model-variant-ownership-boundary>`
