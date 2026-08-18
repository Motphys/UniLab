# Manager-Based API

UniLab uses a community-compatible manager API on its NumPy runtime. Manager modules,
term configs, function/class terms, lifecycle ordering, and reset semantics follow the
pinned mjlab 1.6.0 source. Numeric execution uses NumPy while preserving UniLab's
`NpEnvState`, Hydra owner YAML, `SimBackend`, registry, and IPC contracts.

The normative compatibility matrix and mechanical migration example are in
{doc}`ADR-0006 </adr/ADR-0006-community-manager-api-on-numpy-runtime>`.

## Invariants

- Community manager semantics and a general structure take priority over local
  optimizations that would create a UniLab-only term API.
- Manager buffers, term returns, environment IDs, and entity views use `np.ndarray`
  or `slice`; manager core does not depend on Torch, Warp, runners, learners, or IPC.
- `SceneEntityCfg` resolves through a base-owned scene/entity facade on the cold path.
  The facade uses only the public `SimBackend` contract, and hot paths reuse cached IDs
  and views.
- Named-sensor observation terms bind a backend-owned view through
  `EntityScene.bind_sensor_data(...)` during construction; their hot path only reads
  that view and never re-resolves sensor names or XML/model metadata.
- Explicitly empty configuration may use a Null manager. A requested capability that
  is unavailable fails at the nearest boundary; it is never skipped, zero-filled, or
  routed back to a legacy environment.
- Hot paths avoid obvious repeated parsing, per-environment Python loops, copies, and
  temporary allocations. Further optimization requires benchmark evidence and must
  not add disproportionate structural complexity.

Only surfaces backed by registration, configuration, and tests may be called
Compatible. NumPy/env/config adapters are Adapted; capabilities without a formal
backend contract are Unsupported and fail closed.
