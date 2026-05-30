# Capability Gaps

A living table of backend capabilities that are surfaced through
`SimBackend` and the current backend implementations. Update this page when a
capability is added to or removed from the backend contract.

```{list-table}
:header-rows: 1
:widths: 30 15 15 40

* - Capability
  - MuJoCo
  - Motrix
  - Notes
* - `supports_physics_state_playback`
  - Yes
  - No
  - Reported by `get_play_capabilities()`.
* - `supports_native_interactive_renderer`
  - No
  - Yes
  - Reported by `get_play_capabilities()`.
* - `supports_native_video_capture`
  - No
  - Yes
  - Reported by `get_play_capabilities()`.
* - `create_hfield_scanner(...)`
  - Yes
  - Yes
  - Backend-owned height-field scanner for rough-terrain observations.
* - `apply_interval_randomization(...)`
  - Yes
  - Yes
  - Supported fields still depend on each backend's `get_dr_capabilities()`.
* - Position-actuator gain DR
  - Yes
  - Conditional
  - Motrix reports support from runtime capability detection.
* - Geom-friction DR
  - Yes
  - Conditional
  - Motrix reports support from runtime capability detection.
```

Use `src/unilab/base/backend/base.py` as the contract source and backend tests
under `tests/base/` as evidence for support claims.

## How to update this page

When you add or remove a capability:

1. Update the table above.
2. If the change is a new contract, add or update the relevant ADR under
   {doc}`/adr/ADR-0000-index`.
3. Add or update backend tests under `tests/base/`.
4. Mention user-visible changes in the changelog (see {doc}`/changelog`).

## See also

- {doc}`1-backend_swap`
- {doc}`../../4-developer_guide/2-contracts/2-backend_contract`
