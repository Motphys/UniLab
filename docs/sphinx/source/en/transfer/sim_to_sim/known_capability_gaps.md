# Known Capability Gaps

A living table of features that are *currently* available on one backend
but not the other. Update this page whenever a new gap is discovered or
closed.

```{list-table}
:header-rows: 1
:widths: 30 15 15 40

* - Feature
  - MuJoCo
  - Motrix
  - Notes
* - Native snapshot/restore
  - ❌
  - ✅
  - See ADR-0002.
* - Headless video export (macOS)
  - ⚠️ requires extra setup
  - ✅
  - Motrix is the default for video.
* - MJX kernel acceleration
  - ✅ (via `mujoco-uni`)
  - ❌
  - Only matters if you opt into MJX path.
* - Penalty-based soft contact
  - ⚠️ via solver tuning
  - ✅
  - Choose deliberately; affects reward parity.
* - Skinned mesh visualization
  - ✅
  - ⚠️
  - Visual-only difference.
* - Native multithread step
  - ❌
  - ✅
  - Motrix scales better on big core counts.
* - URDF import
  - ⚠️ via mjcf converter
  - ⚠️ subset
  - Always prefer MJCF as source of truth.
```

Legend: ✅ supported · ⚠️ partial / qualified · ❌ unsupported.

## How to update this page

When you add or remove a capability:

1. Update the table above.
2. If the change is a *new contract*, add an ADR under
   `developer_guide/adr/`.
3. Update the affected task owner YAMLs' `capabilities` blocks.
4. Mention the change in the next release notes (see {doc}`../../changelog`).

## See also

- {doc}`why_switch`
- {doc}`../../developer_guide/contracts/backend_capability`
