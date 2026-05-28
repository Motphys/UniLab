# Playback and Snapshot Differences

Beyond the physics step itself, the two backends differ in **how they let
you replay** a run. This page captures the practical implications.

## Playback

| Backend | Mechanism | Best for |
|---|---|---|
| MuJoCo | Re-run the simulation from initial state + action trace | Deterministic re-run, debugging |
| Motrix | Native frame-by-frame snapshot stream | Video export, fast scrubbing, headless render |

Motrix's `--render-mode record` works headlessly out of the box; MuJoCo
playback on macOS / servers needs additional setup
(GLFW / EGL / Xvfb).

## Snapshot

Motrix can serialize the simulator state mid-run; MuJoCo can't natively at
the same granularity. This matters if you:

- Want to fork an episode at a checkpoint and explore multiple actions.
- Are building an analysis tool that scrubs through long trajectories.

The capability-boundary contract
({doc}`../../developer_guide/contracts/backend_capability`) requires that
*neither* env code nor algorithm code call snapshot-only paths directly —
it must be routed through a capability-aware abstraction. ADR-0002
codifies this.

## What to put in your task owner

If your task **requires** snapshot (e.g. tree-search policies), declare it
in the owner YAML's capability block:

```yaml
capabilities:
  required:
    - snapshot
    - playback_native_video
```

If a backend lacks a required capability the registry must refuse the
task. See
{doc}`../../developer_guide/architecture/registry_bootstrap`.

## See also

- {doc}`known_capability_gaps`
- {doc}`/adr/ADR-0002-backend-capability-boundary-for-play-and-snapshot`
