# Playback and Snapshot Differences

Beyond the physics step itself, the two backends differ in **how they let
you replay** a run. This page captures the practical implications.

## Playback

| Backend | Mechanism | Best for |
|---|---|---|
| MuJoCo | Physics-state playback path reported by `supports_physics_state_playback` | Record-mode playback and offline video export |
| Motrix | Native interactive renderer and native video capture reported by `get_play_capabilities()` | Interactive playback and record-mode capture |

Both backends resolve `training.play_render_mode` through
`SimBackend.resolve_play_render_plan(...)`; unsupported modes should fail at
the backend boundary instead of branching in training scripts.

## Snapshot

MuJoCo currently reports `supports_physics_state_playback=True`. Motrix reports
native interactive rendering and native video capture instead. Treat these as
different backend capabilities rather than feature parity.

The capability-boundary contract
({doc}`../../4-developer_guide/2-contracts/2-backend_contract`) requires that
*neither* env code nor algorithm code call snapshot-only paths directly —
it must be routed through a capability-aware abstraction. ADR-0002
codifies this.

## What to put in your task owner

If a task starts to require a playback or snapshot capability, add that need to
the backend contract first, then validate it at the env/backend boundary. Do not
hide the requirement in a training script branch.

## See also

- {doc}`6-capability_gaps`
- {doc}`/adr/ADR-0002-backend-capability-boundary-for-play-and-snapshot`
