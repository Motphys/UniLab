# Backend Capability Contract

Backends declare capabilities. Tasks require capabilities. The registry
refuses (task, backend) pairs where required > declared.

Capability examples: `snapshot`, `playback_native_video`,
`contact_with_friction_anisotropy`, `gpu_step`.

The contract is normative — env code must NOT probe backend internals via
`getattr` / `hasattr` to detect a capability. Always go through the
capability table. See ADR-0002
({doc}`../adr/ADR-0002-backend-capability-boundary-for-play-and-snapshot`).
