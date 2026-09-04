# Shared-Memory Runtime — moved to `uni_rl`

The async IPC layer moved out of the `unilab` package into the independently
released **uni_rl** package (issue #1480): `uni_rl.ipc` hosts the async
runner, shared-memory buffers, replay pipelines, inference slot, DP launcher /
sync, and weight sync.

| Submodule | Role |
|---|---|
| `uni_rl.ipc.async_runner` | The high-level orchestration loop |
| `uni_rl.ipc.shared_buffer` | NumPy-backed shared-memory ring/buffer |
| `uni_rl.ipc.rollout_ring_buffer` | Rollout window used by on-policy collectors |
| `uni_rl.ipc.replay_buffer` | Bounded shared ingress for off-policy transitions |
| `uni_rl.ipc.replay_pipelines.*` | Authoritative CUDA/MPS replay ring, device gather, and native H2D |
| `uni_rl.ipc.inference_slot` | Fixed shared observation/action slot for learner-owned off-policy inference |
| `uni_rl.ipc.weight_sync` | Push learner weights to on-policy collector workers |
