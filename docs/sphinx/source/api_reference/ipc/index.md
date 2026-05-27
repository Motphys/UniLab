# `unilab.ipc` — Shared-Memory Runtime

The bridge between CPU simulation workers and the GPU learner. Everything
here is a building block of the **async runner** that powers APPO / FastSAC
/ FastTD3 / FlashSAC.

| Submodule | Role |
|---|---|
| `async_runner` | The high-level orchestration loop |
| `shared_buffer` | NumPy-backed shared-memory ring/buffer |
| `rollout_ring_buffer` | Rollout window used by on-policy collectors |
| `replay_buffer` | Off-policy replay backed by shared memory |
| `replay_pipelines.*` | Host-to-device staging (CPU-pinned double buffer, native h2d) |
| `shared_obs_stats` | Running mean/std shared across workers |
| `weight_sync` | Push learner weights back to workers |

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :template: autosummary/module.rst
   :recursive:

   unilab.ipc
```

## Async runner

```{eval-rst}
.. automodule:: unilab.ipc.async_runner
   :members:
   :show-inheritance:
```
