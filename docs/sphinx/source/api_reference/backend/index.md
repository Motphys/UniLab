# `unisim-core` — Simulation Backends

UniSim owns the unified physics contract and all production adapters. UniLab
only assembles task-owned scenes through `unilab.base.backend_factory`.

| Backend | Strengths | Notes |
|---|---|---|
| **MuJoCo** (`mujoco` + `mujoco-uni-runtime`) | Mature, broad asset support, deterministic | Default for research |
| **Motrix** (`motrixsim-core`) | High-throughput, multithread step, snapshot/playback | Cross-platform; required for video export on macOS |

Pick a backend per task via the top-level `--sim <backend>` CLI flag — see
{doc}`../../en/2-user_guide/3-backends/0-index`.

```{eval-rst}
.. autoclass:: unisim.backend.base.SimBackend
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :template: autosummary/module.rst
   :recursive:

   unisim.backend.mujoco
   unisim.backend.motrix
   unisim.backend.drake
   unisim.backend.mjwarp
   unisim.backend.genesis
   unisim.backend.isaacgym
   unisim.backend.isaacsim
```

```{eval-rst}
.. automodule:: unisim.backend.playback_common
   :members:
```

```{eval-rst}
.. automodule:: unisim.backend.motrix_camera
   :members:
```
