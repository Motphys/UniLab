# `unilab.base.backend` — Simulation Backends

UniLab abstracts two CPU-side physics backends behind a single
`SimBackend` contract.

| Backend | Strengths | Notes |
|---|---|---|
| **MuJoCo** (`mujoco-uni`) | Mature, broad asset support, deterministic | Default for research |
| **Motrix** (`motrixsim-core`) | High-throughput, multithread step, snapshot/playback | Cross-platform; required for video export on macOS |

Pick a backend per task via `task=<task>/<backend>` — see
{doc}`../../user_guide/backends/index`.

```{eval-rst}
.. autoclass:: unilab.base.backend.base.SimBackend
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :template: autosummary/module.rst
   :recursive:

   unilab.base.backend.mujoco
   unilab.base.backend.motrix
```

```{eval-rst}
.. automodule:: unilab.base.backend.playback_common
   :members:
```

```{eval-rst}
.. automodule:: unilab.base.backend.motrix_camera
   :members:
```
