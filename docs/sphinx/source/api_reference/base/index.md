# `unilab.base` — Core Contracts

The shared substrate everything else stands on. If you only read one module
in this reference, read this one.

| Symbol | Role |
|---|---|
| `NpEnv` / `NpEnvState` | The env contract every task implements |
| `SimBackend` | Abstract backend interface (MuJoCo / Motrix implement it) |
| `Registry` | Task / backend / algorithm registration and lookup |
| `Scene` | Cold-path scene materialization |
| `observations`, `final_observation` | Observation builders & terminal handling |
| `augmentation` | Symmetry / mirror augmentation utilities |
| `curriculum` | Curriculum schedule primitives |

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :template: autosummary/module.rst
   :recursive:

   unilab.base
```

## Selected detail

```{eval-rst}
.. autoclass:: unilab.base.np_env.NpEnv
   :members:
   :show-inheritance:
   :member-order: bysource
```

```{eval-rst}
.. autoclass:: unilab.base.backend.base.SimBackend
   :members:
   :show-inheritance:
   :member-order: bysource
```

```{eval-rst}
.. autoclass:: unilab.base.registry.Registry
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: unilab.base.scene.Scene
   :members:
   :show-inheritance:
```
