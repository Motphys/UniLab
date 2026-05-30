# `unilab.envs` — Tasks

Concrete RL tasks split by family:

- **locomotion** — Go1, Go2, Go2w, Go2 + Airbot, Unitree G1
- **manipulation** — Allegro / Sharpa in-hand cube
- **motion_tracking** — G1 whole-body motion tracking + flips

Every env inherits `NpEnv` and is registered into the task `Registry` so it
can be selected via `uv run train --algo <algo> --task <name> --sim <backend>`.

```{toctree}
:maxdepth: 2

locomotion
manipulation
motion_tracking
common
```

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :template: autosummary/module.rst
   :recursive:

   unilab.envs
```
