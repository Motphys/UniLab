# `unilab.tasks` — Concrete tasks

Concrete RL tasks split by family:

- **locomotion** — A2, Go1, Go2, Go2w, Go2 + Airbot, and Unitree G1
- **manipulation** — Allegro / Sharpa in-hand cube and Stewart balance
- **motion_tracking** — G1 and X2 whole-body motion tracking

Every task is registered into the task `Registry` so it can be selected via
`uv run train --algo <algo> --task <name> --sim <backend>`.

```{toctree}
:maxdepth: 2

locomotion
manipulation
motion_tracking
```

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :template: autosummary/module.rst
   :recursive:

   unilab.tasks
```
