# `unilab.tasks` — Concrete tasks

Concrete RL tasks split by family:

- **locomotion** — Go2 and Unitree G1 reference owners
- **manipulation** — Allegro in-hand cube and Stewart balance
- **motion_tracking** — G1 and X2 whole-body motion tracking

Every task is registered into the task `Registry` so it can be selected via
`uv run train --algo <algo> --task <name> --sim <backend>`.
Unitree production variants are documented in `unitree_rl_unilab`.

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
