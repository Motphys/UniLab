# `unilab.training` — Training Runtime

Glue between `algos`, `envs` and `ipc`: experiment lifecycle and the
top-level `run` helpers invoked by the `train` / `eval` / `demo` CLI
entrypoints. Layer-0 helpers (seeding, monitoring, reward bookkeeping,
checkpoint resolution, sim2sim contracts) live in `unilab.utils`; resolved
env config adaptation lives in `unilab.base.config_adapter`.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :template: autosummary/module.rst
   :recursive:

   unilab.training
```
