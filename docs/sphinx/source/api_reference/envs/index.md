# `unilab.envs` — Environment runtime

Task-agnostic Manager-Based environment runtime and reusable MDP terms.
Concrete task implementations are owned by {doc}`../tasks/index`.

`ManagerBasedRLEnv` preserves UniLab's NumPy `NpEnv` contract while executing
community-style action, observation, reward, termination, event, command, and
curriculum managers.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :template: autosummary/module.rst
   :recursive:

   unilab.envs
```
