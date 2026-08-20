# `unilab.algos` — Learning Algorithms

- **`unilab.algos`** — PPO (RSL-RL), APPO, FastSAC, FastTD3, FlashSAC,
  HIM-PPO, HORA + distillation, generic off-policy runner.

All trainers conform to a single runner contract — see
{doc}`../../en/4-developer_guide/2-contracts/5-runner_lifecycle`.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :template: autosummary/module.rst
   :recursive:

   unilab.algos.common
   unilab.algos.appo
   unilab.algos.fast_sac
   unilab.algos.fast_td3
   unilab.algos.flash_sac
   unilab.algos.him_ppo
   unilab.algos.hora
   unilab.algos.offpolicy
```

## Standalone PPO entrypoints

```{eval-rst}
.. automodule:: unilab.algos.rsl_rl_ppo
   :members:
```

```{eval-rst}
.. automodule:: unilab.algos.rsl_rl_runtime
   :members:
```
