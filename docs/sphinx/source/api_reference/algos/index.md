# `unilab.algos` — Learning Algorithms

Two parallel trees:

- **`unilab.algos.torch`** — PPO (RSL-RL), APPO, FastSAC, FastTD3, FlashSAC,
  HIM-PPO, HORA + distillation, generic off-policy runner.
- **`unilab.algos.mlx`** — Apple-silicon native PPO via [MLX](https://github.com/ml-explore/mlx).

All trainers conform to a single runner contract — see
{doc}`../../en/4-developer_guide/2-contracts/5-runner_lifecycle`.

```{toctree}
:maxdepth: 2

torch
mlx
```

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :template: autosummary/module.rst
   :recursive:

   unilab.algos
```
