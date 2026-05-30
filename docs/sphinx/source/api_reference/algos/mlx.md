# `unilab.algos.mlx`

Apple-silicon native PPO built on [MLX](https://github.com/ml-explore/mlx).
Available only on `sys_platform == 'darwin'`.

The public entrypoint is `scripts/train_mlx_ppo.py`, with shared MLX helpers
under `src/unilab/algos/mlx/`. The Linux documentation builder does not
autogenerate MLX API pages because MLX is an Apple-platform dependency.
