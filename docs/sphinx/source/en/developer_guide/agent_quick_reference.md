# Agent Quick Reference

This page is for maintainers and agents who need the shortest route to current
repo facts.

## Start Here

- Install and smoke check: {doc}`../getting_started/installation`
- Backend choice: {doc}`../user_guide/backends/choosing_a_backend`
- PPO entrypoint: `scripts/train_rsl_rl.py`
- MLX PPO entrypoint: `scripts/train_mlx_ppo.py`
- APPO entrypoint: `scripts/train_appo.py`
- SAC / TD3 / FlashSAC entrypoint: `scripts/train_offpolicy.py`
- HIM-PPO entrypoint: `scripts/train_him_ppo.py`
- HORA distillation entrypoint: `scripts/train_hora_distill.py`

## Contracts To Keep In Mind

- Env contract: `src/unilab/base/np_env.py`
- Backend contract: `src/unilab/base/backend/base.py`
- Training helpers: `src/unilab/training/run.py`
- Config schema: `src/unilab/structured_configs.py`
- Developer standard: {doc}`architecture/overview`

Use `uv run scripts/...` for script examples, choose backends through task owner
YAML, and write only facts that can be traced to code, config, tests, or current
docs.
