# Agent Quick Reference

This page is for maintainers and agents who need the shortest route to current
repo facts.

## Start Here

- Install and smoke check: {doc}`../1-getting_started/2-installation`
- Backend choice: {doc}`../2-user_guide/3-backends/3-choosing_a_backend`
- Task index: {doc}`../2-user_guide/4-tasks/0-index`
- Algorithms index: {doc}`../2-user_guide/2-algorithms/0-index`
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
- Developer standard: {doc}`1-architecture/1-overview`
- High-risk areas: see the repo-root `AGENTS.md`.

Use `uv run train`, `uv run eval`, or `uv run demo` for command examples.
Choose algorithm, task, and backend through `--algo`, `--task`, and `--sim`;
write only facts that can be traced to code, config, tests, or current docs.
