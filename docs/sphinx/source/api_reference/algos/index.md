# Learning Algorithms — moved to `uni_rl`

The RL algorithm layer moved out of the `unilab` package into the
independently released **uni_rl** package (distribution name `unilab-rl`,
published on PyPI; issue #1480):

- `uni_rl.algos.rsl_rl` / `uni_rl.algos.rsl_rl_ppo` / `uni_rl.algos.rsl_rl_runtime` — PPO (RSL-RL) integration
- `uni_rl.algos.appo` — APPO runner, learner, staging, worker
- `uni_rl.algos.fast_sac` / `uni_rl.algos.fast_td3` / `uni_rl.algos.flash_sac` — off-policy learners and runners
- `uni_rl.offpolicy` — generic off-policy runner, worker, thread budget
- `uni_rl.algos.him_ppo` — HIM-PPO
- `uni_rl.algos.hora` — HORA models, trainers, and distillation
- `uni_rl.algos.common` — shared actor factory, networks, normalization, compile helpers

UniLab keeps the training *entrypoints* (`src/unilab/scripts/train_*.py`),
which inject environments into uni_rl runners through
`uni_rl.env_contract.EnvFactory`; see `src/unilab/base/env_factory.py` for the
registry-backed adapter.

All trainers conform to a single runner contract — see
{doc}`../../en/4-developer_guide/2-contracts/5-runner_lifecycle`.
