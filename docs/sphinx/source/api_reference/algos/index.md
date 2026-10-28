# Learning Algorithms — `unilab.rl.algos`

The RL algorithm layer lives in the `unilab` distribution under `unilab.rl`
(the former `unilab-rl` package is merged into this repository; issue #2372,
{doc}`/adr/ADR-0016-rl-reintegration-into-unilab-rl`):

- `unilab.rl.algos.appo` — APPO runner, learner, staging, worker
- `unilab.rl.algos.sac` / `unilab.rl.algos.flash_sac` — off-policy learners and runners
- `unilab.rl.runtime.offpolicy` — generic off-policy runner, worker, thread budget
- `unilab.rl.algos.common` — shared actor factory, networks, normalization, compile helpers

The PPO (RSL-RL) integration lives in `unilab.rl.adapters.rsl_rl`:
`RslRlVecEnvAdapter` plus distributed/seed helpers, and
`src/unilab/scripts/train_rsl_rl.py` drives upstream
`rsl_rl.runners.OnPolicyRunner` / `rsl_rl.algorithms:PPO` directly.

UniLab keeps the training *entrypoints* (`src/unilab/scripts/train_*.py`),
which inject environments into the RL runners through
`unilab.base.contracts.env.EnvFactory`; see `src/unilab/base/env_factory.py` for the
registry-backed adapter.

All trainers conform to a single runner contract — see
{doc}`../../en/4-developer_guide/2-contracts/5-runner_lifecycle`.
