# Layer Boundaries

This page is the English checklist for the architecture rule recorded in
{doc}`/adr/ADR-0001-runtime-model-and-layer-boundaries`. The canonical project
standard is {doc}`/zh_CN/developer_guide/development-standard`.

## Owner Layers

| Layer | Owner paths | Owns |
| --- | --- | --- |
| L0 Backend | `src/unilab/base/backend/` | Physics backend abstraction, backend-owned scene materialization, backend capabilities. |
| L1 Env | `src/unilab/envs/`, `src/unilab/base/np_env.py` | MDP semantics, observations, rewards, reset logic, backend-to-task adaptation. |
| L2 Config and Registry | `conf/`, `src/unilab/structured_configs.py`, `src/unilab/base/registry.py`, `src/unilab/training/reward.py` | Hydra composition, owner YAML identity, env/reward registration. |
| L3 Algo and IPC | `src/unilab/algos/`, `src/unilab/ipc/` | Learners, runners, collectors, replay and rollout buffers, weight sync. |
| L4 Scripts | `scripts/` | Entrypoint assembly only. |

## Rules

- Fix behavior at the owner layer. A training script should not carry long-term
  env, backend, reward, or algorithm business rules.
- Env code may depend on the declared `SimBackend` contract in
  `src/unilab/base/backend/base.py`; if shared env logic needs a new backend
  capability, add it to `SimBackend` before using it.
- Config choices should stay in Hydra owner YAMLs under `conf/`, not in
  Python-side backend switches.
- Asset, XML, and model metadata work belongs on init, materialization, or cache
  paths. Do not move asset parsing into `step()`, `reset()`, or runtime domain
  randomization loops.

## Evidence In Repo

- Architecture contract: {doc}`/adr/ADR-0001-runtime-model-and-layer-boundaries`
- Backend boundary: `src/unilab/base/backend/base.py`
- Env state contract: `src/unilab/base/np_env.py`
- Registry construction path: `src/unilab/base/registry.py`
- Training entrypoints: `scripts/train_rsl_rl.py`, `scripts/train_mlx_ppo.py`,
  `scripts/train_appo.py`, `scripts/train_offpolicy.py`
