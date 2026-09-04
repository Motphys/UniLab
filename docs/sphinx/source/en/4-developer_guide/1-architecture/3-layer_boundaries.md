# Layer Boundaries

This page is the English checklist for the architecture rule recorded in
{doc}`/adr/ADR-0001-runtime-model-and-layer-boundaries`. The canonical project
standard is {doc}`/zh_CN/4-developer_guide/0-index`.

## Owner Layers

| Layer | Owner paths | Owns |
| --- | --- | --- |
| L0 Backend | `unisim.backend` | Physics backend abstraction, backend-owned scene materialization, backend capabilities. |
| L1 Env | `src/unilab/envs/`, `src/unilab/base/np_env.py` | MDP semantics, observations, rewards, reset logic, backend-to-task adaptation. |
| L2 Config and Registry | `src/unilab/conf/`, `src/unilab/structured_configs.py`, `src/unilab/base/registry.py`, `src/unilab/training/reward.py` | Hydra composition, owner YAML identity, env/reward registration. |
| L3 Algo and IPC | `uni_rl` (unilab-rl repo), `uni_rl.ipc` (unilab-rl repo) | Learners, runners, collectors, replay and rollout buffers, weight sync. |
| L4 Scripts | `scripts/` | Entrypoint assembly only. |

## Rules

- Fix behavior at the owner layer. A training script should not carry long-term
  env, backend, reward, or algorithm business rules.
- Env code may depend on the declared `SimBackend` contract in
  `unisim.backend.base`; if shared env logic needs a new backend
  capability, add it to `SimBackend` before using it.
- Config choices should stay in Hydra owner YAMLs under `src/unilab/conf/`, not in
  Python-side backend switches.
- Asset, XML, and model metadata work belongs on init, materialization, or cache
  paths. Do not move asset parsing into `step()`, `reset()`, or runtime domain
  randomization loops.

## Evidence In Repo

- Architecture contract: {doc}`/adr/ADR-0001-runtime-model-and-layer-boundaries`
- Backend boundary: `unisim.backend.base`
- Env state contract: `src/unilab/base/np_env.py`
- Registry construction path: `src/unilab/base/registry.py`
- Training entrypoints: `src/unilab/scripts/train_rsl_rl.py`,
  `src/unilab/scripts/train_appo.py`, `src/unilab/scripts/train_sac.py`,
  `src/unilab/scripts/train_td3.py`, `src/unilab/scripts/train_flashsac.py`
