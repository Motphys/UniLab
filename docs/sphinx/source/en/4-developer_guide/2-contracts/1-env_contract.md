# Env Contract

The env contract is code-owned by `src/unilab/base/base.py` and
`src/unilab/base/np_env.py`. Observation semantics are recorded in
{doc}`/adr/ADR-0005-unified-obs-critic-env-and-ipc-contract`.

## Required Shape

- `NpEnvState.obs` is `dict[str, np.ndarray]`. It is not a flat tensor.
- The required actor observation key is `obs`.
- The only optional critic-only observation key is `critic`.
- `obs_groups_spec` maps each observation group name to its flat dimension.
  Wrappers and learners use this map to size actor and critic paths.
- `reset(env_indices)` returns `(obs_dict, info_dict)` for the reset env rows.
- `step(actions)` on `NpEnv` returns `NpEnvState`; external adapters may map that
  state into third-party trainer APIs at the adapter boundary.

## Owner Responsibilities

- Env code owns MDP semantics, observation construction, rewards, termination,
  truncation, reset behavior, and final-observation handling.
- Runners and learners must not invent critic observations by concatenating
  fields outside the env owner layer.
- If a third-party library still calls critic observations "privileged", keep
  that name translation inside the adapter. Inside UniLab, the key is `critic`.

## Evidence In Repo

- Env base contract: `src/unilab/base/base.py`
- Numpy env state: `src/unilab/base/np_env.py`
- RSL-RL adapter boundary: `src/unilab/training/rsl_rl.py`
- Final observation helper: `src/unilab/base/final_observation.py`
- Tests: `tests/base/test_np_env.py`, `tests/utils/test_final_observation.py`,
  `tests/ipc/`
