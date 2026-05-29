# Extending UniLab: New Algorithm

Algorithm work must preserve the env, config, and runner contracts. Start with
{doc}`../contracts/env_contract`, {doc}`../contracts/task_owner`, and
{doc}`../contracts/runner_lifecycle`.

## Choose The Integration Path

- Synchronous on-policy examples: `scripts/train_rsl_rl.py` and
  `scripts/train_mlx_ppo.py`.
- Async on-policy example: `scripts/train_appo.py` with `APPORunner`.
- Off-policy examples: `scripts/train_offpolicy.py` with SAC, TD3, and
  FlashSAC configs under `conf/offpolicy/`.

## Implementation Checklist

1. Put reusable learner or runner code under `src/unilab/algos/`.
2. Add Hydra config under the owning config root. A new off-policy variant should
   add `conf/offpolicy/algo/<algo>.yaml` and matching
   `conf/offpolicy/task/<algo>/<task>/<backend>.yaml` owner YAMLs.
3. If a new top-level training script is required, keep it as assembly:
   compose Hydra, call `ensure_registries()`, construct the env through the
   registry path, then hand control to the runner or trainer.
4. Keep third-party adapter naming at adapter boundaries. Do not change the
   internal `obs` plus optional `critic` env contract to match a library.
5. For async algorithms, reuse `AsyncRunner`, `ReplayBuffer` or
   `RolloutRingBuffer`, and `SharedWeightSync` instead of creating a new IPC
   lifecycle.
6. For off-policy algorithms, keep `algo=<algo>` aligned with
   `task=<algo>/<task>/<backend>`; `assert_offpolicy_task_choice_matches_algo`
   enforces this guard.

## Validation Near Risk

- Algorithm unit tests under `tests/algos/`
- IPC tests under `tests/ipc/` for async paths
- Script/config tests: `tests/scripts/test_train_script_configs.py`,
  `tests/scripts/test_train_scripts.py`

## Evidence In Repo

- Structured config dataclasses: `src/unilab/structured_configs.py`
- Training helpers: `src/unilab/training/common.py`,
  `src/unilab/training/run.py`
- Existing algorithm packages: `src/unilab/algos/torch/`,
  `src/unilab/algos/mlx/`
