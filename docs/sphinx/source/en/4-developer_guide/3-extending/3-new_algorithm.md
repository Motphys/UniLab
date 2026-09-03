# Extending UniLab: New Algorithm

Algorithm work must preserve the env, config, and runner contracts. Start with
{doc}`../2-contracts/1-env_contract`, {doc}`../2-contracts/3-task_owner`, and
{doc}`../2-contracts/5-runner_lifecycle`.

## Choose The Integration Path

- Synchronous on-policy example: `src/unilab/scripts/train_rsl_rl.py`.
- Async on-policy example: `src/unilab/scripts/train_appo.py` with `APPORunner`.
- Off-policy examples: `src/unilab/scripts/train_sac.py`, `src/unilab/scripts/train_td3.py`, and
  `src/unilab/scripts/train_flashsac.py`, each with its own config tree under
  `src/unilab/conf/<algo>/`.

## Implementation Checklist

1. Put reusable learner or runner code under `uni_rl` (uni_rl repo).
2. Add Hydra config under the owning config root. A new off-policy variant should
   add its own config tree: `src/unilab/conf/<algo>/config.yaml` with the algorithm
   hyperparameters inlined, plus matching
   `src/unilab/conf/<algo>/task/<task>/<backend>.yaml` owner YAMLs.
3. If a new top-level training script is required, keep it as assembly:
   compose Hydra, call `ensure_registries()`, construct the env through the
   registry path, then hand control to the runner or trainer.
4. Keep third-party adapter naming at adapter boundaries. Do not change the
   internal `obs` plus optional `critic` env contract to match a library.
5. For async algorithms, reuse `AsyncRunner`, `ReplayBuffer` or
   `RolloutRingBuffer`, and `SharedWeightSync` instead of creating a new IPC
   lifecycle.
6. For off-policy algorithms, the CLI `--algo <algo>` selection maps to the
   per-algorithm config tree; owner YAMLs live at
   `src/unilab/conf/<algo>/task/<task>/<backend>.yaml`.

## Validation Near Risk

- Algorithm unit tests under `tests/algos/`
- IPC tests under `tests/ipc/` for async paths
- Script/config tests: `tests/scripts/test_train_script_configs.py`,
  `tests/scripts/test_train_scripts.py`

## Evidence In Repo

- Structured config dataclasses: `src/unilab/structured_configs.py`
- Training helpers: `src/unilab/training/common.py`,
  `src/unilab/training/run.py`
- Existing algorithm packages: `uni_rl` (uni_rl repo)
