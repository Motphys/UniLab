# Runner Lifecycle

Runner code owns training lifecycle. Scripts compose Hydra config and start the
right runner; they should not create a second collector/learner protocol.

## Shared Entrypoint Flow

The training scripts follow the same high-level sequence:

1. Compose Hydra config from the algorithm config root and the selected task
   owner YAML.
2. Call `ensure_registries()`.
3. Construct the env through `registry.make(...)` or the shared
   `create_env(...)` helper.
4. Build the algorithm runner or trainer.
5. Train, checkpoint, play back if requested, and close owned resources.

## Runtime-Specific Owners

- `scripts/train_rsl_rl.py` uses `RslRlVecEnvWrapper` and RSL-RL's
  `OnPolicyRunner`.
- `scripts/train_mlx_ppo.py` uses the MLX PPO trainer path.
- `scripts/train_appo.py` uses `APPORunner`, `RolloutRingBuffer`, and
  `SharedWeightSync`.
- `scripts/train_offpolicy.py` uses off-policy runners with `ReplayBuffer` and
  `SharedWeightSync`.
- `AsyncRunner` owns collector process lifecycle and shared-resource cleanup for
  async runners.

## Rules

- Do not bypass `AsyncRunner.close()` semantics for async collectors.
- Do not patch env observation or critic semantics inside runner code; preserve
  the `obs` plus optional `critic` contract.
- Use `src/unilab/training/run.py` for shared log-root, checkpoint, and playback
  resolution helpers instead of copying those rules into scripts.

## Evidence In Repo

- Shared training helpers: `src/unilab/training/common.py`,
  `src/unilab/training/run.py`
- Async lifecycle: `src/unilab/ipc/async_runner.py`
- Runner tests: `tests/algos/test_appo_runner.py`,
  `tests/algos/test_offpolicy_runner.py`, `tests/ipc/test_async_runner.py`
