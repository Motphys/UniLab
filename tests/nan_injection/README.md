# NaN guard injection validation scripts

These scripts manually inject `NaN` into env state at well-defined points and
assert that the `NanGuard` (added in #584) detects + dumps. They are NOT pytest
tests and NOT part of CI/CD — see [conftest.py](conftest.py).

## Why not in CI

- Each case launches a real `OnPolicyRunner` / `HIMOnPolicyRunner` and trains
  for one iteration. ~10–30s per case, ~3 min total for the full sweep.
- Requires a real MuJoCo install + GPU or CPU torch. CI workers don't have
  this stack provisioned.
- The validation pattern (one-shot monkey-patch of `update_state` /
  `apply_action`) is a debugging tool, not a regression contract.

## Files

| Script | Purpose | Runtime | Cases |
|---|---|---|---|
| [proto_nan_inject.py](proto_nan_inject.py) | Stub-env smoke test for the inject + dump pattern. | ~5s | 4 |
| [proto_him_ppo_inject.py](proto_him_ppo_inject.py) | HIM-PPO inject prototype. | ~30s | 3 |
| [stage2_nan_inject.py](stage2_nan_inject.py) | Single-process algorithms, real runner. | ~3 min | 6 (PPO×3, HIM-PPO×3) |
| [stage3_nan_inject.py](stage3_nan_inject.py) | Multi-process algorithms, static wiring check. | <1s | 4 (APPO/SAC/TD3/FlashSAC) |

## Running

```bash
cd /path/to/UniLab
.venv/bin/python tests/nan_injection/stage2_nan_inject.py
.venv/bin/python tests/nan_injection/stage3_nan_inject.py
```

Stage 2 patches `env.update_state` (for `obs`/`reward`) or `env.apply_action`
(for `ctrl`) one-shot at call `K`, runs `runner.learn(num_learning_iterations=1)`,
then asserts:
- `guard._dumped == True`
- A `nan_dump_*.npz` file exists in the output dir
- `meta_detection_step` in the dump matches the expected step

For obs/reward (checked after `step_counter += 1`), expected step == K.
For ctrl (checked before increment), expected step == K - 1.

Stage 3 verifies the 3-piece wiring (train script reads cfg → collector worker
calls `env.set_nan_guard()` in subprocess → hydra config has `training.nan_guard`
block) for the multi-process algorithms. End-to-end runtime testing for these
requires a manual env edit (see the recipe printed by the script).
