# MuJoCo executor-swap drift characterization (#1554)

BEFORE: mujoco-uni-runtime 0.5.0 executor, captured from `scripts/tools/drift_baseline/before/`.
AFTER: mjbatch executor (unilabsim fork), captured from `scripts/tools/drift_baseline/after/`.

Both captures use the identical configuration: 8 envs, 300 steps, env seed 42, action seed 1234, `cpu_ids=[0, 1, 2, 3]`, fixed pseudo-random action sequence (no policy network). Drift between the two executors is expected and accepted; this report is a regression reference, not a pass/fail gate.

- BEFORE dir: `/home/user/ws/unilabsim2/UniLab/scripts/tools/drift_baseline/before`
- AFTER dir: `/home/user/ws/unilabsim2/UniLab/scripts/tools/drift_baseline/after`

## Go2JoystickFlat

| array | shape | max abs diff | mean abs diff | ref max abs | first divergence step | max abs diff at first divergence |
|---|---|---|---|---|---|
| `actions` | (300, 8, 12) | 0.000000e+00 | 0.000000e+00 | 1.000000e+00 | -1 | 0.000000e+00 |
| `obs/critic` | (300, 8, 52) | 2.768040e-03 | 4.664116e-06 | 1.162892e+01 | 1 | 1.633167e-05 |
| `obs/obs` | (300, 8, 49) | 2.768040e-03 | 4.869826e-06 | 1.162892e+01 | 1 | 1.633167e-05 |
| `obs_init/critic` | (8, 52) | 0.000000e+00 | 0.000000e+00 | 1.000000e+00 | -1 | 0.000000e+00 |
| `obs_init/obs` | (8, 49) | 0.000000e+00 | 0.000000e+00 | 1.000000e+00 | -1 | 0.000000e+00 |
| `qpos` | (300, 8, 19) | 1.286268e-04 | 1.379656e-06 | 2.227637e+00 | 1 | 4.768372e-07 |
| `qpos_init` | (8, 19) | 0.000000e+00 | 0.000000e+00 | 1.500000e+00 | -1 | 0.000000e+00 |
| `qvel` | (300, 8, 18) | 2.768040e-03 | 1.204221e-05 | 1.162892e+01 | 1 | 1.633167e-05 |
| `qvel_init` | (8, 18) | 0.000000e+00 | 0.000000e+00 | 5.306464e-01 | -1 | 0.000000e+00 |
| `reward` | (300, 8) | 4.128367e-05 | 1.929190e-07 | 1.274497e-01 | 1 | 1.303852e-08 |
| `terminated` | (300, 8) | 0.000000e+00 | 0.000000e+00 | 1.000000e+00 | -1 | 0.000000e+00 |
| `truncated` | (300, 8) | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | -1 | 0.000000e+00 |

First divergence step (any array): **1**

## Go2WJoystickFlat

| array | shape | max abs diff | mean abs diff | ref max abs | first divergence step | max abs diff at first divergence |
|---|---|---|---|---|---|
| `actions` | (300, 8, 16) | 0.000000e+00 | 0.000000e+00 | 1.000000e+00 | -1 | 0.000000e+00 |
| `obs/critic` | (300, 8, 72) | 3.289003e+01 | 1.981591e-01 | 4.543000e+01 | 0 | 5.564094e-05 |
| `obs/obs` | (300, 8, 53) | 1.065044e+01 | 8.987349e-02 | 2.687317e+01 | 0 | 5.564094e-05 |
| `obs_init/critic` | (8, 72) | 0.000000e+00 | 0.000000e+00 | 1.000000e+00 | -1 | 0.000000e+00 |
| `obs_init/obs` | (8, 53) | 0.000000e+00 | 0.000000e+00 | 1.000000e+00 | -1 | 0.000000e+00 |
| `qpos` | (300, 8, 23) | 1.096748e+00 | 2.048076e-02 | 1.769883e+01 | 0 | 2.784654e-07 |
| `qpos_init` | (8, 23) | 0.000000e+00 | 0.000000e+00 | 1.500000e+00 | -1 | 0.000000e+00 |
| `qvel` | (300, 8, 22) | 1.065044e+01 | 2.105295e-01 | 2.687317e+01 | 0 | 5.564094e-05 |
| `qvel_init` | (8, 22) | 0.000000e+00 | 0.000000e+00 | 5.306464e-01 | -1 | 0.000000e+00 |
| `reward` | (300, 8) | 8.046474e-02 | 4.191729e-03 | 7.397851e-01 | 0 | 1.490116e-08 |
| `terminated` | (300, 8) | 0.000000e+00 | 0.000000e+00 | 1.000000e+00 | -1 | 0.000000e+00 |
| `truncated` | (300, 8) | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | -1 | 0.000000e+00 |

First divergence step (any array): **0**

