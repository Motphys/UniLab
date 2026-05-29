# Sim-to-Real Troubleshooting

A cookbook of symptoms → likely causes → fixes. When a deployment goes
sideways, start here.

## High-frequency joint oscillation / "buzz"

| Likely cause | Check | Fix |
|---|---|---|
| PD gains too high vs trained | Compare driver Kp/Kd against owner YAML | Match training values; or retrain with realistic Kp/Kd DR |
| Action latency too low in training | Sweep `torque_delay_ms` in DR | Retrain with measured-latency × 1.5 |
| Velocity noise too low | Compare encoder σ in sim vs hardware | Increase `joint_vel_noise_std` in DR |

## Drift while standing / walking

| Likely cause | Check | Fix |
|---|---|---|
| State estimator velocity bias | Log `base_lin_vel` vs ground truth (motion capture) | Tune KF or switch to HIM-PPO |
| IMU bias not calibrated | Robot static, check `gyro_bias` | Run 30-second calibration before policy start |
| Foot contact misclassified | Check contact event timestamps | Hysteresis on contact force threshold |

## Policy succeeds in sim, falls on hardware immediately

Almost always one of:

1. **Joint order swapped.** Inspect `policy.onnx` input width and the joint
   order in your motor driver. Use `unilab-export-scene` to dump the
   training joint order.
2. **Action scale unit mismatch.** Policy outputs unscaled values; the
   driver expects rad, but you fed it normalized [-1, 1]. Apply
   the `action_scale` / default-angle convention from `deploy_config.yaml`
   before sending targets to the driver.
3. **Observation layout mismatch.** Compare `deploy_config.yaml` `obs_layout`
   against the training owner and validate it with `scripts/deploy/sim_prototype.py`
   before running on hardware.

## Cube drops in Allegro / Sharpa inhand

| Likely cause | Check | Fix |
|---|---|---|
| Friction mismatch | Real cube vs sim μ | Sweep friction DR wider, retrain |
| Grasp distribution mismatch | Log operator grip poses | Augment grasp generator |
| Pose estimator latency | Measure vision pipeline ms | Add observation lag DR |

## Policy succeeds in MuJoCo but fails in Motrix (or vice versa)

That's a **sim-to-sim** problem, not sim-to-real. See
{doc}`../sim_to_sim/contact_and_friction_alignment` and
{doc}`../sim_to_sim/reward_parity`.

## What to capture before opening a bug report

When the lab WhatsApp blows up at 11 pm, save the following so the
investigation tomorrow takes 30 minutes instead of 4 hours:

- Full hardware trace (`obs / action / wall_clock` for the entire run).
- Sim-side YAML used to train: `runs/<run>/config.yaml`.
- `policy.onnx` and, for the G1 WBT path, `deploy_config.yaml`.
- One sim rollout video using **the same** seed: `eval --seed <same>
  --render-mode record`.
- A `git diff` between the run's commit and `main` if any.

## See also

- {doc}`overview`
- {doc}`safety_layers`
- {doc}`latency_budget`
