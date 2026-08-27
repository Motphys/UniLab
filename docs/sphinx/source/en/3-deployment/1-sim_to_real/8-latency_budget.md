# Latency Budget

This page documents the latency controls that are visible in the repository and
the deploy-side measurements you need before hardware bring-up. Treat numeric
budgets as robot-specific measurements, not UniLab defaults.

## Latency Surfaces In Repo

| Surface | Repo evidence | What it covers |
| --- | --- | --- |
| One-step action delay | `control_config.simulate_action_latency` in locomotion and G1 motion-tracking envs | Executes the previous action instead of the current action. |
| G1 WBT observation history | `noise_config.obs_history_length` in `conf/offpolicy/task/sac/g1_wbt_obs/mujoco.yaml` | Per-term history for `gyro`, `joint_pos_rel`, `dof_vel`, and `last_actions`. |
| Sharpa tactile contact latency | `domain_rand.contact_latency` in Sharpa in-hand configs | Keeps previous tactile contact values for sampled contact channels. |
| Obs history ordering guard | `tests/scripts/test_obs_alignment_g1_wbt.py` | Asserts per-term oldest-first flatten for the G1 WBT actor obs. |

## Action Latency

For tasks that expose `control_config.simulate_action_latency`, the env applies
`last_actions` when the flag is enabled. Keep this in the selected task owner
YAML instead of adding deploy-only behavior later.

```yaml
env:
  control_config:
    simulate_action_latency: true
```

The checked-in G1 WBT owner enables this flag in
`conf/offpolicy/task/sac/g1_wbt_obs/mujoco.yaml`.

## Observation Lag And History

Observation width is a function of the owner's `noise_config`, not something a
hardware runtime may guess. For the G1 WBT owner, `obs_history_length: 5` gives
each proprioceptive term a 5-step history flattened oldest-first, while
reference terms stay single-step. See {doc}`2-g1_whole_body` for the full term
order.

Do not lag command/reference terms unless the training owner did so.

## Deploy-Side Measurements

Record these per policy tick in the hardware runtime:

1. `policy_input_timestamp`
2. source timestamps for each sensor or estimator channel
3. `policy_output_timestamp`
4. actuator command send timestamp
5. the action vector before and after clamp / smoothing

Compare the observation vector against a sim rollout built from the same task
owner YAML. If the measured pipeline needs filtering or buffering, encode the
matching behavior in the task owner and retrain, rather than adding it only on
the deploy side.

## Symptoms Of Mismatch

- Contact oscillation after enabling torque.
- Action saturation during the first few policy ticks.
- Velocity tracking drift even when the ONNX input width and observation layout
  match.

## See also

- {doc}`6-domain_randomization`
- {doc}`7-safety_layers`
- `src/unilab/dr/manager.py`
