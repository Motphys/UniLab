# Latency Budget

This page documents the latency controls that are visible in the repository and
the deploy-side measurements you need before hardware bring-up. Treat numeric
budgets as robot-specific measurements, not UniLab defaults.

## Latency Surfaces In Repo

| Surface | Repo evidence | What it covers |
| --- | --- | --- |
| One-step action delay | `control_config.simulate_action_latency` in locomotion and G1 motion-tracking envs | Executes the previous action instead of the current action. |
| G1 WBT observation history | `noise_config.obs_history_length` and `scripts/deploy/export_deploy_config.py` | Exports per-term `obs_layout` history for `gyro`, `joint_pos_rel`, `dof_vel`, and `last_actions`. |
| Sharpa tactile contact latency | `domain_rand.contact_latency` in Sharpa in-hand configs | Keeps previous tactile contact values for sampled contact channels. |
| Deploy-side ONNX contract check | `scripts/deploy/sim_prototype.py` | Validates `obs_layout`, `obs_dim`, ONNX input width, clipping, and EMA action smoothing for the G1 WBT path. |

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

The G1 WBT deployment helpers do not guess observation width. They export a
schema with `obs_layout`, per-term `history_length`, and `obs_dim`; the
prototype then assembles the same layout and refuses mismatches.

```bash
uv run scripts/deploy/export_deploy_config.py \
  --output logs/deploy/deploy_config.yaml

uv run scripts/deploy/sim_prototype.py \
  --onnx runs/<run>/policy.onnx \
  --config logs/deploy/deploy_config.yaml \
  --motion logs/deploy/dance1.bin
```

Do not lag command/reference terms unless the training owner did so. In the G1
WBT schema, reference terms stay single-step while proprioceptive terms carry
history.

## Deploy-Side Measurements

Record these per policy tick in the hardware runtime:

1. `policy_input_timestamp`
2. source timestamps for each sensor or estimator channel
3. `policy_output_timestamp`
4. actuator command send timestamp
5. the action vector before and after clamp / smoothing

Compare the observation vector against a sim rollout built from the same
`deploy_config.yaml`. If the measured pipeline needs filtering or buffering,
encode the matching behavior in the task owner and re-export the deployment
artifacts.

## Symptoms Of Mismatch

- Contact oscillation after enabling torque.
- Action saturation during the first few policy ticks.
- Velocity tracking drift even when the ONNX input width and observation layout
  match.

## See also

- {doc}`6-domain_randomization`
- {doc}`7-safety_layers`
- `src/unilab/dr/manager.py`
