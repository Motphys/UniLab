# Go2 nominal cross-engine evaluation (v2)

This is a frozen-policy comparison. Do not retrain or change rewards for this experiment.
Use the same `student_mapping_v1_500.pt` and its matching ONNX export.

Changes are opt-in through `--sim2sim-nominal`; ordinary training is unchanged.
Both engines use home reset, zero initial velocities, zero observation noise,
friction 1.0, added payload 0, actuator kp 35 and kd 0, and the original passive
joint damping 0.5. Policy period is 0.02 s with two 0.01 s physics steps.
MuJoCo initial timeout counters are explicitly cleared, so the first episode is
1000 steps rather than a seed-dependent remaining training episode.

Important sensor contract: the existing policy was trained on negative MJCF
framezaxis (the site z-axis expressed in world coordinates), despite the name
projected_gravity. Nominal Genesis now preserves that definition. Replacing it
with conventional body-frame projected gravity would change this policy input.
Do not alter the training definition for this frozen checkpoint.
Genesis velocity diagnostics include the IMU offset via omega cross r.

Run a fresh comparison from the project root, keeping previous results:

```bash
out="runs/rma/evaluations/nominal_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$out"
set -o pipefail
uv run --no-sync python scripts/evaluate_rma.py --policy student --checkpoint runs/rma/student_mapping_v1_500.pt --task go2_joystick_flat/rma_forward --num-envs 1 --steps 1000 --seed 123 --command 0.5 0 0 --diagnostics --sim2sim-nominal --trace-dir "$out/mujoco" 2>&1 | tee "$out/mujoco.log"
scripts/run_genesis_go2.sh --steps 1000 --seed 123 --command 0.5 0 0 --sim2sim-nominal --output "$out/genesis.csv" 2>&1 | tee "$out/genesis.log"
uv run --no-sync python scripts/compare_sim2sim_traces.py --mujoco-trace "$out/mujoco" --genesis-csv "$out/genesis.csv" --output-dir "$out/plots"
```

Plot meanings:
- velocity_comparison.png: forward/lateral/yaw tracking against the command.
- action_near_bounds.png: fraction with normalized action magnitude >= 0.95;
  this is not a torque or joint-limit saturation metric.
- genesis_stability.png: Genesis tilt and height over time.
- comparison.json: both metrics computed from post-step velocity to avoid mixing
  the MuJoCo evaluator pre-step summary with Genesis post-step samples.

The original seed123 validation lives in
`runs/rma/evaluations/sim2sim_nominal_v2_seed123`. Its MuJoCo raw log was
captured before the mode label was renamed to v2; actual kd=0 and initial
counter clearing were already active. Both traces end at step 1000, no falls.

Limitations: matching settings does not make contact solvers or integrations
identical; no universal acceptance threshold is claimed. With home reset and
noise off, different seeds largely repeat the same condition. They are not a
robustness test over varied initial states or physical parameters. Validate
those separately after auditing the interfaces.
