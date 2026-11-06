# Training Diagnostics

This page collects the diagnosis paths for NaN/Inf failures, reward problems,
stalled learning, backend mismatches, and performance questions. Every key and
command below is wired in the repository; steps that need a simulator install,
a GPU, or a display are marked.

## NaN Diagnostic Chain

The NaN guard watches observations, rewards, and controls for non-finite rows
on-device and writes one state dump at an explicit host boundary. The chain:

1. Configure it under `training.nan_guard` — shown with the PPO defaults from
   `src/unilab/conf/ppo/config.yaml` (SAC/FlashSAC owners carry the same
   block):

   ```yaml
   training:
     nan_guard:
       enabled: true
       buffer_size: 100
       max_envs_to_dump: 5
       output_dir: null        # null -> /tmp/unilab/nan_dumps
       defer_validation: true
       check_interval: 32
       capture_interval: 32
   ```

   Fields are defined by `NanGuardCfg` in
   `src/unilab/training/tensor_diagnostics.py`. `capture_interval` controls how
   often a physics-state snapshot is retained (0 disables snapshots);
   `check_interval` is the poll period in deferred mode.
2. The training entrypoint attaches the guard to the env
   (`apply_env_nan_guard` in `src/unilab/training/run.py`), and
   `TorchEnv.step` calls `NanGuard.check(...)` each step. On detection,
   `NanGuard.dump(...)` writes `nan_dump_<timestamp>_step<N>.npz` (state ring
   buffer plus `meta_*` fields such as `detection_step` and `nan_env_ids`),
   copies the model file next to it, and updates the `nan_dump_latest.npz`
   symlink.
3. Replay the dump in a MuJoCo viewer (needs the `mujoco` extra and a
   display):

   ```bash
   uv run unilab-viz-nan /tmp/unilab/nan_dumps/nan_dump_latest.npz --env-index 0
   ```

   `unilab-viz-nan` is the console entry for `src/unilab/utils/nan_viz.py`
   (`pyproject.toml`); it prints the detection step and NaN env ids, then
   replays the buffered states. Backends without physics-state playback ship
   no states in the dump — the viewer prints the metadata only.

Details and the viewer walkthrough: {doc}`../2-user_guide/7-tooling/3-nan_visualizer`.
Round-trip coverage: `tests/training/test_tensor_diagnostics.py`.

## Reading `info["log"]`

Manager-Based environments place structured metrics in `info["log"]`; the
training logger forwards slash-prefixed entries to TensorBoard / W&B. The full
semantics table is in {doc}`../2-user_guide/1-training/3-logging`; the key families and their producers:

| Key | Producer | Meaning |
| --- | --- | --- |
| `reward/<term>` | `RewardManager.step_reward_extras()` (`src/unilab/managers/reward_manager.py`) | Per-term mean of `raw_value * weight` across envs, before `dt` scaling |
| `Episode_Termination/<term>` | `TerminationManager.reset()` (`src/unilab/managers/termination_manager.py`) | Count of reset env rows for which that termination term fired |
| `Episode_Metrics/<key>` | `MetricsManager` (`src/unilab/managers/metrics_manager.py`) | Completed-episode metric aggregates |

Use `Train/mean_reward` and `Train/mean_episode_length` for overall progress
and `reward/<term>` to attribute it; see {doc}`../2-user_guide/1-training/3-logging` for the distinction.

## Common Failure Modes

### Reward explosion or non-finite values

- Symptom: training aborts with `RewardManager term '<name>' returned NaN/Inf
  for environments ...` (`src/unilab/managers/reward_manager.py`), or the NaN
  guard writes a dump.
- Check: the per-term `reward/<term>` curves — the diverging term usually
  separates first. Compare its raw magnitude against its `weight` in the owner
  YAML's `reward:` block.
- Fix: retune the weight from the CLI, e.g.
  `uv run train --algo ppo --task <task> --sim mujoco
  reward.<term>.weight=-0.01` (override mechanics: the "Reward term weights"
  section of {doc}`../2-user_guide/2-algorithms/1-ppo`). Keep the smoothness/regularization
  terms (`action_rate`-style penalties) enabled; dropping them commonly
  destabilizes the action stream.

### Policy does not learn

- Symptom: `Train/mean_reward` stays flat while iterations advance.
- Check the hyperparameters against the per-algorithm reference tables
  ({doc}`../2-user_guide/2-algorithms/1-ppo` for PPO): `algo.algorithm.learning_rate`,
  `algo.algorithm.entropy_coef`, `algo.algorithm.lam`. Check
  `algo.num_envs` — it is a per-rank count, and very small counts starve the
  gradient estimate. Check observation conditioning:
  `algo.actor.obs_normalization` (the Go2 owner enables it) and whether any
  observation term emits values with unexpected scale.
- Check the reward mix: if a tracking term (`reward/tracking_*`) stays near
  zero while penalties dominate the `reward/<term>` curves, the weights are
  misbalanced, not the optimizer.

### Two backends behave differently

- Symptom: a checkpoint trained on backend A plays poorly or refuses to load
  on backend B.
- Check: the sim2sim contract. Training snapshots the policy-defining fields
  into `run_config.json`; play entrypoints validate the target config against
  that snapshot before env construction (`resolve_sim2sim_config` in
  `src/unilab/utils/sim2sim.py`). DENYLIST fields (observations, actions,
  network shapes, `ctrl_dt`, ...) must match — under the default
  `training.sim2sim_strict: true` a mismatch raises
  `CrossBackendIncompatibleError`; WARNING_LIST fields only print a warning.
  Audit the shipped pairs with `uv run scripts/audit_sim2sim_contracts.py`.
- Check capability differences per backend:
  {doc}`../3-deployment/2-sim_to_sim/6-capability_gaps`; exact support
  grades: {doc}`../5-reference/5-support_matrix`.

### NaN appears mid-run

- Follow the NaN chain above: the dump's `meta_detection_step` and
  `meta_nan_env_ids` narrow the window, and the replayed states show whether
  the pose diverged (asset/keyframe problem) or the controls blew up
  (reward/action scaling problem).
- Cross-check the `reward/<term>` curves just before the detection step to
  find the first diverging term.

### Performance and trace analysis

- Off-policy configs (SAC/FlashSAC) expose `training.trace_enabled`,
  `training.trace_output_dir`, `training.trace_thread_time`, and
  `training.trace_cuda_events` (`src/unilab/conf/sac/config.yaml`), plus
  `training.nvtx_profile_ranges` for NVTX annotations on CUDA. Traces are
  Perfetto JSON; scalar timing keys are documented in {doc}`../2-user_guide/1-training/3-logging`.
- Analyze a captured trace host-side (no GPU needed):

  ```bash
  uv run scripts/analyze_offpolicy_trace.py <trace.json> --training-e2e \
    --event collector/env_step --gap-event learner/training_e2e
  ```

  The script reports mean/median/p90/p95/p99 durations per event name;
  `--steps-per-cycle` and `--drop-first` adjust the throughput summary.
- For the production long-soak setup and its failure classification fields,
  see {doc}`../2-user_guide/1-training/7-tensor_runtime_production`.
