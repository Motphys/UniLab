# Migrating from Isaac Lab

If you have an Isaac Lab task you want to run in UniLab, this page tells
you what stays the same, what changes, and where the sharp edges are.

## What stays the same

- Gymnasium-style env interface (`reset`, `step`, `obs/reward/info`).
- Hydra-based configuration. Most of your existing YAML can be ported with
  field-name remapping.
- The general idea of a "task" that composes scene + reward + DR + obs.
- PPO as the default algo — UniLab ships RSL-RL's PPO out of the box.

## What changes

```{list-table}
:header-rows: 1
:widths: 30 35 35

* - Isaac Lab concept
  - UniLab equivalent
  - Notes
* - `DirectRLEnv`
  - `unilab.base.np_env.NpEnv`
  - UniLab obs is always a **dict**, not a tensor.
* - `RigidBody.cfg`
  - Task-side asset import + scene composition
  - See {doc}`../../4-developer_guide/1-architecture/4-scene_composition`.
* - GPU PhysX backend
  - CPU MuJoCo / Motrix + GPU learner
  - Architectural inversion — see below.
* - `RandomizationCfg`
  - {doc}`../../4-developer_guide/2-contracts/4-dr_contract`
  - UniLab DR runs in cold-path resampling only.
* - `RewardManager` chains
  - Reward composition in env, plus
    `unilab.training.reward` bookkeeping
  - Reward terms still keyed for component-wise logging.
* - `EventCfg` event-driven hooks
  - Phase + curriculum + DR providers
  - Hooks are explicit, not implicit.
```

## The architectural inversion

Isaac Lab places the simulator on GPU and lets you batch thousands of
envs in PhysX. UniLab places the simulator on CPU (often multithread) and
batches across worker **processes**, sharing memory with a single GPU
learner.

Implications:

- **Per-env step time** in UniLab is comparable or worse than Isaac on a
  single env. **Throughput** comes from process parallelism + asynchrony
  (see `unilab.ipc.async_runner`).
- You can run on **MPS, ROCm, XPU** as the learner device — Isaac is
  CUDA-only.
- **No GPU contention** between simulator and learner — your trainer's
  memory usage is predictable.

## Step-by-step migration

1. **Audit observations.** Make sure every observation key is a vector
   you can express without GPU PhysX queries. If not, add a state
   estimator or move the query to cold path.
2. **Port the asset.** UniLab consumes MJCF as its source of truth. If
   you have USD, convert to MJCF first.
3. **Port the env.** Subclass `unilab.base.np_env.NpEnv`. Move
   reward computation into the env's `compute_reward()`.
4. **Port the YAML.** Map Isaac Lab's `EnvCfg` fields to UniLab task owner
   YAML following the table in
   {doc}`5-task_config_translation`.
5. **Port the reward.** Use the cookbook at
   {doc}`6-reward_porting`.
6. **Validate.** Train a small run, compare reward curves against your
   Isaac baseline.

## What you'll miss (and how to compensate)

- **Isaac Sim renderer.** Use Motrix's headless video export or build a
  viser scene (`unilab.visualization.viser_scene`).
- **Per-env tensor obs.** UniLab gives you dict-of-arrays; wrap with your
  own `obs_to_tensor` if you need a tensor.
- **Built-in GPU-side DR.** UniLab DR is CPU-side per process. For most
  tasks this is plenty; for extreme parallelism use more worker
  processes.

## See also

- {doc}`2-from_legged_gym`
- {doc}`3-from_rsl_rl`
- {doc}`5-task_config_translation`
- {doc}`6-reward_porting`
