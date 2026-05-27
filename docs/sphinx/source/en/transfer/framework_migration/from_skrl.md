# Migrating from skrl

skrl's strength is algorithm breadth. UniLab focuses on a curated set
(PPO, SAC, TD3 with several optimized variants) but adds a real-hardware
deployment path.

## Mapping skrl concepts to UniLab

| skrl | UniLab |
|---|---|
| `Agent` (PPO, SAC, …) | {py:mod}`unilab.algos.torch.*` |
| `RolloutMemory` | {py:mod}`unilab.ipc.rollout_ring_buffer` |
| `ReplayMemory` | {py:mod}`unilab.ipc.replay_buffer` |
| `Trainer` | {py:mod}`unilab.training.run` |
| `Wrapper` for env | `NpEnv` subclassing |

## What to expect

- **No algorithm parity** for niche algos (CQL, IQL, etc.) — UniLab
  intentionally focuses on a few highly optimized actor-critic variants.
- **Different runner lifecycle.** skrl's monolithic trainer becomes a
  collector + learner pair connected by shared memory. See
  {doc}`../../developer_guide/contracts/runner_lifecycle`.
- **Different env interface.** skrl tolerates many env styles. UniLab
  insists on `NpEnv` + dict obs.

## Migration checklist

1. Decide which UniLab algorithm best matches your skrl agent.
2. Port the env into `NpEnv` form.
3. Convert hyperparameter YAML into Hydra groups under `conf/<algo>/<task>/`.
4. Validate reward parity.
