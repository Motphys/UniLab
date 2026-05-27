# Migrating from Legged Gym

Legged Gym was the GPU-resident PPO template that taught the field how to
train quadrupeds. Its core ideas — joystick command spaces, terrain
curricula, RSL-RL PPO — live on inside UniLab. Migration is therefore
mostly mechanical.

## Direct equivalences

| Legged Gym | UniLab |
|---|---|
| `LeggedRobot` env class | {py:class}`unilab.envs.locomotion.common.base` |
| `compute_observations()` | env-side obs builder + {py:mod}`unilab.base.observations` |
| `_reward_*` methods | env's `compute_reward()` + reward term registry |
| `command_ranges` | task owner YAML's `commands` block |
| Terrain curriculum | {doc}`../../user_guide/terrain/procedural` |
| RSL-RL PPO | {py:mod}`unilab.algos.torch.rsl_rl_ppo` |

## What's new

- **Two backends.** Legged Gym is Isaac Gym only; UniLab gives you MuJoCo
  + Motrix. Pick one (or both) before porting; see
  {doc}`../sim_to_sim/why_switch`.
- **Async collection.** Legged Gym collects on-GPU synchronously; UniLab's
  APPO ({py:mod}`unilab.algos.torch.appo`) decouples collectors from
  learner. If wall-clock matters, port to APPO once your reward parity is
  established.
- **Hardware deployment.** Legged Gym → real-world deployment is a
  hand-rolled story per lab. UniLab gives you the
  {doc}`../sim_to_real/overview` pipeline as a first-class artifact.

## Migration checklist

1. Copy your URDF / MJCF assets under `assets/robots/<robot>/`.
2. Create a task module under `src/unilab/envs/locomotion/<robot>/`.
3. Mirror your reward terms; keep the same names so reward parity is
   diff-able.
4. Translate command sampling — Legged Gym's `_resample_commands` becomes
   a curriculum provider in UniLab.
5. Translate terrain — Legged Gym's heightfield generator has a UniLab
   counterpart at {py:mod}`unilab.terrains.heightfield_terrains`.

## Validation gate

Before deleting your Legged Gym checkout, train a Go2-equivalent task in
UniLab on flat ground. Reward should converge to the same plateau within
factor of 2 in sample efficiency. If not, there's a reward or DR mismatch
— see {doc}`../sim_to_sim/reward_parity_checks`.

## See also

- {doc}`from_isaac_lab`
- {doc}`from_rsl_rl`
- {doc}`task_config_translation`
