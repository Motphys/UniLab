# Migrating from Legged Gym

Legged Gym was the GPU-resident PPO template that taught the field how to
train quadrupeds. Its core ideas — joystick command spaces, terrain
curricula, RSL-RL PPO — live on inside UniLab. Migration is therefore
mostly mechanical.

## Direct equivalences

| Legged Gym | UniLab |
|---|---|
| `LeggedRobot` env class | `unilab.envs.locomotion.common.base` |
| `compute_observations()` | env-side obs builder + `unilab.base.observations` |
| `_reward_*` methods | env's `compute_reward()` + reward term registry |
| `command_ranges` | task owner YAML's `commands` block |
| Terrain curriculum | {doc}`../../2-user_guide/6-terrain/1-procedural` |
| RSL-RL PPO | `unilab.algos.torch.rsl_rl_ppo` |

## What's new

- **Two backends.** Legged Gym is Isaac Gym only; UniLab gives you MuJoCo
  + Motrix. Pick one (or both) before porting; see
  {doc}`../2-sim_to_sim/1-backend_swap`.
- **Async collection.** Legged Gym collects on-GPU synchronously; UniLab's
  APPO (`unilab.algos.torch.appo`) decouples collectors from
  learner. If wall-clock matters, port to APPO once your reward parity is
  established.
- **Hardware deployment.** Legged Gym → real-world deployment is a
  hand-rolled story per lab. UniLab gives you the
  {doc}`../1-sim_to_real/1-overview` pipeline as a first-class artifact.

## Migration checklist

1. Copy your URDF / MJCF assets under `src/unilab/assets/robots/<robot>/`.
2. Create a task module under `src/unilab/envs/locomotion/<robot>/`.
3. Mirror your reward terms; keep the same names so reward parity is
   diff-able.
4. Translate command sampling — Legged Gym's `_resample_commands` becomes
   a curriculum provider in UniLab.
5. Translate terrain — Legged Gym's heightfield generator has a UniLab
   counterpart at `unilab.terrains.heightfield_terrains`.

## Validation gate

Before deleting your Legged Gym checkout, train a Go2-equivalent task in
UniLab on flat ground and compare the reward-term traces against the source
implementation. If the traces diverge before the policy learns, there is a
reward, command, reset, or DR mismatch; see {doc}`../2-sim_to_sim/4-reward_parity`.

## See also

- {doc}`1-from_isaac_lab`
- {doc}`3-from_rsl_rl`
- {doc}`5-task_config_translation`
