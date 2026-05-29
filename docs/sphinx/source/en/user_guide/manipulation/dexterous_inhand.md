# Dexterous In-Hand Manipulation

This page covers the checked-in Allegro and Sharpa in-hand manipulation paths.
Select backends through task owner YAMLs such as `task=allegro_inhand/mujoco`;
do not override `training.sim_backend` alone.

## Allegro

Allegro rotation uses the registered env `AllegroInhandRotation`. The rotation
owner is `allegro_inhand`, and grasp-cache generation uses
`allegro_inhand_grasp`.

Owner evidence:

- `conf/ppo/task/allegro_inhand/mujoco.yaml`
- `conf/ppo/task/allegro_inhand/motrix.yaml`
- `conf/ppo/task/allegro_inhand_grasp/mujoco.yaml`
- `conf/ppo/task/allegro_inhand_grasp/motrix.yaml`
- `conf/appo/task/allegro_inhand/mujoco.yaml`
- `conf/appo/task/allegro_inhand/motrix.yaml`

Generate a grasp cache, then train rotation:

```bash
uv run scripts/train_rsl_rl.py task=allegro_inhand_grasp/mujoco training.no_play=true
uv run scripts/train_rsl_rl.py task=allegro_inhand/mujoco training.no_play=true
```

Motrix owner YAMLs also exist for the PPO Allegro paths:

```bash
uv run scripts/train_rsl_rl.py task=allegro_inhand_grasp/motrix training.no_play=true
uv run scripts/train_rsl_rl.py task=allegro_inhand/motrix training.no_play=true
```

## Sharpa

Sharpa rotation uses the registered env `SharpaInhandRotation`. Current checked
in training paths are MuJoCo owner paths.

Owner evidence:

- `conf/ppo/task/sharpa_inhand/mujoco.yaml`
- `conf/ppo/task/sharpa_inhand/mujoco_hora.yaml`
- `conf/ppo/task/sharpa_inhand_grasp/mujoco.yaml`
- `conf/appo/task/sharpa_inhand/mujoco.yaml`
- `conf/appo/task/sharpa_inhand/mujoco_hora.yaml`
- `conf/hora_distill/task/sharpa_inhand/mujoco.yaml`

Generate caches by scale, train a teacher, then distill a student:

```bash
uv run scripts/train_rsl_rl.py task=sharpa_inhand_grasp/mujoco \
  'env.domain_rand.scale_list=[0.5]' \
  training.no_play=true

uv run scripts/train_rsl_rl.py task=sharpa_inhand/mujoco_hora training.no_play=true
uv run scripts/train_hora_distill.py task=sharpa_inhand/mujoco
```

For the category-level task page, see {doc}`../tasks/manipulation`.
