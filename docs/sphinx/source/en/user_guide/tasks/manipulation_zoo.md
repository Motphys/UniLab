# Manipulation Zoo

Manipulation tasks live in `src/unilab/envs/manipulation/` and the Go2 arm
manip-loco env lives in `src/unilab/envs/locomotion/go2_arm/`.

## In-Hand

- `allegro_inhand` and `allegro_inhand_grasp` have MuJoCo and Motrix PPO owners.
- `sharpa_inhand`, `sharpa_inhand_grasp`, and `sharpa_inhand/mujoco_hora` are
  MuJoCo owner paths in the current configs.

```bash
uv run scripts/train_rsl_rl.py task=allegro_inhand/mujoco
uv run scripts/train_rsl_rl.py task=allegro_inhand/motrix training.no_play=true
uv run scripts/train_rsl_rl.py task=sharpa_inhand/mujoco_hora training.no_play=true
uv run scripts/train_hora_distill.py task=sharpa_inhand/mujoco
```

## Mobile Manipulation

`go2_arm_manip_loco` is the committed Go2 + Airbot owner path:

```bash
uv run scripts/train_rsl_rl.py task=go2_arm_manip_loco/mujoco training.no_play=true
```

See {doc}`../manipulation/dexterous_inhand` and
{doc}`../manipulation/manip_loco` for task-specific notes.
