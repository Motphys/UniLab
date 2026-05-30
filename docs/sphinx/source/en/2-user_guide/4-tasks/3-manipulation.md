# Manipulation

Manipulation tasks live in `src/unilab/envs/manipulation/` and the Go2 arm
manip-loco env lives in `src/unilab/envs/locomotion/go2_arm/`.

## In-Hand

- `allegro_inhand` and `allegro_inhand_grasp` have MuJoCo and Motrix PPO owners.
- `sharpa_inhand`, `sharpa_inhand_grasp`, and the `hora` profile for
  `sharpa_inhand` are MuJoCo owner paths in the current configs.

```bash
uv run train --algo ppo --task allegro_inhand --sim mujoco
uv run train --algo ppo --task allegro_inhand --sim motrix training.no_play=true
uv run train --algo ppo --task sharpa_inhand --sim mujoco --profile hora training.no_play=true
```

HORA student distillation is configured by
`conf/hora_distill/task/sharpa_inhand/mujoco.yaml`; it is not currently exposed
as a separate top-level CLI route.

## Mobile Manipulation

`go2_arm_manip_loco` is the committed Go2 + Airbot owner path:

```bash
uv run train --algo ppo --task go2_arm_manip_loco --sim mujoco training.no_play=true
```

See {doc}`../8-manipulation/1-dexterous_inhand` and
{doc}`../8-manipulation/2-manip_loco` for task-specific notes.
