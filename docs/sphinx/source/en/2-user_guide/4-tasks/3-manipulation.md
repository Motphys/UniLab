# Manipulation

Manipulation tasks live in `src/unilab/tasks/manipulation/`.

## In-Hand

- `allegro_inhand` and `allegro_inhand_grasp` have MuJoCo and Motrix PPO owners.

```bash
uv run train --algo ppo --task allegro_inhand --sim mujoco
uv run train --algo ppo --task allegro_inhand --sim motrix training.no_play=true
```

## Platform Balancing

`stewart_balance` is a 6-DOF parallel (Stewart) platform that balances a free
ball on its top plate. The policy commands a 2-D plate tilt (roll, pitch); an
inverse-kinematics step converts the commanded plate pose into the six prismatic
leg lengths that the position actuators track. The reward combines centering,
zero-velocity progress and a stillness bonus, with a fall penalty; an episode ends
on a fall or on sustained-still success.

The base is welded to the world. Motrix is the validated training backend; the
mujoco owner constructs and steps, but its stiff closed-loop solver is not yet
training-stable under load.

```bash
uv run train --algo ppo --task stewart_balance --sim motrix training.no_play=true
```
