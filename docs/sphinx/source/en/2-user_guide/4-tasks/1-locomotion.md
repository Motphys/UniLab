# Locomotion

UniLab keeps a small reference/conformance locomotion set and the shared
Manager-Based runtime. Unitree production variants live in the downstream
`unitree_rl_unilab` package.

## Core reference tasks

- Go2 joystick: `go2_joystick_flat`
- G1 walking: `g1_walk_flat`
- G1 motion tracking reference profiles: see {doc}`2-motion_tracking`

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco
uv run train --algo ppo --task g1_walk_flat --sim mujoco
uv run train --algo sac --task g1_walk_flat --sim mujoco
```

Check the core support matrix for evidence grade by entrypoint, task owner, and
backend: {doc}`../../5-reference/5-support_matrix`.

## Unitree ecosystem tasks

Unitree production variants are maintained in
[unitree_rl_unilab](https://github.com/unilabsim/unitree_rl_unilab), which
depends only on the published UniLab/UniRL distributions. Its support evidence
and task documentation are owned by that repository and are not part of
UniLab's core support matrix.
