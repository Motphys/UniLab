# Configuration

Domain randomization is configured inside the selected task owner YAML, usually
under `env.domain_rand`. Use `--task` and `--sim` to select backend-specific
behavior first, then override fields inside that selected owner.

```bash
uv run train --algo ppo --task g1_walk_flat --sim mujoco \
  env.domain_rand.randomize_gravity=true \
  'env.domain_rand.gravity_range=[[0.0,0.0,-10.5],[0.0,0.0,-8.5]]'
```

Common lifecycle boundaries:

- Init-lifecycle items change model identity or geometry and must run during
  env/backend initialization.
- Reset-lifecycle items perturb state or model parameters at reset through a
  backend-supported payload.
- Interval-lifecycle items apply perturbations between steps.

The detailed task status and field semantics are in {doc}`0-index`.

Domain randomization is split by lifecycle: init, reset, and interval. The
manager path is `src/unilab/dr/manager.py`; task providers live near the env
owners, and backend capabilities are declared through
`src/unilab/base/backend/base.py`.

## Reset Gravity

Use `--sim mujoco` when enabling gravity reset randomization; Motrix does not
advertise the same gravity capability in the current backend.

```bash
uv run train --algo ppo --task g1_walk_flat --sim mujoco \
  env.domain_rand.randomize_gravity=true \
  'env.domain_rand.gravity_range=[[0.0,0.0,-10.5],[0.0,0.0,-8.5]]'
```

## Interval Push

```bash
uv run train --algo ppo --task g1_walk_flat --sim mujoco \
  env.domain_rand.push_robots=true \
  env.domain_rand.push_interval=500 \
  'env.domain_rand.max_force=[20.0,20.0,5.0]'
```

## Owner-Local Defaults

Keep ranges in the task owner YAML when they are part of the task contract. For
example, `conf/ppo/task/go2_joystick_rough/mujoco.yaml` enables base mass,
center-of-mass, kp/kd, and push randomization, while
`conf/ppo/task/sharpa_inhand/mujoco.yaml` configures object scale, friction, and
force disturbance for Sharpa.

For the full current inventory, see {doc}`0-index`.
