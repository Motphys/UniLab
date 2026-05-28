# DR Recipes

Domain randomization is split by lifecycle: init, reset, and interval. The
manager path is `src/unilab/dr/manager.py`; task providers live near the env
owners, and backend capabilities are declared through
`src/unilab/base/backend/base.py`.

## Reset Gravity

Use MuJoCo owners when enabling gravity reset randomization; Motrix does not
advertise the same gravity capability in the current backend.

```bash
uv run scripts/train_rsl_rl.py task=g1_walk_flat/mujoco \
  env.domain_rand.randomize_gravity=true \
  'env.domain_rand.gravity_range=[[0.0,0.0,-10.5],[0.0,0.0,-8.5]]'
```

## Interval Push

```bash
uv run scripts/train_rsl_rl.py task=g1_walk_flat/mujoco \
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

For the full current inventory, see {doc}`index`.
