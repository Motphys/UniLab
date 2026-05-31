# Manip-Loco

`go2_arm_manip_loco` combines Go2 locomotion with the Airbot arm. The registered
env is `Go2ArmManipLoco`, the PPO owner is
`conf/ppo/task/go2_arm_manip_loco/mujoco.yaml`, and the HIM-PPO owner is
`conf/ppo_him/task/go2_arm_manip_loco/mujoco.yaml`.

## PPO

```bash
uv run train --algo ppo --task go2_arm_manip_loco --sim mujoco training.no_play=true
```

## HIM-PPO

HIM-PPO is configured by `conf/ppo_him/task/go2_arm_manip_loco/mujoco.yaml` and
implemented by `scripts/train_him_ppo.py`. It is not currently declared as a
top-level `uv run train --algo ...` route in `src/unilab/cli.py`.

The env currently raises if constructed with a backend other than MuJoCo. Keep
backend selection in `--task go2_arm_manip_loco --sim mujoco`, and do not
override `training.sim_backend` alone.

## Tuning Hints

- `env.control_config.arm_action_scale`: magnitude of the arm residual action.
- `env.goal_ee`: end-effector goal sampling ranges and trajectory timing.
- `reward.scales.tracking_lin_vel`, `reward.scales.tracking_ang_vel`,
  `reward.scales.stand_still`: base-behavior trade-offs.
- `env.domain_rand`: mass, friction, push, and PD randomization change the
  training difficulty.

## Near-Risk Checks

Run the env contract and site-Jacobian tests after touching this task:

```bash
uv run pytest tests/envs/locomotion/go2_arm tests/base/backend/test_mujoco_site_jacobian.py
```

If you changed the XML or assets, at least confirm MuJoCo can load the scene:

```bash
uv run python -c "import mujoco; m=mujoco.MjModel.from_xml_path('src/unilab/assets/robots/go2_arm/scene_flat.xml'); print(m.nq, m.nv, m.nu, m.nsensor)"
```

See {doc}`../4-tasks/4-manip_loco` for the task entry.
