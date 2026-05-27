# Task Config Translation Cheatsheet

A side-by-side map of common config fields across Isaac Lab / Legged Gym
/ skrl and UniLab task owner YAMLs.

## Env-level

```{list-table}
:header-rows: 1
:widths: 20 25 25 30

* - Concept
  - Isaac Lab
  - Legged Gym
  - UniLab
* - Action scale
  - `action_scale` on env cfg
  - `action_scale` in env class
  - `env.action.scale` in owner YAML
* - Decimation
  - `decimation` on env cfg
  - `cfg.control.decimation`
  - `env.decimation`
* - Episode length (s)
  - `episode_length_s`
  - `cfg.env.episode_length_s`
  - `env.episode_length_s`
* - Default joint pos
  - `init_state.joint_pos`
  - `default_joint_angles`
  - `env.default_joint_pos` (or asset-side)
* - Observation noise
  - `noise.obs.*`
  - `cfg.noise.add_noise`
  - DR provider; see DR docs
```

## Reward

```{list-table}
:header-rows: 1
:widths: 20 25 25 30

* - Concept
  - Isaac Lab
  - Legged Gym
  - UniLab
* - Reward term registration
  - `RewardManager` cfg
  - `_reward_*` methods
  - reward registry + env's `compute_reward`
* - Reward weight
  - `RewTerm(weight=…)`
  - `reward_scales.<name>`
  - `reward.<name>.weight`
* - Termination penalty
  - `Termination` cfg
  - `_reward_termination`
  - reward registry term + termination signal
```

## DR

```{list-table}
:header-rows: 1
:widths: 20 25 25 30

* - Concept
  - Isaac Lab
  - Legged Gym
  - UniLab
* - Randomize friction
  - `EventTerm(...friction)`
  - `cfg.domain_rand.friction_range`
  - `dr.friction.*` in owner YAML
* - Push robot
  - `EventTerm(...push)`
  - `cfg.domain_rand.push_robots`
  - `dr.push.*`
* - PD gain DR
  - `EventTerm(...stiffness)`
  - `cfg.domain_rand.randomize_motor_strength`
  - `dr.actuator.pd_kp_factor`
```

## Curriculum

```{list-table}
:header-rows: 1
:widths: 20 25 25 30

* - Concept
  - Isaac Lab
  - Legged Gym
  - UniLab
* - Terrain curriculum
  - `TerrainCfg.curriculum`
  - `cfg.terrain.curriculum`
  - `terrain.curriculum.*`
* - Command range curriculum
  - bespoke
  - `update_command_curriculum`
  - {py:mod}`unilab.base.curriculum`
```

## See also

- {doc}`reward_porting_cookbook`
- {doc}`../../developer_guide/contracts/task_owner_config`
