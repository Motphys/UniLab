# 任务配置翻译速查表

一张并排对照表，列出 Isaac Lab / Legged Gym / skrl 与 UniLab 任务 owner YAML
之间常见配置字段的对应关系。

## Env 级别

```{list-table}
:header-rows: 1
:widths: 20 25 25 30

* - 概念
  - Isaac Lab
  - Legged Gym
  - UniLab
* - Action scale
  - env cfg 上的 `action_scale`
  - env 类中的 `action_scale`
  - owner YAML 中的 `env.action.scale`
* - Decimation
  - env cfg 上的 `decimation`
  - `cfg.control.decimation`
  - `env.decimation`
* - Episode 长度（秒）
  - `episode_length_s`
  - `cfg.env.episode_length_s`
  - `env.episode_length_s`
* - 默认关节位置
  - `init_state.joint_pos`
  - `default_joint_angles`
  - `env.default_joint_pos`（或 asset 侧）
* - 观测噪声
  - `noise.obs.*`
  - `cfg.noise.add_noise`
  - DR provider；参见 DR 文档
```

## Reward

```{list-table}
:header-rows: 1
:widths: 20 25 25 30

* - 概念
  - Isaac Lab
  - Legged Gym
  - UniLab
* - Reward 项注册
  - `RewardManager` cfg
  - `_reward_*` 方法
  - reward registry + env 的 `compute_reward`
* - Reward 权重
  - `RewTerm(weight=…)`
  - `reward_scales.<name>`
  - `reward.<name>.weight`
* - 终止惩罚
  - `Termination` cfg
  - `_reward_termination`
  - reward registry 项 + 终止信号
```

## DR

```{list-table}
:header-rows: 1
:widths: 20 25 25 30

* - 概念
  - Isaac Lab
  - Legged Gym
  - UniLab
* - 随机化摩擦
  - `EventTerm(...friction)`
  - `cfg.domain_rand.friction_range`
  - owner YAML 中的 `dr.friction.*`
* - 推搡机器人
  - `EventTerm(...push)`
  - `cfg.domain_rand.push_robots`
  - `dr.push.*`
* - PD 增益 DR
  - `EventTerm(...stiffness)`
  - `cfg.domain_rand.randomize_motor_strength`
  - `dr.actuator.pd_kp_factor`
```

## Curriculum

```{list-table}
:header-rows: 1
:widths: 20 25 25 30

* - 概念
  - Isaac Lab
  - Legged Gym
  - UniLab
* - 地形课程
  - `TerrainCfg.curriculum`
  - `cfg.terrain.curriculum`
  - `terrain.curriculum.*`
* - 命令范围课程
  - 定制实现
  - `update_command_curriculum`
  - `unilab.base.curriculum`
```

## 另请参阅

- {doc}`6-reward_porting`
- {doc}`../../4-developer_guide/2-contracts/3-task_owner`
