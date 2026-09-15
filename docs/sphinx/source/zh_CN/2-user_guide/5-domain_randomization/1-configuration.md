# 配置

域随机化在所选 Manager-Based task owner YAML 内部配置。先使用 `--task` 和
`--sim` 选择后端专属行为，然后在所选 owner 内部 override 字段。

生命周期边界：

- 固定 model/tool identity 在 `env.fixed_model_variants` 上声明一次，并在
  backend construction 期间 realization；它不是 reset 随机化。
- reset 生命周期 event term 通过一个 `ResetStateTransaction` 和一个 backend
  payload 扰动状态或 curated model parameters。
- interval 生命周期 event term 在 step 之间施加扰动。

Backend 支持通过 `unisim.backend.base` 显式声明。所选 backend 未声明的能力
会 fail closed。

## Reset Gravity

启用 gravity reset 随机化时使用 `--sim mujoco`；Motrix 未声明 gravity reset
能力。请通过调用 `randomize_physics_scene_gravity` 的 reset event term 配置。

## Interval Push

Manager-Based 任务通过 `env.events.push_robot` term 配置周期推扰。例如，
保留的 `g1_wbt_obs` owner 使用 `push_by_setting_velocity`，并按轴声明速度范围。

```bash
uv run train --algo sac --task g1_wbt_obs --sim mujoco \
  'env.events.push_robot.interval_range_s=[10.0,10.0]'
```

## Owner 本地默认值

当取值范围是任务 contract 的一部分时，将其保留在 task owner YAML 中。例如，
rough 任务族可以把 base mass、质心、kp/kd 和 push 随机化作为 event term
声明在自己的共享 owner base 中。

完整当前清单见 {doc}`0-index`。
