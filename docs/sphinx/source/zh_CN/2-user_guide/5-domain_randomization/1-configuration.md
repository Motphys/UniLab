# 配置

域随机化在所选的 task owner YAML 内部配置。先使用 `--task` 和 `--sim` 选择后端专属行为，
然后在所选的 owner 内部 override 字段。

当前有两条声明路径：

- Manager-Based（Compatible）任务通过 owner YAML 的 `events:` manager term 声明
  reset / interval 随机化，例如 `src/unilab/conf/ppo/task/go1_joystick_flat/base.yaml`。
- 任务也可以挂载任务级 provider，并在 `env.domain_rand` 下配置 legacy provider
  字段；目前仓内没有任务使用这条路径。

常见的生命周期边界：

- init 生命周期项会改变模型 identity 或几何，必须在 env/backend 初始化期间运行。
- reset 生命周期项通过后端支持的 payload 在 reset 时扰动状态或模型参数。
- interval 生命周期项在 step 之间施加扰动。

详细的任务状态和字段语义见 {doc}`0-index`。

域随机化按生命周期划分：init、reset 和 interval。legacy 路径的 manager 位于
`src/unilab/dr/manager.py`；task provider 位于 env owner 附近，
后端能力通过 `unisim.backend.base` 声明。

## Reset Gravity

在启用 gravity reset 随机化时使用 `--sim mujoco`；Motrix 在当前后端中
未提供相同的 gravity 能力。该项只在任务级 provider 路径
（`env.domain_rand.randomize_gravity` 与 `env.domain_rand.gravity_range`）上可用，
目前仓内没有任务使用这条路径。

## Interval Push

Manager-Based 任务通过 `env.events.push_robot` term 配置周期推扰。例如，
`src/unilab/conf/ppo/task/go1_joystick_flat/base.yaml` 使用
`push_by_setting_velocity`，间隔为 15 秒，并按轴声明速度范围。

```bash
uv run train --algo ppo --task go1_joystick_flat --sim mujoco \
  'env.events.push_robot.interval_range_s=[10.0,10.0]'
```

## Owner 本地默认值

当取值范围是任务 contract 的一部分时，将其保留在 task owner YAML 中。例如，
rough 四足家族的 base mass、质心、kp/kd 和 push 随机化作为 event term 声明在共享 base
`src/unilab/conf/ppo/task/quadruped_joystick_rough/base.yaml`（`go2_joystick_rough` 的 backend
owner 通过 Hydra defaults 组合它）。

完整的当前清单见 {doc}`0-index`。
