# Reward 移植

把 Legged Gym 的 `_reward_*` 方法映射到 owner YAML 的 `reward` term。
Manager-Based reward 接收 env，通过 entity facade 和各 manager 读取批量状态，
返回形状为 `(num_envs,)` 的 NumPy 数组。

## 跟踪、平滑与关节限制

以下示例的 `twist` command 必须由 owner 的 `env.commands` 定义。
跟踪与动作平滑项参考 `src/unilab/conf/ppo/task/go2_joystick_flat/base.yaml`；
关节限制与终止项展示现有 helper 的配置方式，权重应按任务评估。

```yaml
reward:
  tracking_lin_vel:
    func: unilab.tasks.locomotion.common.manager_terms.track_lin_vel_xy_exp
    weight: 1.0
    params:
      std: 0.5
      command_name: twist
  action_rate:
    func: unilab.envs.mdp.action_rate_l2
    weight: -0.005
  joint_limits:
    func: unilab.envs.mdp.joint_pos_limits
    weight: -1.0
  termination:
    func: unilab.envs.mdp.is_terminated
    weight: -1.0
```

`track_lin_vel_xy_exp` 使用底盘坐标系的 xy 速度误差，公式为
`exp(-error_squared / std**2)`。若源实现分母为 `tracking_sigma`，
应令 `std = sqrt(tracking_sigma)`，并保持坐标系与 command 维度一致。

`action_rate_l2` 从 action manager 读取当前与前一步动作。
`joint_pos_limits` 从 entity facade 读取 soft joint limits。
两者返回非负代价，惩罚符号由负的 `weight` 提供，不要重复取负。

## 执行器力读回

在机器人或任务的源 XML 中，为每个执行器声明一个命名的
`<actuatorfrc>` 传感器。每个传感器读取 MuJoCo 执行空间中的标量作用力，
而不是关节空间中的广义力：

```xml
<sensor>
  <actuatorfrc name="hip_force" actuator="hip"/>
  <actuatorfrc name="knee_force" actuator="knee"/>
</sensor>
```

基于类的 Manager term 通过 `tensor_sensor_names` 声明相同的名称。在 packed
Torch 路径上，按执行器顺序读取这些视图；需要矩阵时，将各个
`(num_envs, 1)` 视图拼接：

```python
names = ("hip_force", "knee_force")
entity = env.scene["robot"]
views = env.scene.read_plan.sensor_tensor_views(entity, names).values
actuator_force = torch.cat([views[name] for name in names], dim=1)
```

结果形状为 `(num_envs, num_actuators)`。L2 力矩惩罚应使用相同的 packed
视图，不要根据 PD 增益重建力矩。

## 接触条件奖励

接触历史不是通用 `state.prev_contact` 字段。需要计时或边沿检测的 reward
使用有状态的 manager term，并在局部 reset 时重置对应 env 的缓存。
现有示例位于 `unilab.tasks.locomotion.common.gait_terms`：
`feet_air_time` 是时间窗口奖励，`foot_air_time` 提供当前腾空计时。
它们不等同于 Legged Gym 的首次接触奖励；迁移时核对触发时刻、计时单位和
command gating，再通过固定轨迹比较逐项输出。

## 终止处理

`unilab.envs.mdp.is_terminated` 读取 termination manager 的非 timeout
终止掩码。通过负权重构成终止惩罚，并单独核对 timeout 是否应参与源任务惩罚。

## 参考

- {doc}`5-task_config_translation`
- `src/unilab/envs/mdp/rewards.py`
- `src/unilab/tasks/locomotion/common/manager_terms.py`
- `src/unilab/tasks/locomotion/common/gait_terms.py`
