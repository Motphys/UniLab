# 演练：接入新机器人

本演练从 URDF/MJCF 资产出发，一步步走到首次 PPO 训练跑通——
`uv run train --algo ppo --task <task> --sim <backend>`——每步都带验证。
它只组合现有的 owner 层，不新增代码路径：资产
（{doc}`../2-user_guide/7-tooling/5-robot_import`）、任务/场景 keyframe、
owner YAML（{doc}`../2-user_guide/1-training/2-hydra_config`）以及 registry
条目（清单视角见 {doc}`../4-developer_guide/3-extending/1-new_task`）。

需要安装仿真器、显示器或 GPU 的步骤都已标注。纯组合检查（`--cfg job`）仍会
经过 CLI 的后端预检，因此 `--sim mujoco` 需要 `mujoco` extra；被路由的 PPO
入口还会导入内置 RL 栈，所以还需要 `unirl` extra（源码检出中执行
`uv sync --extra mujoco --extra unirl`——与 `make setup` 相同的同步命令）。

## 0. 选定名称

三个名称在所有地方保持一致：

- `<robot>` —— 资产目录名，例如 `src/unilab/assets/robots/go2/` 下的 `go2`。
- `<task>` —— snake_case 的 CLI 与 owner YAML 目录名，例如
  `go2_joystick_flat`。CLI 会拒绝带斜杠或点的名称
  （`src/unilab/cli.py` 中的 `_check_task_name`）。
- `<RegistryName>` —— PascalCase 的 registry 键，例如 `Go2JoystickFlat`。

## 1. 准备资产

按 {doc}`../2-user_guide/7-tooling/5-robot_import` 操作。URDF 源：

```bash
uv run scripts/tools/import_robot.py <urdf_path> [robot_name]
```

脚本通过 `urdf-to-mjcf` 转换 URDF、整理 mesh，并打开 `mujoco.viewer` 供
keyframe 调整（实现：`scripts/tools/import_robot.py`）。它把场景 fragment
写为 `scene.xml`；fragment 的名字可以保留或自行修改——Go2 的参考实现用的是
`scene_flat.xml`。如果源已经是 MJCF，手工按同一目录契约摆放：

- `src/unilab/assets/robots/<robot>/assets/` —— 仅 `.stl` / `.obj` mesh。
- `src/unilab/assets/robots/<robot>/<robot>.xml` —— 仅含机器人的 MJCF：
  body、joint、actuator、sensor 与 mesh 引用。
- `src/unilab/assets/robots/<robot>/scene_flat.xml`（名字自定；导入工具写的是
  `scene.xml`）—— 任务/场景 fragment：`<include>` 机器人 XML、地面、任务
  传感器与 keyframe。参考形态：`src/unilab/assets/robots/go2/scene_flat.xml`。

碰撞几何优先用简单 primitive，不要直接用高面数视觉 mesh；后者同时伤害仿真
速度与接触稳定性。

验证：

```bash
ls src/unilab/assets/robots/<robot>/
# 预期：assets/  <robot>.xml  以及场景 fragment
#（导入工具产物名为 scene.xml）。
```

## 2. 调整 keyframe

默认位姿是任务/场景 fragment 里的 `<keyframe>`——绝不放进 `<robot>.xml`。
这是仓库硬约束：机器人 XML 保持任务无关，多个任务才能共享同一机器人。
参考 `src/unilab/assets/robots/go2/scene_flat.xml` 的 `home` keyframe：

```xml
<keyframe>
  <key name="home" qpos="
      0 0 0.3  1 0 0 0
      0 0.8 -1.5
      ...
      "
    ctrl="..."/>
</keyframe>
```

导入工具的 viewer 关闭时会把调好的 keyframe 自动写回场景 fragment。人工
核对时至少确认：

- 站立姿态：浮动基座高度让足底正常接触地面——不悬空、不穿透。在 viewer 中
  按 `c` 显示接触点。
- 关节角度位于关节行程内、接近自然站姿；`qpos` 的排列是自由关节
  （3 个位置 + 4 元数姿态）后接 XML 顺序的驱动关节。
- 重心：基座位于支撑多边形上方，机器人出生时不会倾倒。

验证：

```bash
grep -n '<key name=' src/unilab/assets/robots/<robot>/scene_flat.xml  # 或导入工具写出的 scene.xml
# 预期：每个 keyframe 一行，例如 <key name="home" ...>
grep -c '<keyframe' src/unilab/assets/robots/<robot>/<robot>.xml
# 预期：0 —— keyframe 不进入机器人 XML。
```

## 3. 编写 owner YAML

PPO 把每个 task×backend 组合解析为两个文件（组合规则见
{doc}`../2-user_guide/1-training/2-hydra_config`）：

- `src/unilab/conf/ppo/task/<task>/base.yaml` —— 共享的任务声明。
- `src/unilab/conf/ppo/task/<task>/mujoco.yaml` —— 后端 owner：继承 base，
  绑定 registry 名与后端身份。

最小 base owner，改写自 Go2 的规范 owner
`src/unilab/conf/ppo/task/go2_joystick_flat/base.yaml`：

```yaml
# @package _global_
env:
  scene:
    model_file: src/unilab/assets/robots/<robot>/scene_flat.xml  # 若沿用导入工具的名字则是 scene.xml
    default_keyframe_name: home   # 第 2 步的 keyframe
    entities:
      robot:
        root_body_name: base
        body_names: [base]
        joint_names: [<robot>_hip_joint, ...]   # XML 关节名，按动作顺序
        actuator_names: [<robot>_hip, ...]      # XML 执行器名
  sim_dt: 0.01
  ctrl_dt: 0.02                   # 必须是 sim_dt 的整数倍
  max_episode_seconds: 20.0
  observations:
    policy:
      terms:
        base_ang_vel:
          func: unilab.envs.mdp.builtin_sensor
          params: {sensor_name: gyro}
        projected_gravity:
          func: unilab.envs.mdp.projected_gravity_from_sensor
          params: {sensor_name: upvector}
        joint_pos:
          func: unilab.envs.mdp.joint_pos_rel
        joint_vel:
          func: unilab.envs.mdp.joint_vel_rel
        actions:
          func: unilab.envs.mdp.last_action
        command:
          func: unilab.envs.mdp.generated_commands
          params: {command_name: twist}
  actions:
    joint_pos:
      _target_: unilab.envs.mdp.JointPositionActionCfg
      entity_name: robot
      actuator_names: [".*"]
      scale: 0.25
      use_default_offset: true
  commands:
    twist:
      _target_: unilab.envs.mdp.UniformVelocityCommandCfg
      entity_name: robot
      resampling_time_range: [20.0, 20.0]
      heading_command: false
      ranges:
        lin_vel_x: [-0.6, 1.0]
        lin_vel_y: [-0.4, 0.4]
        ang_vel_z: [-0.8, 0.8]
  events:
    reset_scene_to_default:
      func: unilab.envs.mdp.reset_scene_to_default
      mode: reset
    reset_root_state_uniform:
      func: unilab.envs.mdp.reset_root_state_uniform
      mode: reset
      params:
        pose_range:
          x: [-0.5, 0.5]
          y: [-0.5, 0.5]
          yaw: [-3.141592653589793, 3.141592653589793]
        velocity_range:
          x: [-0.5, 0.5]
          y: [-0.5, 0.5]
  terminations:
    time_out:
      func: unilab.envs.mdp.time_out
      time_out: true
    bad_orientation:
      func: unilab.envs.mdp.bad_orientation
      params: {limit_angle: 1.0471975511965976}
  policy_observation_group: policy

reward:
  tracking_lin_vel:
    func: unilab.envs.mdp.track_linear_velocity
    weight: 1.0
    params: {std: 0.5, command_name: twist}
  tracking_ang_vel:
    func: unilab.envs.mdp.track_angular_velocity
    weight: 0.2
    params: {std: 0.5, command_name: twist}
  action_rate:
    func: unilab.envs.mdp.action_rate_l2
    weight: -0.005
```

关于最小集的说明：

- 传感器名（`gyro`、`upvector`）是 Go2 机器人 XML 里的名字
  （`src/unilab/assets/robots/go2/go2.xml`）。绑定传感器的 term 在构造期对
  缺失的名字 fail closed，因此请在你的机器人 XML 里声明等价传感器，或删掉
  绑定它们的 term。
- 框架级跟踪 term（`unilab.envs.mdp.track_linear_velocity` /
  `track_angular_velocity`）读 root state 而不是具名传感器，只要 command 名为
  `twist` 就能用于任意机器人；参数与语义见
  {doc}`../5-reference/6-manager_terms/1-rewards`。
- 当算法需要消费特权观测时，再加 `critic` 观测组和
  `env.critic_observation_group: critic`（Go2 owner 两者都有）。

后端 owner（`src/unilab/conf/ppo/task/<task>/mujoco.yaml`），参照
`src/unilab/conf/ppo/task/go2_joystick_flat/mujoco.yaml`：

```yaml
# @package _global_
defaults:
  - /task/<task>/base
  - _self_

training:
  task_name: <RegistryName>     # 第 4 步的 registry 键
  sim_backend: mujoco
algo:
  num_envs: 1024
  max_iterations: 151
```

验证——仅做配置组合（仍会经过 CLI 的后端预检）：

```bash
uv run train --algo ppo --task <task> --sim mujoco --cfg job \
  | grep -E 'task_name|default_keyframe_name|model_file'
# 预期：输出你两个 YAML 组合后的值。
```

## 4. 注册任务

registry 把 `<RegistryName>` 映射到一个 config 工厂和每个后端各一个 env
工厂。仓库内任务注册在 `src/unilab/tasks/` 下的包里；Go2 包就是最小模板
（`src/unilab/tasks/locomotion/go2/__init__.py`）：

```python
"""Hydra-owned Manager-Based <robot> production registrations."""

from unilab.base import registry
from unilab.envs import ManagerBasedRlEnvCfg, make_manager_based_rl_env

registry.register_env_config("<RegistryName>", ManagerBasedRlEnvCfg)
registry.register_env("<RegistryName>", make_manager_based_rl_env, sim_backend="mujoco")
```

然后把该模块加入 `src/unilab/tasks/__init__.py`（或你的家族包
`__init__.py`）的 `__unilab_registry_modules__` 元组，让
`ensure_registries()` 导入它。仓库外的第三方任务包改用 `unilab.tasks`
entry-point 组注册——见 {doc}`../4-developer_guide/1-architecture/5-registry` 与
`src/unilab/base/registry.py`。

验证——纯主机侧，不需要后端：

```bash
uv run python -c "
from unilab.base import registry
registry.ensure_registries()
print(registry.list_registered_envs()['<RegistryName>'])
"
# 预期：{'config_factory': 'ManagerBasedRlEnvCfg', 'available_backends': ['mujoco']}
```

## 5. 选择后端

从 `mujoco`（默认后端）开始。每增加一个后端需要：

1. 任务包里加一行 `register_env(..., sim_backend="<backend>")`；
2. 对应的 owner YAML `src/unilab/conf/ppo/task/<task>/<backend>.yaml`，其中
   `training.sim_backend: <backend>`；
3. 该后端能提供你的任务所绑定 term 的证据——查生成的
   {doc}`../5-reference/5-support_matrix`，DR term 另查
   {doc}`../2-user_guide/5-domain_randomization/2-field_reference`。
   不支持的能力在构造期 fail closed；后端无法服务的 term 在该后端 owner 里
   置为 `null`（模式：`src/unilab/conf/ppo/task/go2_joystick_flat/motrix.yaml`
   中的 `pd_gains: null`）。

每加一个后端验证一次：用 `--sim <backend>` 重跑 `--cfg job` 组合检查。

## 6. 首次训练

需要后端的 Python extra 以及 `unirl` RL 栈 extra
（`uv sync --extra mujoco --extra unirl`）；有意义的运行需要 GPU。先小规模：

```bash
uv run train --algo ppo --task <task> --sim mujoco \
  algo.num_envs=64 algo.max_iterations=5 training.no_play=true
```

在 1.5.9（PPO，MuJoCo 后端，单卡 RTX 4090）上以
`algo.num_envs=16 algo.max_iterations=1` 跑 `go2_joystick_flat` 实测，
终端输出节选：

```text
Storing git diff for 'unilab' in: .../logs/rsl_rl_ppo/Go2JoystickFlat/2026-11-08_22-34-23_mujoco/git/unilab.diff
                            Learning iteration 0/1
                            Total steps: 384
                       Steps per second: 1119
                        Collection time: 0.247s
                          Learning time: 0.096s
                        Mean value loss: 0.0143
                    Mean surrogate loss: -0.0449
                      Mean entropy loss: 8.7087
                            Mean reward: 0.02
                    Mean episode length: 1.00
```

首跑检查点：

1. 组合与 env 构造会快速失败并指出有问题的 term 或字段——例如观测 term
   绑定了不存在的传感器时会在 manager 构造期抛错。修 owner YAML 后重跑。
2. 运行目录出现在 `logs/rsl_rl_ppo/<RegistryName>/` 下——该级目录是
   `training.task_name` 的 registry 名（例如 `Go2JoystickFlat`），不是
   snake_case 的 CLI 任务名——目录名为 `<时间戳>_<后端>/`，其中包含
   `run_config.json`（细节见 {doc}`../2-user_guide/1-training/3-logging`）。
3. 首个 episode 结束后，TensorBoard 中出现终止计数器
   `Episode_Termination/<term>`
   （`uv run tensorboard --logdir logs/rsl_rl_ppo/<RegistryName>`）。自 1.5.8
   起 PPO 路由不再转发逐项 `reward/<term>` 标量——这些值改以定长
   `info["reward_components"]` 张量的形式传递给 off-policy 运行时；逐项数值
   的读取方式见 {doc}`2-custom_reward` 第 3 步。任何 reward 或观测出现
   NaN/Inf 时，训练会带着 term 名抛错，并且 NaN guard 会写出状态
   dump——见 {doc}`3-training_diagnostics`。

然后恢复到 owner 的规模（`algo.num_envs` / `algo.max_iterations`），并评估
最近一次运行：

```bash
uv run eval --algo ppo --task <task> --sim mujoco --load-run -1
```

## 下一步

- 添加或调权 reward term：{doc}`2-custom_reward`。
- 内置 term 速查表：{doc}`../5-reference/6-manager_terms/0-index`。
- 同一契约的清单视角：{doc}`../4-developer_guide/3-extending/1-new_task`。
- 对出生场景做交互式目视冒烟检查（需要显示器）：
  `uv run scripts/visualize_task_env.py --task <RegistryName> --backend mujoco --num_envs 4`。
