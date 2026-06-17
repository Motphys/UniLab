# Sim2Sim 跨后端任务契约状态

本页给出每个 task 在两后端（MuJoCo / Motrix）上 DENYLIST 契约字段是否一致的当前状态，
回答「这个 task 当前能否 A 后端训练 → B 后端 play」。契约字段定义见
`src/unilab/training/sim2sim.py`，运行时守卫与机制概述见 AGENTS.md 的 Sim2Sim 章节。

## 复跑方式

```
uv run scripts/audit_sim2sim_contracts.py
```

该脚本只读，对每个 task 用 hydra `compose` 得到有效配置（展开 `defaults` / `base.yaml` /
`# @package _global_`），再按守卫同一套归一化逐字段比对，输出本页相同的判定。

判定分三档：

- ✅ **可迁移**：DENYLIST 在 composed config 上无分歧。
- ❌ **阻断**：存在 value-diff，或 env 结构字段不对称出现；守卫会抛 `CrossBackendIncompatibleError`。
- ⚪ **N/A**：缺少 mujoco↔motrix 配对。

> 一个口径细节：脚本比对的是 composed YAML，「某后端没写该字段」显示为 `<absent>`；运行时
> env dataclass 会给该路径填默认值。对 env 结构字段（`action_scale` / `sampling_mode`）
> 守卫对不对称出现一律 fail-closed；`algo` 专属字段（`empirical_normalization` /
> `obs_normalization`）在目标缺省时按设计跳过（跨算法合法）。

## `conf/ppo/task/`

| Task | 判定 | 分歧 |
|---|---|---|
| allegro_inhand · allegro_inhand_grasp · g1_climb_tracking · g1_motion_tracking · g1_wall_flip_tracking · go1_joystick_rough · go2_arm_manip_loco · go2_handstand · go2_joystick_flat · go2_joystick_rough · go2w_joystick_flat · go2w_joystick_rough · sharpa_inhand · sharpa_inhand_grasp | ✅ | 无 |
| g1_box_tracking | ❌ | `empirical_normalization` false↔true；`obs_groups` critic 组差异 |
| g1_flip_tracking | ❌ | `empirical_normalization` true↔false；`obs_groups`；`action_scale` 29 维↔默认 0.25；`sampling_mode` 两后端运行时同为 `start`（无害） |
| g1_walk_flat | ❌ | `action_scale` 0.25↔0.5；`empirical_normalization` false↔true；`obs_groups` |
| go1_joystick_flat | ❌ | `empirical_normalization` false↔true |
| g1_motion_tracking_deploy · go2_footstand | ⚪ | 仅 mujoco |

## `conf/appo/task/`

| Task | 判定 | 分歧 |
|---|---|---|
| allegro_inhand · g1_climb_tracking · g1_motion_tracking · go2_joystick_flat | ✅ | 无 |
| g1_flip_tracking | ❌ | `action_scale` 29 维↔默认 0.25；`sampling_mode` 同为 `start`（无害） |
| g1_wall_flip_tracking | ❌ | `action_scale` 29 维↔默认 0.25；`sampling_mode` `start`↔默认 `adaptive` |
| g1_walk_flat · go1_joystick_flat | ⚪ | 仅 mujoco |

## 其它配置树

`conf/ppo_him/task`、`conf/offpolicy/task`、`conf/hora_distill/task` 均无 mujoco↔motrix
配对，sim2sim 不适用。

## 字段语义速查

- **`env.control_config.action_scale`** —— 策略输出到关节目标的线性缩放系数；
  改动等价于动作幅值整体放缩，无法跨值迁移。
- **`algo.empirical_normalization`** —— 是否在 actor 前插入 running mean/std 归一化层。
  该层 buffer 烘进 checkpoint，ON / OFF 两类 checkpoint 不可互换；统一必须重训。
- **`algo.obs_groups`** —— actor / critic 取哪几个 obs group 作为输入。
  在多数 env 上 actor 实际输入由 `env.obs_groups_spec` 决定，YAML 中的差异常只影响 critic
  的训练侧，play 时不影响 actor。
- **`env.sampling_mode`** —— motion-tracking task 的参考帧采样策略。

## 把 BLOCKED task 变为可迁移

| 字段 | 是否可仅改 YAML | 说明 |
|---|---|---|
| `obs_groups` | 可（多数情况） | 命名差异统一进 `base.yaml`，actor 部署不变 |
| `sampling_mode` | 可（取决于 task） | 两后端运行时已同值时只需补齐显式声明 |
| `action_scale` | **不可** | 改值即改训练动力学，必须 owner 决策 + 重训 |
| `empirical_normalization` | **不可** | 改变网络结构，必须重训 |

试点示例：`conf/ppo/task/g1_walk_flat/{base,mujoco,motrix}.yaml`。`base.yaml` 承载共享契约，
`mujoco.yaml` 直接继承，`motrix.yaml` 为单后端调参 override 了若干契约字段——这种 override
即令该 task 在该后端不可 sim2sim 迁移，去掉 override 即可恢复。
