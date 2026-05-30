# 部署

一本实操手册，帮助你将 UniLab 策略跨硬件、仿真后端与源框架迁移。每篇教程都遵循
相同的结构：

1. **你从什么开始** —— 训练好的产物与配置。
2. **要改什么** —— 代码、YAML 与资产中最小的一组改动。
3. **如何验证** —— 具体的命令与检查点。

## 选择你的路线

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} 🤖 仿真 → 真机
:link: 1-sim_to_real/0-index
:link-type: doc
:class-card: sd-shadow-md

为 G1 / Go2 / Allegro 上机准备训练好的策略，包含 ONNX 导出与部署侧契约检查。
:::

:::{grid-item-card} 🔀 仿真 → 仿真
:link: 2-sim_to_sim/0-index
:link-type: doc
:class-card: sd-shadow-md

在 MuJoCo 与 Motrix 之间切换同一任务，无需从头重新训练。
:::

:::{grid-item-card} 🔁 框架迁移
:link: 3-framework_migration/0-index
:link-type: doc
:class-card: sd-shadow-md

从 Isaac Lab / Legged Gym / rsl_rl / skrl 迁移任务。
:::

::::

---

## 🤖 仿真 → 真机

::::{grid} 1 1 2 3
:gutter: 3

:::{grid-item-card} 🗺 总览与上机前检查
:link: 1-sim_to_real/1-overview
:link-type: doc
端到端流程 + go/no-go 清单。
:::

:::{grid-item-card} 🦿 G1 全身
:link: 1-sim_to_real/2-g1_whole_body
:link-type: doc
29 自由度人形；运动跟踪部署。
:::

:::{grid-item-card} 🐕 Go2 运动
:link: 1-sim_to_real/3-go2_locomotion
:link-type: doc
摇杆、崎岖地形、Go2W 轮足。
:::

:::{grid-item-card} 🤚 Allegro 手内操作
:link: 1-sim_to_real/4-allegro_inhand
:link-type: doc
方块旋转；摩擦 + 视觉。
:::

:::{grid-item-card} 📦 ONNX 导出与运行时
:link: 1-sim_to_real/5-onnx_runtime
:link-type: doc
训练回放导出、ONNX Runtime 检查以及部署原型输入。
:::

:::{grid-item-card} 🎲 仿真到真机的 DR
:link: 1-sim_to_real/6-domain_randomization
:link-type: doc
按优先级排序的 DR 配方。
:::

:::{grid-item-card} 🛡 安全层
:link: 1-sim_to_real/7-safety_layers
:link-type: doc
软限位、EMA、急停、看门狗。
:::

:::{grid-item-card} ⏱ 延迟与观测滞后
:link: 1-sim_to_real/8-latency_budget
:link-type: doc
训练侧的延迟旋钮与部署侧的测量检查。
:::

:::{grid-item-card} 🔧 故障排查
:link: 1-sim_to_real/9-troubleshooting
:link-type: doc
症状 → 原因 → 修复速查手册。
:::

::::

---

## 🔀 仿真 → 仿真（MuJoCo ↔ Motrix）

::::{grid} 1 1 2 3
:gutter: 3

:::{grid-item-card} 🤔 后端切换
:link: 2-sim_to_sim/1-backend_swap
:link-type: doc
:::

:::{grid-item-card} 📝 Owner YAML 切换
:link: 2-sim_to_sim/2-owner_yaml_swap
:link-type: doc
:::

:::{grid-item-card} 🔬 接触与摩擦对齐
:link: 2-sim_to_sim/3-contact_and_friction_alignment
:link-type: doc
:::

:::{grid-item-card} ⚖ 奖励一致性检查
:link: 2-sim_to_sim/4-reward_parity
:link-type: doc
:::

:::{grid-item-card} 🎞 回放差异
:link: 2-sim_to_sim/5-playback_and_snapshot_differences
:link-type: doc
:::

:::{grid-item-card} 🚫 已知能力缺口
:link: 2-sim_to_sim/6-capability_gaps
:link-type: doc
:::

::::

---

## 🔁 框架迁移

::::{grid} 1 1 2 3
:gutter: 3

:::{grid-item-card} 来自 **Isaac Lab**
:link: 3-framework_migration/1-from_isaac_lab
:link-type: doc
GPU 常驻 → CPU + 共享内存。
:::

:::{grid-item-card} 来自 **Legged Gym**
:link: 3-framework_migration/2-from_legged_gym
:link-type: doc
基于类的环境 → NpEnv。
:::

:::{grid-item-card} 来自 **rsl_rl**
:link: 3-framework_migration/3-from_rsl_rl
:link-type: doc
训练器拆分：collector + learner。
:::

:::{grid-item-card} 来自 **skrl**
:link: 3-framework_migration/4-from_skrl
:link-type: doc
算法覆盖范围与取舍。
:::

:::{grid-item-card} 📋 配置翻译速查表
:link: 3-framework_migration/5-task_config_translation
:link-type: doc
逐字段对照表。
:::

:::{grid-item-card} 📒 奖励移植手册
:link: 3-framework_migration/6-reward_porting
:link-type: doc
以 UniLab 风格表达常见奖励项。
:::

::::

```{toctree}
:hidden:
:caption: Deployment

1-sim_to_real/0-index
2-sim_to_sim/0-index
3-framework_migration/0-index
```
