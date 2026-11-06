# 演练：添加自定义 Reward Term

本演练从零开始添加一个 reward term，直到 `reward/<term>` 出现在训练日志中。
运行时是 Torch-only（1.3.4 起；见 changelog 与 ADR-0011）——term 返回
`torch.Tensor`，没有 NumPy 路径。

## 1. 定义 term

任务自有的 term 放在任务包里，例如
`src/unilab/tasks/locomotion/common/sensor_reward_terms.py`；框架级 term 位于
`src/unilab/envs/mdp/rewards.py`，并统一再导出为 `unilab.envs.mdp.<name>`。
新的任务专属 term 应放在所属的任务包中。

term 是一个普通函数：第一个参数是 env，其余来自 `params`：

```python
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from unilab.managers.types import ManagerBasedRlEnv


def action_magnitude_l1(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Sum of absolute raw policy actions, one value per environment."""
    return torch.sum(torch.abs(env.action_manager.action), dim=1)
```

返回契约由 `RewardManager._compute_term` 强制执行
（`src/unilab/managers/reward_manager.py`）：形状为 `(num_envs,)` 的
`torch.Tensor`，dtype 为 `float32`，位于 env 设备上。类型、dtype、设备或
形状错误会在第一次 `compute()` 调用时抛错；返回非有限值会带 term 名抛错。
manager 以 `func(env, **params)` 调用 term，因此 YAML 中 `params:` 下的每个
键都必须是该函数接受的参数。

有状态或需要绑定传感器的 term 改用类形式：类在 manager 构造时以
`(cfg, env)` 实例化一次，之后每步以 `func(env, **params)` 调用；可选的
`reset(env_ids)` 清理按 episode 保存的状态（
`src/unilab/managers/manager_base.py` 中的 `ManagerTermBaseCfg`）。在构造期
绑定具名 XML 传感器是
`src/unilab/tasks/locomotion/common/manager_terms.py` 中 `SensorTermBase`
的模式。

## 2. 接入 owner YAML

reward term 位于任务 owner YAML 顶层的 `reward:` 映射下
（`src/unilab/conf/<algo>/task/<task>/base.yaml` 或其 backend 兄弟文件），
每个 term 一段，含 `func`、`weight` 与可选 `params`——真实结构取自
`src/unilab/conf/ppo/task/go2_joystick_flat/base.yaml`：

```yaml
reward:
  action_magnitude:
    func: unilab.tasks.locomotion.common.manager_terms.action_magnitude_l1
    weight: -0.001
```

`func` 是可导入的 `module.path.name`。该 term 贡献的单步 reward 为
`raw_value * weight`（env 内部再乘 `dt`；日志显示的是未乘 dt 的加权速率）。
惩罚项用负权重——框架的惩罚 term 返回非负代价，不要再取负一次。

验证组合结果（仍会经过 CLI 的后端预检；`--sim mujoco` 需要 `mujoco`
extra）：

```bash
uv run train --algo ppo --task <task> --sim mujoco --cfg job \
  | grep -A3 'action_magnitude'
# 预期：输出你 owner YAML 中的 func/weight 行。
```

## 3. 在训练日志中确认

跑一次短训练（需要后端的 extra；有意义的运行需要 GPU）：

```bash
uv run train --algo ppo --task <task> --sim mujoco \
  algo.num_envs=64 algo.max_iterations=5 training.no_play=true
```

Manager-Based 环境把逐项 reward 诊断以 `reward/<term>` 键发布到
`info["log"]`——由 `RewardManager.step_reward_extras()` 产生
（`src/unilab/managers/reward_manager.py`，跨环境对 `raw_value * weight`
取均值，未乘 `dt`）——训练 logger 再把它们转发到 TensorBoard / W&B
（{doc}`../2-user_guide/1-training/3-logging`）：

```bash
uv run tensorboard --logdir logs/rsl_rl_ppo/<task>
# 预期：与其他 reward/<term> 标签并列出现 reward/action_magnitude。
```

term 出错时失败是显式的：构造错误会指出 term 名；返回类型/形状错误由
`RewardManager` 抛出；返回 NaN/Inf 会抛出
`RewardManager term '<term>' returned NaN/Inf`（启用 NaN guard 时还会写出
状态 dump）。诊断流程见
{doc}`3-training_diagnostics`。

## 4. 调整权重

已声明的 term 可以在命令行直接调权，无需改 YAML：

```bash
uv run train --algo ppo --task <task> --sim mujoco \
  reward.action_magnitude.weight=-0.01
```

这可行是因为 `reward` 是组合配置顶层的 Hydra section（见
{doc}`../2-user_guide/2-algorithms/1-ppo` 的 “Reward term weights” 小节）。
新增 term 请写进 owner YAML 以保证任务可复现；CLI override 适合扫参。

从其他框架移植 term 时先查内置表——该 term 可能已经存在：
{doc}`../5-reference/6-manager_terms/1-rewards`。Legged Gym 的映射指南见
{doc}`../3-deployment/3-framework_migration/6-reward_porting`。
