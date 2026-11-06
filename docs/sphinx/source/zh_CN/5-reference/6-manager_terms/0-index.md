# Manager Term 参考

UniLab 内置 Manager-Based reward、observation、termination term 的速查表，
按 owner 层组织：`unilab.envs.mdp` 下的框架级 term，以及
`src/unilab/tasks/` 下的任务层 term。表中的名称、参数与默认值均以所引
源码模块为准整理；权威签名以 API Reference 为准。

:::{only} api_ref
```{raw} html
<ul>
  <li>框架级 term：<a href="../../../api_reference/envs/index.html"><code>unilab.envs</code> API 参考</a></li>
  <li>任务层 term：<a href="../../../api_reference/tasks/locomotion.html"><code>unilab.tasks.locomotion</code> API 参考</a>、
      <a href="../../../api_reference/tasks/motion_tracking.html"><code>unilab.tasks.motion_tracking</code> API 参考</a></li>
</ul>
```
:::

:::{only} prose_only
纯散文预览构建会跳过 API Reference。权威签名见源码树：
[unilab/envs/mdp](https://github.com/unilabsim/UniLab/tree/main/src/unilab/envs/mdp)、
[unilab/tasks](https://github.com/unilabsim/UniLab/tree/main/src/unilab/tasks)。
:::

## 如何引用 term

term 在任务 owner YAML（例如
`src/unilab/conf/ppo/task/go2_joystick_flat/base.yaml`）中以
`func + params` 形式声明。`func` 是可导入的 `模块.路径.名字`；所有框架级
term 都在 `unilab.envs.mdp.<name>` 处重导出，owner YAML 使用的正是这个短
路径。类形式的 term 在 manager 构建时用 `(cfg, env)` 实例化一次，之后每步
调用；普通函数则每步直接调用。两种形式下 manager 都以
`func(env, **params)` 调用 term
（见 `src/unilab/managers/manager_base.py`、
`src/unilab/managers/reward_manager.py`），因此 `params:` 必须与 term 接受
的参数一致。

- Reward term 声明在顶层 `reward:` 映射下，含 `func`、`weight` 与可选的
  `params`：
  ```yaml
  reward:
    action_rate:
      func: unilab.envs.mdp.action_rate_l2
      weight: -0.005
  ```
- Observation term 声明在 `env.observations.<组名>.terms.<名称>` 下，含
  `func` 与可选的 `params`：
  ```yaml
  env:
    observations:
      policy:
        terms:
          joint_pos:
            func: unilab.envs.mdp.joint_pos_rel
  ```
- Termination term 声明在 `env.terminations.<名称>` 下，含 `func` 与可选
  `params`；加上 `time_out: true` 表示该 term 计入回合超时（截断）而非失败
  （见 `src/unilab/managers/termination_manager.py` 的
  `TerminationTermCfg.time_out`）：
  ```yaml
  env:
    terminations:
      time_out:
        func: unilab.envs.mdp.time_out
        time_out: true
  ```

依赖后端能力的 term——具名传感器、射线投射器、reset 字段——在构造期绑定
该能力，后端不具备时直接抛错（fail-closed），不会静默降级（机制见
`src/unilab/envs/mdp/observations.py` 的 `bind_sensor_data` 绑定与
`src/unilab/envs/mdp/events.py` 的 `_ResetRandomizationTerm` 约定）。

## 范围说明

本表只覆盖 reward、observation、termination 三类 term。任务包还在其他
YAML 段中配置了下列 term 类型，未列入表中：

- Command 与 action term，例如
  `unilab.envs.mdp.commands.UniformVelocityCommandCfg`、
  `unilab.tasks.locomotion.g1.manager_terms.G1VelocityCommandCfg`（带 0.2
  平面模长死区的 G1 速度指令）、
  `unilab.tasks.motion_tracking.common.manager_terms.MotionCommandCfg` /
  `MotionJointPositionActionCfg`（运动参考指令及其关节位置动作）。
- Curriculum term，例如
  `unilab.tasks.locomotion.g1.manager_terms.G1PenaltyCurriculum`（声明在
  `curriculum:` 段；当被跟踪的平均回合长度越过
  `level_down_threshold` / `level_up_threshold` 时，按 `degree` 步进在
  `initial_scale`/`min_scale` 与 `max_scale` 之间缩放负权重 reward 项）。
- 事件（域随机化与 reset）term，见 `unilab.envs.mdp.events`，声明在
  `env.events` 段。
- Reset owner，例如
  `unilab.tasks.locomotion.g1.manager_terms.G1WalkResetOwner`。

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Reward term
:link: 1-rewards
:link-type: doc
框架级、locomotion、G1 与 motion tracking 的 reward term 及参数。
:::

:::{grid-item-card} Observation term
:link: 2-observations
:link-type: doc
状态、传感器、指令、步态相位与射线扫描观测 term。
:::

:::{grid-item-card} Termination term
:link: 3-terminations
:link-type: doc
超时与失败终止条件，含融合终止包。
:::

::::

```{toctree}
:hidden:

1-rewards
2-observations
3-terminations
```
