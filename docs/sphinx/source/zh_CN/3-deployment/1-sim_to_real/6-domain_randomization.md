# 面向真机迁移的域随机化

本页是域随机化的部署检查清单。关于**契约**层（一个 DR provider 必须实现什么），见
{doc}`../../4-developer_guide/2-contracts/4-dr_contract`。

## 随机化什么，按优先级排序

```{list-table}
:header-rows: 1
:widths: 25 30 45

* - 类别
  - 示例
  - 为什么重要
* - 执行器动力学
  - PD 增益、动作缩放、任务 owner 启用时的单步动作延迟
  - 硬件上策略振荡的首要驱动因素。
* - 质量 / 惯量
  - 躯干质量、连杆质心偏移、负载
  - 影响平衡与跟踪裕度。
* - 摩擦
  - 足 ↔ 地面 μ、手 ↔ 物体 μ
  - 手内方块任务没有它就会失败。
* - 观测噪声
  - IMU 噪声、关节编码器偏置、部署侧观测历史
  - 让 actor 输入贴近部署侧传感器行为。
* - 外力
  - 推力、阵风、对负载的拖拽
  - 对未建模扰动的鲁棒性。
* - 复位状态
  - 初始姿态、初始速度
  - 降低在回合边界处的脆弱性。
```

::::{admonition} 经验法则
:class: tip
如果某个参数会实质性地影响闭环响应，而你又没有部署侧的测量，那就不要把相关结论写进
文档；只有在记录了为何该范围合理之后，才在任务 owner 中编码一个保守范围。
::::

## UniLab 如何组织 DR

使用 DR 的任务通过环境初始化路径挂接一个 provider：

```python
from unilab.envs.locomotion.common.dr_provider import LocomotionDRProvider

class MyTaskEnv(NpEnv):
    def __init__(self, cfg):
        super().__init__(cfg)
        self._init_domain_randomization(LocomotionDRProvider(cfg.domain_rand))
```

管理器位于 `src/unilab/dr/manager.py`；provider 位于其环境 owner 附近，并遵循
{doc}`../../4-developer_guide/2-contracts/4-dr_contract` 中的契约。

## 配方：起始范围

把所选的 owner YAML 作为权威来源。例如，
`conf/ppo/task/go2_joystick_rough/mujoco.yaml` 启用了基座质量、质心、kp/kd 以及推力
随机化；`conf/ppo/task/sharpa_inhand/mujoco.yaml` 配置了 PD 增益、摩擦、质心、质量、
关节噪声与接触噪声字段。

```yaml
# conf/ppo/task/go2_joystick_rough/mujoco.yaml
env:
  domain_rand:
    randomize_base_mass: true
    added_mass_range: [-1.0, 3.0]
    random_com: true
    randomize_kp: true
    kp_multiplier_range: [0.5, 2.0]
    randomize_kd: true
    kd_multiplier_range: [0.5, 2.0]
    push_robots: true
    push_interval: 625
```

## 课程：随技能逐步加大 DR

在第 0 步就过于激进的 DR 会让学习停滞。UniLab 的课程辅助工具由任务自身拥有；把它们的
字段保留在所选的 owner YAML 中，不要在训练脚本里添加 Python 侧的解释。

## 验证 DR 覆盖

训练之后，在配置中扫动 DR 范围的同时，对照同一后端 owner YAML 回放检查点：

```bash
uv run eval --algo ppo --task go2_joystick_flat --sim motrix --load-run -1
```

为每个扫动点记录奖励分量与任务成功指标。一次陡降或奖励分量的不连续，就是 DR 范围
改变了任务契约、而不仅仅是拓宽部署覆盖的证据。

## 另请参阅

- {doc}`../../2-user_guide/5-domain_randomization/0-index`
- {doc}`../../4-developer_guide/2-contracts/4-dr_contract`
- {doc}`../2-sim_to_sim/3-contact_and_friction_alignment`
