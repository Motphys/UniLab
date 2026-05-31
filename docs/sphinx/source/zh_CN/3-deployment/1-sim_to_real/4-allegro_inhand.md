# Allegro / Sharpa 手内操作部署

在 16 自由度的 Allegro 手或 17 自由度的 Sharpa 手上进行方块重定向。UniLab 以无触觉
方式训练它们 —— 观测仅为关节状态 + 方块位姿。

## 难点在哪

手内操作是对**摩擦与接触敏感**的任务。在以下方面的微小偏差：

- 方块边缘几何（圆角半径、表面粗糙度）
- 手指接触垫摩擦（取决于温度和湿度！）
- 关节回差

都可能把策略推到它在仿真中未曾见过的工况之外。使用任务自带的域随机化配置，并在硬件
上机前于仿真中验证这些范围；见 {doc}`6-domain_randomization`。

## 观测契约

```{list-table}
:header-rows: 1
:widths: 30 15 55

* - 分组
  - 维度
  - 硬件上的来源
* - 关节位置
  - 16（Allegro）/ 17（Sharpa）
  - 编码器
* - 关节速度
  - 16 / 17
  - 编码器差分，低通滤波
* - 方块位姿（世界系）
  - 7
  - 视觉（RGB-D + 位姿估计器，或基准标记）
* - 方块线速度/角速度
  - 6
  - 位姿的有限差分，低通滤波；**在真机上有噪声**
* - 目标旋转四元数
  - 4
  - 指令
* - 上一步动作
  - 16 / 17
  - 上一次策略输出
```

::::{admonition} 视觉管线延迟
:class: warning
位姿估计延迟取决于具体的部署栈。在硬件观测构建器中测量它，然后在硬件部署前让训练
owner 与部署运行时在观测时序上达成一致。见
{doc}`8-latency_budget`。
::::

## 抓取生成器

`4-allegro_inhand` 与 `sharpa_inhand` 两个环境都自带一个**抓取生成器**，用于采样
合理的初始手部构型。硬件侧的等价物是操作员把方块放到手里 —— 请核实你的起始构型分布
与训练环境的抓取生成器输出相匹配（参见
`unilab.envs.manipulation.allegro_inhand.grasp_gen`）。

如果你真实世界的起始握持存在系统性差异，**把这些位姿加入抓取生成器**，重新训练，
然后再试。

## 动作接口

操作类环境通过任务控制配置把策略动作映射为关节位置目标
（`src/unilab/envs/manipulation/allegro_inhand/base.py` 和
`src/unilab/envs/manipulation/sharpa_inhand/base.py`）。部署控制器必须使用相同的
关节顺序、动作缩放与限位策略。

## 失败恢复

掉落在不重新抓取的情况下无法恢复。硬件侧安全层应当持有掉落检测器，使用部署栈实际
提供的任何方块位姿或力感知。一旦检测到，进入控制器的安全状态并告警操作员。

## 另请参阅

- {doc}`5-onnx_runtime`
- {doc}`6-domain_randomization`
- {doc}`../../2-user_guide/8-manipulation/1-dexterous_inhand`
