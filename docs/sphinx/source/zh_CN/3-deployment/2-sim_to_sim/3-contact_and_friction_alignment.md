# 在后端之间对齐接触与摩擦

语言: 简体中文

接触处理是 MuJoCo 与 Motrix 之间产生漂移的常见来源。本页教你如何识别并消除
这一差距。

## 诊断：尽早探测接触

在两个后端中运行同一份 owner YAML、相同的复位种子（reset seed）以及确定性的
动作序列。从 `info["log"]` 记录依赖接触的 reward 项，以及 env 已经使用的任何
后端拥有的接触信号。如果在策略学习介入之前，名义接触响应就已经不同，那么先修正
摩擦 / 阻尼 / 恢复系数（restitution）的声明，再去调试 reward 一致性。

## 常见陷阱

| 陷阱 | 表现 | 修复 |
|---|---|---|
| 在 MuJoCo 中摩擦声明在 geom 上，而在 Motrix 中声明在 material 上 | 某个后端出现 μ=0 | 两边都声明；在场景导出中核对 |
| Solver 迭代次数过低 | MuJoCo 陷入地面 | 在 owner YAML 中调高后端拥有的 solver 设置 |
| 接触对特定的 override | 打滑不一致 | 在两个后端 YAML 中都显式写出接触对级别的 override |
| 恢复系数不匹配 | 落地一边弹跳、一边粘滞 | 显式设置；默认值不同 |

## 对齐 DR 范围

一旦**名义**参数的接触已经对齐，就审查 DR：

- `friction_static_mu` 和 `friction_dynamic_mu` 在两个后端中应当以相同方式
  采样。
- 如果某个 DR 字段在一个后端中是空操作（no-op），就在 episode 初始化时记录一条
  警告。无声的忽略会导致无声的 reward 漂移。

## 另请参阅

- {doc}`4-reward_parity`
- {doc}`../1-sim_to_real/6-domain_randomization`
- {doc}`6-capability_gaps`

## Navigation

- Index: [文档](0-index.md)
