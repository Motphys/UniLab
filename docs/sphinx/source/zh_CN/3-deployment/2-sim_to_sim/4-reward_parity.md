# 跨后端的 Reward 一致性

两个拥有"相同" reward 函数的后端，很少会产生**数值上完全相同**的 reward ——
这没关系。你想要的是**轨迹级别的一致性（trajectory-level parity）**：同一个策略，
施加到相同的初始状态上，会产生相似的 reward 曲线。

## 协议

1. 冻结一个**固定种子**、固定的初始状态、固定的动作序列。
   动作序列可以是对各关节的正弦扫频，也可以是来自真实 rollout 的回放 ——
   只要是确定性的即可。
2. 在两个后端中回放它。从 `info["log"]` 记录每一步的 reward 分量；当 reward
   日志开启时，locomotion 的 reward 分发会在那里写入 `reward/<term>` 条目。
3. 对每个 reward 项，比较两条时间序列，并检查它们开始分叉的第一帧。

## 良好的一致性是什么样的

| 项的类型 | 检查什么 |
|---|---|
| 平滑惩罚项 | 单位、帧以及命令输入是否相同。 |
| 接触条件项 | 两个后端中的接触时序与传感器可用性。 |
| 终止惩罚项 | 终止掩码（termination mask）与最终观测（final-observation）路径。 |

## 哪些是危险信号

- **reward 在 episode 后期出现分叉。** 通常意味着策略在每个后端中探索了不同的
  状态分布，而这通常又意味着接触 / 摩擦不匹配。参见
  {doc}`3-contact_and_friction_alignment`。
- **某个 reward 项在某个后端中恒为零。** 能力缺口：该项读取了某个后端不暴露的
  特性。参见
  {doc}`6-capability_gaps`。

## 自动化

仓库目前没有在 `scripts/` 中提供独立的 reward 一致性辅助工具。在新增一致性覆盖时，
让测试贴近 backend/task-owner 边界：组合两份 owner YAML，用固定种子复位，回放一段
确定性的动作序列，并在 `tests/` 下对记录下来的 reward 分量做断言。

## 另请参阅

- {doc}`3-contact_and_friction_alignment`
- {doc}`../../4-developer_guide/2-contracts/2-backend_contract`
