# 编写 Provider

语言: 简体中文

任务级域随机化 provider 与 task env owner 放在一起。它们采样任务专属的
状态，并返回由 `DomainRandomizationManager` 消费的 plan。

## Provider 形态

当前的 provider 示例定义了以下 plan 方法中的一个或多个：

- 为模型变体或几何 materialization 构建 init plan。
- 返回带有状态更新和 reset 随机化 payload 的 reset plan。
- 返回用于 push 或 body-force 扰动的 interval plan。

共享类型位于 `src/unilab/dr/types.py`，manager 位于
`src/unilab/dr/manager.py`。

## 规则

- 将 XML、asset 和模型元数据访问保留在冷路径上，例如 init、
  materialization 或 cache 创建。
- 不要从 env 热路径探测后端私有方法。
- 只 dispatch 后端通过其 DR 能力声明的字段。
- 将任务专属采样放在 task provider 中，而不是训练脚本中。

## 证据

具有代表性的 provider 实现位于：

- `src/unilab/envs/locomotion/go1/joystick.py`
- `src/unilab/envs/locomotion/g1/joystick.py`
- `src/unilab/envs/motion_tracking/g1/tracking.py`
- `src/unilab/envs/manipulation/allegro_inhand/rotation.py`
- `src/unilab/envs/manipulation/sharpa_inhand/rotation.py`

开发者 contract 详情见
{doc}`../../4-developer_guide/2-contracts/4-dr_contract`。

## Navigation

- Index: [文档](0-index.md)
