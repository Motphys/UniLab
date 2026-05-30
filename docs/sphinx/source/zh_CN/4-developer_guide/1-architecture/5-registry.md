# Registry Bootstrap

语言: 简体中文

Registry bootstrap 是一个针对环境的显式导入契约。它由
{doc}`/adr/ADR-0004-registry-bootstrap-contract` 定义，并在
`src/unilab/base/registry.py` 中实现。

## 运行时流程

1. 训练入口调用 `unilab.training.common.ensure_registries()`。
2. 该 helper 委托给 `unilab.base.registry.ensure_registries()`。
3. registry 导入已声明的 bootstrap 包：
   `unilab.envs.locomotion`、`unilab.envs.manipulation` 与
   `unilab.envs.motion_tracking`。
4. 每个包都暴露 `__unilab_registry_modules__`，即一个包含注册副作用的模块元组。
5. 被导入的模块通过 `@registry.envcfg(...)` 注册 config，并通过
   `@registry.env(..., sim_backend=...)` 或 `registry.register_env(...)` 注册
   env 实现。
6. 运行时构造经由 `registry.make(...)`，它会应用 env config override、校验
   env config、选择所请求的 backend，并实例化已注册的 env 类。

## 扩展规则

- 如果新的 env 模块位于某个尚未被现有 bootstrap 条目导入的新模块中，需将其加入
  包级别的 `__unilab_registry_modules__` 元组。
- 保持注册过程轻量。场景 materialization、XML 处理、资源访问以及 backend 构造
  应放在 `registry.make(...)` 之后，而不是放在装饰器注册中。
- 重复的 env config 以及重复的 `(env, sim_backend)` 注册会在
  `src/unilab/base/registry.py` 中抛出 `ValueError`；请保留该失败边界。

## 仓库中的证据

- Bootstrap helper：`src/unilab/base/registry.py`
- 训练 helper：`src/unilab/training/common.py`
- 包声明：`src/unilab/envs/locomotion/__init__.py`、
  `src/unilab/envs/manipulation/__init__.py`、
  `src/unilab/envs/motion_tracking/__init__.py`
- 测试：`tests/base/test_registry.py`、`tests/utils/test_algo_utils.py`

## Navigation

- Index: [文档](0-index.md)
