# Registry Bootstrap

Registry bootstrap 是一个针对环境的显式导入契约。它由
{doc}`/adr/ADR-0004-registry-bootstrap-contract` 定义，并在
`src/unilab/base/registry.py` 中实现。

## 运行时流程

1. 训练入口调用 `unilab.training.common.ensure_registries()`。
2. 该 helper 委托给 `unilab.base.registry.ensure_registries()`。
3. registry 依次从三个来源收集 bootstrap 包：内置的 `unilab.tasks`、第三方包
   通过 `[project.entry-points."unilab.tasks"]` 声明的 entry point（值为可导入
   包名）、以及 `UNILAB_EXTRA_REGISTRY_PACKAGES` 环境变量（主要供测试 fixture
   使用）。
4. 每个 bootstrap 包暴露 `__unilab_registry_modules__`，即一个包含注册副作用的
   task leaf module 显式元组。
5. 被导入的模块通过 `@registry.envcfg(...)` 注册 config，并通过
   `@registry.env(..., sim_backend=...)` 或 `registry.register_env(...)` 注册
   env 实现。
6. 运行时构造经由 `registry.make(...)`，它会应用 env config override、校验
   env config、选择所请求的 backend，并实例化已注册的 env 类。

## 扩展规则

- 如果新的 task leaf 尚未被现有 bootstrap 条目导入，需将其加入
  `unilab.tasks.__unilab_registry_modules__`。
- 仓库外的第三方任务包在自己的 `pyproject.toml` 中声明
  `[project.entry-points."unilab.tasks"]`（例如
  `microduck = "microduck_rl_unilab.tasks"`）完成自注册，被声明的包同样暴露
  `__unilab_registry_modules__`。entry-point 元数据位于 site-packages 中，
  spawn 出的 collector 子进程无需转发环境变量即可发现同样的包；entry-point
  包的导入失败按 fail-closed 处理（安装是显式行为）。
  `UNILAB_EXTRA_REGISTRY_PACKAGES` 环境变量仍保留，供测试 fixture 与临时
  调试使用。
- 保持注册过程轻量。场景 materialization、XML 处理、资源访问以及 backend 构造
  应放在 `registry.make(...)` 之后，而不是放在装饰器注册中。
- 重复的 env config 以及重复的 `(env, sim_backend)` 注册会在
  `src/unilab/base/registry.py` 中抛出 `ValueError`；请保留该失败边界。

## 仓库中的证据

- Bootstrap helper：`src/unilab/base/registry.py`
- 训练 helper：`src/unilab/training/common.py`
- Task bootstrap 声明：`src/unilab/tasks/__init__.py`
- 测试：`tests/base/test_registry.py`、`tests/utils/test_algo_utils.py`
