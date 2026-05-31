# 扩展 UniLab：新后端

在添加后端代码之前，请先阅读 {doc}`../2-contracts/2-backend_contract` 与
{doc}`/adr/ADR-0002-backend-capability-boundary-for-play-and-snapshot`。

## 当前后端形态

仓库目前在两个重要位置识别 `mujoco` 与 `motrix`：

- `src/unilab/base/registry.py` 中的 `registry.register_env(...)`
- `src/unilab/base/backend/__init__.py` 中的 `create_backend(...)`

因此，引入第三个后端是一项架构变更，而不仅仅是新增一个包。

## 实现清单

1. 在 `src/unilab/base/backend/<backend>/` 下实现一个 `SimBackend` 子类。
2. 把后端构造逻辑加入 `create_backend(...)`。
3. 如果新后端应当被 `@registry.env(..., sim_backend=...)` 接受，
   则更新 registry 校验逻辑。
4. 通过 `SimBackend` 方法、`BackendPlayCapabilities`、域随机化能力方法，
   或新的显式抽象方法，暴露后端特定的可选能力。
5. 为受支持的 task/backend 组合添加任务 owner YAML。让面向用户的示例
   走 `--task <task> --sim <backend>`，不要依赖
   `training.sim_backend=<backend>` 作为 override。
6. 把 XML、asset 与 model 检查保留在后端 init 或 materialization 代码中。
   热路径的 env 应当接收缓存好的 ID、数组或已声明的后端方法。

## 在风险点附近验证

- 后端导入与接口测试：`tests/base/test_backend_imports.py`、
  `tests/base/test_sim_backend.py`
- 特性边界附近的后端特定行为测试，例如
  `tests/base/test_motrix_backend_options.py`
- Task/backend config 组合：`tests/config/test_config_system.py`

## 仓库内证据

- 后端接口：`src/unilab/base/backend/base.py`
- 后端工厂：`src/unilab/base/backend/__init__.py`
- MuJoCo 后端包：`src/unilab/base/backend/mujoco/`
- Motrix 后端包：`src/unilab/base/backend/motrix/`
