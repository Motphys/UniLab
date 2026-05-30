# Backend 能力契约

语言: 简体中文

Backend 差异是契约边界，而不是脚本层面的特殊处理。play/render 的决策记录在
{doc}`/adr/ADR-0002-backend-capability-boundary-for-play-and-snapshot`。

## 稳定的 Backend 接口

所有面向 env 的 backend 调用都应经由 `src/unilab/base/backend/base.py` 中的
`SimBackend`。该接口包括 base 状态、DOF 状态、世界系与 baselink 系下的 body
状态、具名 sensor、状态 reset、物理 stepping、domain-randomization hook 以及
可选的 playback/render 方法。

可选能力是显式声明的：

- `BackendPlayCapabilities` 报告对原生交互式渲染、物理状态 playback 以及原生
  视频录制的支持情况。
- `BackendHeightScanner` 与 `create_hfield_scanner(...)` 通过一个可复用的、由
  backend 拥有的对象暴露地形扫描支持。
- Domain randomization 支持通过 `get_dr_capabilities()` 以及 init、reset、
  interval 随机化方法对外暴露。
- 不支持的可选方法会从基类抛出 `NotImplementedError`。

## 新增能力的规则

- 如果共享的 env 逻辑需要某个新的 backend 操作，先将其加入 `SimBackend`；若并非
  每个 backend 都能立即支持，则默认抛出 `NotImplementedError`。
- 将 MuJoCo/Motrix 差异保留在 backend 实现、env 适配器与 owner YAML 中。不要在
  env 代码中加入对 backend 私有方法的热路径探测。
- 资源/XML/模型元数据的访问属于冷路径，例如场景 materialization、backend init
  或 cache 创建。

## 仓库中的证据

- Backend 接口与 play 能力：`src/unilab/base/backend/base.py`
- Backend 工厂：`src/unilab/base/backend/__init__.py`
- MuJoCo backend：`src/unilab/base/backend/mujoco/backend.py`
- Motrix backend：`src/unilab/base/backend/motrix/backend.py`
- Backend 契约测试：`tests/base/test_sim_backend.py`、
  `tests/base/test_backend_imports.py`、`tests/base/test_motrix_backend_options.py`

## Navigation

- Index: [文档](0-index.md)
