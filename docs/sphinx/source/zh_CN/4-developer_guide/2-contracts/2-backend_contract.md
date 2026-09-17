# Backend 能力契约

Backend 差异是契约边界，而不是脚本层面的特殊处理。play/render 的决策记录在
{doc}`/adr/ADR-0002-backend-capability-boundary-for-play-and-snapshot`。

## 稳定的 Backend 接口

所有面向 env 的 backend 调用都应经由 `unisim.backend.base` 中的
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

## 物理实体与局部 reset

Issue #1599 的 M2 消费层将 UniSim 物理实体声明与 UniLab 逻辑 selector 分开。`SceneCfg` 物化实体/variant 类型并调用父契约校验，asset factory 收集物理源和 catalog 源路径。`EntityCfg.physical_entity` 显式绑定逻辑 facade；`primary_entity` 选择场景主根，不把任务名称写进 backend。

映射的逻辑 root 必须精确指向物理实体声明的 root。初始化时拒绝将后代 body 绑定为 root，确保 root 查询、默认值和 reset 写入引用同一对象。Reset 暂存以线性时间映射选中行，并只保存请求的字段；只有合并不同 joint position/velocity 选择、需要补齐未写列时才读取当前状态快照。事务仍先完整校验，再调用一次公共 backend 提交。

既有 `ResetStateTransaction` 为 mapped scene 暂存一次公共 `SceneResetRequest`。缺失字段、未选实体和环境保持不变。逐环境默认值来自 `get_entity_default_state`，`restore_default_controls` 在同次提交中恢复 keyframe control，控制值不必等于关节位置。Manager term 不接触引擎私有 tensor 或资产解析。当前消费边界为标量 hinge/slide 和一次事务共用选中环境集合；不支持的 DR/mocap 混写或行模式明确拒绝。

`tests/envs/test_multi_entity_consumer.py` 为 MuJoCo 和 IsaacSim 注册同一个 primitive task，并使用同一个可 pickle 的 EnvFactory。测试检查观测/动作维度、被动关节、局部 reset、variants 和 kinematic mirror。原生 IsaacSim case 通过 `UNILAB_TEST_M2_ISAACSIM=1` 启用。消费层要求已发布的 `unisim-core>=1.5.0`；标准与 ROCm 锁文件均解析 PyPI 包，不使用 Git source 覆盖。`UNILAB_LOCAL_UNISIM` 仍是显式本地开发替代方案。实现和验证边界见 [UniSim roadmap #108](https://github.com/unilabsim/unisim/issues/108)。

## 仓库中的证据

- 配置与资产准备：`src/unilab/base/scene.py`、`src/unilab/base/backend_factory.py`。
- 公共状态/reset 绑定：`src/unilab/base/entity.py`、`src/unilab/base/reset_state.py`。
- 注册 runtime 测试：`tests/base/test_entity_scene_consumer.py`、`tests/envs/test_multi_entity_consumer.py`。

- Backend 接口与 play 能力：`unisim.backend.base`
- Backend 工厂：`src/unilab/base/backend_factory.py`
- MuJoCo backend：`unisim.backend.mujoco.backend`
- Motrix backend：`unisim.backend.motrix.backend`
- Backend 契约测试：`tests/base/test_sim_backend.py`、
  `tests/base/test_backend_imports.py`、`tests/base/test_motrix_backend_options.py`
