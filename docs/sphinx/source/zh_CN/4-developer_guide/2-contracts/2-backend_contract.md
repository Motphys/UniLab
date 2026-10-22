# Backend 能力契约

Backend 差异是契约边界，而不是脚本层面的特殊处理。play/render 的决策记录在
{doc}`/adr/ADR-0002-backend-capability-boundary-for-play-and-snapshot`。

## 稳定的 Backend 接口

所有面向 env 的 backend 调用都应经由 `unisim.backend.base` 中的
`SimBackend`。该接口包括 base 状态、DOF 状态、世界系与 baselink 系下的 body
状态、具名 sensor、状态 reset、物理 stepping、domain-randomization hook 以及
可选的 playback/render 方法。

可选能力是显式声明的：

- `PlayCapabilities` 报告对原生交互式渲染、物理状态 playback 以及原生
  视频录制的支持情况。
- `HeightScanner` 与 `bind_height_scanner(...)` 通过一个可复用的、由
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

对于 tensor 能力可用的标量 reset，`TensorLifecycleCapabilities.packed_reset` 是 Manager-Based 运行时的必需 capability，而不是可选 fast path：backend 未声明该能力时 `ManagerBasedRlEnv` 在初始化即 fail closed；缺少该声明或缺少 `SimBackend.bind_packed_reset()` 时，`ResetStateTransaction.declare_packed_reset_device` 抛出 `NotImplementedError`。Manager 将全部 reset-mode 状态、模型字段与 mocap 写入组合进 UniSim 公共 packed reset 事务；UniSim 拥有全宽 device 状态 staging、选中列合并、已提交模型字段 baseline、声明 `packed_reset_mocap` 时的选中 mocap staging，以及唯一公共提交边界。composer 按声明 device 缓存复用，其 baseline 跨 reset 持久；Manager 只保留面向 term 的校验与行域所有权。不存在 non-packed fallback：UniLab 不再消费 `SimBackend.set_state_tensor_mask()`，未声明 mocap 能力的 backend 上 tensor mocap 写入快速失败。

Owner-subset command 生命周期仍以 fixed-width bool mask 作为输入 carrier，记录实际写入 mask 的并集，并只请求 packed composer 提交这些行；未写入的 active rows 保留 backend 状态。

`tests/envs/test_multi_entity_consumer.py` 为 MuJoCo 和 IsaacSim 注册同一个 primitive task，并使用同一个可 pickle 的 EnvFactory。测试检查观测/动作维度、被动关节、局部 reset、variants 和 kinematic mirror。portable-profile fixture 还组合 robot、被动 object、table 与 collision-free mirror，并使用非 round-robin 的 N5/K2 assignment `[1,1,0,1,0]`。原生 IsaacSim case 通过 `UNILAB_TEST_M2_ISAACSIM=1` 启用。消费层 contract 要求已发布的 `unisim-core>=1.7.3`。wheel-variant lock 解析已发布的 registry 包。实现和验证边界见 [UniSim roadmap #154](https://github.com/unilabsim/unisim/issues/154) 与 [UniSim contract #155](https://github.com/unilabsim/unisim/issues/155)。

## 仓库中的证据

- 配置与资产准备：`src/unilab/base/scene.py`、`src/unilab/base/backend_factory.py`。
- 公共状态/reset 绑定：`src/unilab/base/entity.py`、`src/unilab/base/reset_state.py`。
- 注册 runtime 测试：`tests/base/test_entity_scene_consumer.py`、`tests/envs/test_multi_entity_consumer.py`。

- Backend 接口与 play 能力：`unisim.backend.base`
- Backend 工厂：`src/unilab/base/backend_factory.py`
- MuJoCo backend：`unisim.backend.mujoco.backend`
- Motrix backend：`unisim.backend.motrix.backend`
- Backend 契约测试：`tests/base/test_backend_conformance.py`、
  `tests/base/test_sim_backend_smoke.py`、
  `tests/base/test_backend_imports.py`、`tests/base/test_motrix_backend_options.py`
