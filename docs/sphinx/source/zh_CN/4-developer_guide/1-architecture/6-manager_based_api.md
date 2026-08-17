# Manager-Based API

UniLab 采用“社区兼容 API + UniLab NumPy runtime”：manager-facing 模块、term cfg、
function/class term、生命周期和顺序语义以固定的 mjlab 1.6.0 source 为基线；数值实现使用
NumPy，并保留现有 `NpEnvState`、Hydra owner YAML、`SimBackend`、registry 与 IPC contract。

完整决策、兼容矩阵和机械迁移示例见
{doc}`/adr/ADR-0006-community-manager-api-on-numpy-runtime`。

## 不变量

- 公共 manager 结构优先保持社区语义；不为局部性能制造 UniLab-only term API。
- manager buffer、term return、env ID 和 entity view 使用 `np.ndarray` / `slice`，core 不依赖
  Torch、Warp、runner、learner 或 IPC。
- `SceneEntityCfg` 在冷路径通过 base scene/entity facade 解析；facade 只调用正式
  `SimBackend` contract，热路径复用缓存 ID/view。
- 用户显式空配置可以使用 Null manager；配置请求但 runtime/backend 不支持的能力必须在
  最近边界报错，不能 warning、skip、返回零或回退旧 env。
- 热路径避免明显的重复解析、逐环境 Python 循环、复制和临时分配；进一步优化需要
  benchmark 证明收益，且不能增加不成比例的结构复杂度。

只有被注册、配置和测试覆盖的表面才能声明 Compatible。NumPy/env/config adapter 标为
Adapted；缺少正式 backend contract 的能力标为 Unsupported 并 fail-closed。
