# 仿真后端

UniLab 通过 registry/config 路径暴露后端名称，包括在对应 owner 注册后可用的
`mujoco`、`motrix`、`mjwarp`、`drake`、`isaacgym`、`genesis`、`isaacsim` 和
`newton`。用户命令通过
`--sim` 选择后端，该选项会路由到对应的 task owner YAML；不要仅靠 override
`training.sim_backend` 来切换一次运行。

## 运行时前置条件

- 使用 `uv sync --extra motrix` 安装 Motrix 支持。
- IsaacGym 和 IsaacSim 使用独立的 worker 运行时；安装和运行时要求见各自的
  后端页面。
- 任何使用 `--sim mujoco`、MuJoCo 回放或仅限 MuJoCo 的调试工具的运行，
  仍然需要一个可用的 MuJoCo 运行时。
- Drake 需要外部 `drake-uni` Python 包和针对本地 Drake 编译的 C++ 批量扩展；
  选择 `--sim drake` 前请先阅读 {doc}`6-drake`。
- Newton 使用隔离的 `newton` extra（`uv sync --extra newton`），不能与
  `mujoco` / `mjwarp` extra 装进同一环境；选择 `--sim newton` 前请先阅读
  {doc}`7-newton`。
- 在 macOS 上，软件包 CLI 在需要时会通过 `mxpython` 路由 Motrix 交互式回放。
  直接打开原生 Motrix 渲染器的脚本调用应使用 `uv run mxpython`。

## 选择后端

UniLab 通过 task owner config 选择仿真器。常规用法下，使用 `--task` 和 `--sim`
选择 task 和 backend；off-policy 命令将算法保留在 `--algo` 中，而不是 `--task` 中。
不要仅靠 override `training.sim_backend` 来切换一次运行；该字段由 owner YAML 设置，
用于标识所组合的后端。

### 快速选择

| 需求 | 推荐 |
| --- | --- |
| 默认路径或最广的 owner 覆盖 | MuJoCo |
| 通过后端进行原生交互式回放 | Motrix |
| 仅限 MuJoCo 的工具，例如 `scripts/play_viser.py` | MuJoCo |
| task owner 仅以 `src/unilab/conf/.../<task>/mujoco.yaml` 形式存在 | MuJoCo |
| task owner 以 `src/unilab/conf/.../<task>/motrix.yaml` 形式存在，且支持矩阵将该组合标记为 tested 或 configured | Motrix |

支持矩阵由 registry、owner YAML 和测试生成；将其作为当前证据来源：
{doc}`../../5-reference/5-support_matrix`。

```bash
uv run train --algo ppo --task go1_joystick_flat --sim mujoco
uv run train --algo ppo --task go1_joystick_flat --sim motrix
uv run train --algo ppo --task g1_walk_flat --sim isaacsim
```

更多组合示例：

```bash
uv run train --algo ppo --task stewart_balance --sim drake \
  algo.max_iterations=1 algo.num_envs=8 training.no_play=true
uv run train --algo sac --task g1_walk_flat --sim mujoco
```

Owner YAML 位置：

- PPO / APPO：`src/unilab/conf/{ppo,appo}/task/<task>/<backend>.yaml`
- Off-policy（SAC / TD3 / FlashSAC）：`src/unilab/conf/<algo>/task/<task>/<backend>.yaml`

被选中的 owner YAML 将 `training.sim_backend` 设为身份字段。

## 回放差异

- `--render-mode auto` 在 MuJoCo 路径上导出 `play_video.mp4`。
- `--render-mode auto` 在 Motrix 路径上打开 Motrix 原生交互式渲染。
- `--render-mode record` 在不打开交互式窗口的情况下录制。
- `--render-mode none` 禁用回放。

```bash
uv run eval --algo ppo --task go1_joystick_flat --sim mujoco --load-run -1
uv run eval --algo ppo --task go1_joystick_flat --sim motrix --load-run -1 \
  --render-mode record
```

## 支持证据

Task/backend/entrypoint 的支持情况是按证据分级的。请参阅
{doc}`../../5-reference/5-support_matrix` 获取支持矩阵条目以及指向所生成源数据的链接。

## 相关 contract

- {doc}`Backend contract </zh_CN/4-developer_guide/2-contracts/2-backend_contract>`
- {doc}`Task owner contract </zh_CN/4-developer_guide/2-contracts/3-task_owner>`
- {doc}`Backend capability boundary ADR </adr/ADR-0002-backend-capability-boundary-for-play-and-snapshot>`
- {doc}`Registry bootstrap ADR </adr/ADR-0004-registry-bootstrap-contract>`

## unisim-core 边界

UniLab 的物理后端由独立的 `unisim-core` distribution 提供，Python namespace
为 `unisim`。安装和导入示例：

```bash
uv sync --extra mujoco
uv run python -c "import unisim; print(unisim.ADAPTER_SPECS)"
```

`unisim` 不依赖 UniLab、Hydra 或训练组件；MuJoCo、Motrix、Drake、MJWarp、Genesis、
IsaacGym、IsaacSim 和 Newton 都通过同一个公开 contract 暴露。专有 SDK 或 GPU worker 缺失时，
构造 backend 会在冷路径给出明确诊断，不会静默回退到另一引擎。

UniLab 只保留 `unilab.base.backend_factory` 这一 owner-layer 组装入口；contract
和 adapter 从 `unisim` 导入。原 `unilab.base.backend` 实现及兼容层已经删除，
不要在 UniLab 中新增 backend API。

benchmark v1 目前只保留 `BenchmarkCase`、`BenchmarkResult` 和 provenance schema，
不包含 workload、计时或性能结论。

```{toctree}
:hidden:

1-mujoco
2-motrix
3-isaacgym
4-isaacsim
5-genesis
6-drake
7-newton
```
