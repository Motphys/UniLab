# 仿真后端

UniLab 目前在 registry/config 路径中使用两个后端名称：`1-mujoco` 和
`2-motrix`。用户命令通过 `--sim` 选择它们，该选项会路由到对应的
task owner YAML；不要仅靠 override `training.sim_backend` 来切换一次运行。

## 运行时前置条件

- 使用 `uv sync --extra motrix` 安装 Motrix 支持。
- 任何使用 `--sim mujoco`、MuJoCo 回放或仅限 MuJoCo 的调试工具的运行，
  仍然需要一个可用的 MuJoCo 运行时。
- 在 macOS 上，软件包 CLI 在需要时会通过 `mxpython` 路由 Motrix 交互式回放。
  直接打开原生 Motrix 渲染器的脚本调用应使用 `uv run mxpython`。

## 选择后端

```bash
uv run train --algo ppo --task go1_joystick_flat --sim mujoco
uv run train --algo ppo --task go1_joystick_flat --sim motrix
```

Owner YAML 位置：

- PPO / APPO：`conf/{ppo,appo}/task/<task>/<backend>.yaml`
- Off-policy：`conf/offpolicy/task/<algo>/<task>/<backend>.yaml`

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

```{toctree}
:hidden:

1-mujoco
2-motrix
3-choosing_a_backend
```
