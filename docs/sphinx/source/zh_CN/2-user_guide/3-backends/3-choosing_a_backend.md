# 选择后端

语言: 简体中文

UniLab 通过 task owner config 选择仿真器。常规用法下，使用 `--task` 和 `--sim`
选择 task 和 backend；off-policy 命令将算法保留在 `--algo` 中，而不是 `--task` 中。
不要仅靠 override `training.sim_backend` 来切换一次运行；该字段由 owner YAML 设置，
用于标识所组合的后端。

## 快速选择

| 需求 | 推荐 |
| --- | --- |
| 默认路径或最广的 owner 覆盖 | MuJoCo |
| 通过后端进行原生交互式回放 | Motrix |
| 仅限 MuJoCo 的工具，例如 `scripts/play_viser.py` | MuJoCo |
| task owner 仅以 `conf/.../<task>/mujoco.yaml` 形式存在 | MuJoCo |
| task owner 以 `conf/.../<task>/motrix.yaml` 形式存在，且支持矩阵将该组合标记为 tested 或 configured | Motrix |

支持矩阵由 registry、owner YAML 和测试生成；将其作为当前证据来源：
{doc}`/zh_CN/5-reference/5-support_matrix`。

## 示例

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco
uv run train --algo ppo --task go2_joystick_flat --sim motrix
uv run train --algo sac --task g1_walk_flat --sim mujoco
```

`registry.make(..., sim_backend=None)` 在 `src/unilab/base/registry.py` 中解析默认后端；
`--task` 和 `--sim` 仍然是面向用户的路线。

## Navigation

- Index: [文档](0-index.md)
