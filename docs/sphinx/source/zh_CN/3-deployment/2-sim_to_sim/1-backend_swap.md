# 切换后端

UniLab 支持两个 CPU 物理后端：**MuJoCo**（通过 `mujoco-uni`）和
**Motrix**（通过 `motrixsim-core`）。两者实现了相同的 `SimBackend`
contract 和相同的 env contract。后端特有的行为通过显式方法和能力记录
（capability records）暴露出来。

| 维度 | MuJoCo | Motrix |
|---|---|---|
| 后端类 | `src/unilab/base/backend/mujoco/backend.py` | `src/unilab/base/backend/motrix/backend.py` |
| 回放能力 | `get_play_capabilities()` 中的物理状态回放 | `get_play_capabilities()` 中的原生交互式渲染器与原生视频采集 |
| 高度场扫描 | 实现 `create_hfield_scanner(...)` | 实现 `create_hfield_scanner(...)` |
| DR 能力上报 | `get_dr_capabilities()` | `get_dr_capabilities()` |

**切换后端的正当理由**是以下之一：

1. 目标任务已经有针对另一个后端的 owner YAML。
2. 该后端暴露了工作流所需的能力。
3. 你想在部署或更换后端之前做一次 sim-to-sim 一致性检查。

切换始终是一次任务 owner 的变更，而不是临时的运行时调整。

## 如何切换

UniLab **不**支持通过透传式（passthrough）Hydra override 来选择后端。后端是由
`--task` 和 `--sim` 选定的 *任务 owner 身份* 的一部分：

```bash
# wrong — backend is not an override
uv run train --algo ppo --task go2_joystick_flat --sim motrix training.sim_backend=mujoco

# right — choose the backend with --sim
uv run train --algo ppo --task go2_joystick_flat --sim mujoco
uv run train --algo ppo --task go2_joystick_flat --sim motrix
```

CLI 会把 `--algo`、`--task` 和 `--sim` 解析为一个 owner YAML，例如
`conf/ppo/task/go2_joystick_flat/mujoco.yaml`。如果该文件不存在，则说明该任务
**不支持**这个后端 —— 参见
{doc}`../../4-developer_guide/2-contracts/3-task_owner`。

## 另请参阅

- {doc}`2-owner_yaml_swap`
- {doc}`4-reward_parity`
- {doc}`6-capability_gaps`
