# Hydra 配置

UniLab 使用基于 task owner YAML 的 Hydra 组合。owner YAML 是 task、backend、
reward、scene 以及 task 专属运行时字段的身份标识。

## Owner 路径

| 技术栈 | Owner YAML 形式 |
| --- | --- |
| PPO | `src/unilab/conf/ppo/task/<task>/<backend>.yaml` |
| APPO | `src/unilab/conf/appo/task/<task>/<backend>.yaml` |
| SAC / FlashSAC | `src/unilab/conf/<algo>/task/<task>/<backend>.yaml` |

示例：

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco
uv run train --algo ppo --task go2_joystick_flat --sim mujoco
uv run train --algo sac --task g1_walk_flat --sim mujoco
```

对于 off-policy，`--algo` 选择按算法划分的配置树 `src/unilab/conf/<algo>/`；不要在
`--task` 中包含算法名称。

## 安全的 Override

Hydra override 可以调整所选 owner 路径内部的字段：

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco \
  algo.max_iterations=10 \
  algo.num_envs=128 \
  training.no_play=true
```

常用字段：

- `algo.max_iterations`
- `algo.num_envs`
- `algo.load_run`
- `algo.seed`
- `training.no_play`
- `training.tf32_matmul`（仅 PPO 与 APPO owner config 提供）
- `training.play_only`
- `training.play_render_mode`
- `training.logger`

`training.tf32_matmul` 控制 learner 上 float32 矩阵乘使用的后端。`ieee`
（默认）保持全精度 FP32；`tf32` 启用 TF32 tensor-core kernel，并在 macOS/MPS、
ROCm/HIP、pre-Ampere CUDA GPU（compute capability < 7.0）与 CPU 上自动回退为
IEEE。解析结果与回退原因会写入 `run_config.json` 与 `run_summary.json`。该键
只存在于 PPO 与 APPO owner config：收益来自它们的 FP32 更新路径；SAC 家族的
主要 matmul 已经由 bf16 AMP 接管，#2477 实测 TF32 在该路径为噪声级。

## 查看完整 compose 结果

调试组合时，可在命令末尾追加 `--cfg job`，打印完整组合后的配置而不真正运行训练：

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco --cfg job
```

## 后端身份

`training.sim_backend` 是由所选 owner YAML 设置的身份字段。它不是一个独立的后端切
换开关。请使用统一 CLI 的 `--sim` flag 来选择后端。

参见 {doc}`../../4-developer_guide/2-contracts/3-task_owner` 中的开发者 contract。

## 按算法的配置参考

逐字段表格 —— 训练效率、策略性能（GAE lambda、熵系数、学习率、网络结构）、训练
运行与 reward 权重 —— 见各算法页：

- {doc}`../2-algorithms/1-ppo`
- {doc}`../2-algorithms/2-appo`
- {doc}`../2-algorithms/3-sac`
- {doc}`../2-algorithms/4-flash_sac`
