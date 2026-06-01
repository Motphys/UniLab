# Hydra 配置

UniLab 使用基于 task owner YAML 的 Hydra 组合。owner YAML 是 task、backend、
reward、scene 以及 task 专属运行时字段的身份标识。

## Owner 路径

| 技术栈 | Owner YAML 形式 |
| --- | --- |
| PPO | `conf/ppo/task/<task>/<backend>.yaml` |
| MLX PPO | `conf/ppo/task/<task>/<backend>.yaml`，搭配 `conf/ppo/config_mlx.yaml` |
| APPO | `conf/appo/task/<task>/<backend>.yaml` |
| SAC / TD3 / FlashSAC | `conf/offpolicy/task/<algo>/<task>/<backend>.yaml` |
| HIM-PPO | `conf/ppo_him/task/<task>/<backend>.yaml` |
| HORA 蒸馏 | `conf/hora_distill/task/<task>/<backend>.yaml` |

示例：

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco
uv run train --algo ppo --task go2_joystick_flat --sim motrix
uv run train --algo sac --task g1_walk_flat --sim mujoco
```

对于 off-policy，`--algo` 选择 `conf/offpolicy/task/<algo>/` 下 owner 路径的第一
个分段；不要在 `--task` 中包含算法名称。

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
- `training.play_only`
- `training.play_render_mode`
- `training.logger`

## 查看完整 compose 结果

调试组合时，可在命令末尾追加 `--cfg job`，打印完整组合后的配置而不真正运行训练：

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco --cfg job
```

## 后端身份

`training.sim_backend` 是由所选 owner YAML 设置的身份字段。它不是一个独立的后端切
换开关。请使用统一 CLI 的 `--sim` flag 来选择后端。

参见 {doc}`../../4-developer_guide/2-contracts/3-task_owner` 中的开发者 contract。
