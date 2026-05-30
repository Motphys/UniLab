# 任务 owner config 契约

语言: 简体中文

任务 owner YAML 是一条已 compose 的 任务/backend/算法 路径的身份。该契约记录在
{doc}`/adr/ADR-0003-task-owner-and-config-compose-contract`。

## Owner 路径

- PPO 与 APPO 的 owner YAML 使用
  `conf/{ppo,appo}/task/<task>/<backend>.yaml`。
- MLX PPO 从 `conf/ppo/config_mlx.yaml` compose，并复用 PPO 任务 owner YAML 的
  布局。
- Off-policy owner YAML 多出算法这一维度：
  `conf/offpolicy/task/<algo>/<task>/<backend>.yaml`。
- 其他已有的 config 根目录，例如 `conf/ppo_him/` 与 `conf/hora_distill/`，对其
  所支持的任务遵循相同的 owner YAML 身份规则。

## 必需语义

- 使用对外的 CLI flag 切换 backend，例如
  `uv run train --algo ppo --task go2_joystick_flat --sim mujoco` 或
  `uv run train --algo ppo --task go2_joystick_flat --sim motrix`。
- 对于 off-policy 入口，保持 `--algo <algo>` 与内部 owner YAML 路径
  `conf/offpolicy/task/<algo>/<task>/<backend>.yaml` 对齐。
- `training.sim_backend` 是所选 owner YAML 内部的身份字段，而不是一个独立的
  backend 切换开关。
- 与 backend 相关的 reward、env、scene 与算法差异属于 owner YAML，而不是训练
  脚本。
- 当任务使用奖励时，reward config 必须由 owner YAML 显式注入。

## 仓库中的证据

- PPO owner 示例：`conf/ppo/task/go2_joystick_flat/mujoco.yaml`
- APPO config 根目录：`conf/appo/config.yaml`
- Off-policy config 根目录：`conf/offpolicy/config.yaml`
- Off-policy task/algo guard：`src/unilab/training/common.py`
- Config 测试：`tests/config/test_config_system.py`、
  `tests/scripts/test_train_script_configs.py`、
  `tests/envs/locomotion/g1/test_issue175_regression.py`

## Navigation

- Index: [文档](0-index.md)
