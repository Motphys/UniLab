# Agent 速查

语言: 简体中文

本页面向需要以最短路径获取当前仓库事实的维护者与 agent。

## 从这里开始

- 安装与 smoke 检查：{doc}`../1-getting_started/2-installation`
- 后端选择：{doc}`../2-user_guide/3-backends/3-choosing_a_backend`
- PPO 入口：`scripts/train_rsl_rl.py`
- MLX PPO 入口：`scripts/train_mlx_ppo.py`
- APPO 入口：`scripts/train_appo.py`
- SAC / TD3 / FlashSAC 入口：`scripts/train_offpolicy.py`
- HIM-PPO 入口：`scripts/train_him_ppo.py`
- HORA 蒸馏入口：`scripts/train_hora_distill.py`

## 需要记住的契约

- Env 契约：`src/unilab/base/np_env.py`
- Backend 契约：`src/unilab/base/backend/base.py`
- 训练辅助工具：`src/unilab/training/run.py`
- Config schema：`src/unilab/structured_configs.py`
- Developer 标准：{doc}`1-architecture/1-overview`

命令示例请使用 `uv run train`、`uv run eval` 或 `uv run demo`。
通过 `--algo`、`--task` 与 `--sim` 选择算法、任务与后端；
只写能够追溯到代码、config、测试或当前文档的事实。

## Navigation

- Index: [文档](0-index.md)
