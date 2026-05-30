# HIM-PPO

语言: 简体中文

HIM-PPO 有自己的配置组和脚本。入口是 `scripts/train_him_ppo.py`，基础配置是
`conf/ppo_him/config.yaml`，已提交的 task owner 是
`conf/ppo_him/task/go2_arm_manip_loco/mujoco.yaml`。

## 当前入口

`src/unilab/cli.py` 目前通过顶层的 `uv run train` CLI 暴露了 `1-ppo`、`8-mlx_ppo`、
`2-appo`、`3-sac`、`4-td3` 和 `flashsac`。HIM-PPO 由 `scripts/train_him_ppo.py`
实现，但它尚未拥有顶层的 `--algo` 路由。

## Owner 细节

Go2 机械臂 owner 从基础配置中填充所需的历史维度：

- `algo.num_one_step_obs=76`
- `algo.num_actor_history=5`
- `algo.num_critic_history=1`
- `training.task_name=Go2ArmManipLoco`

一旦有可用的检查点，回放将使用相同的 HIM-PPO 实现入口。请将面向用户的 PPO 示例保
持在受支持的顶层 CLI 形式上；仅在调试该专用技术栈时才使用 HIM-PPO 脚本路径。

HIM-PPO 不是默认的 PPO 路径；请将其用于明确选择 HIM-PPO 配置组的 Go2 机械臂
manip-loco owner。

## Navigation

- Index: [文档](0-index.md)
