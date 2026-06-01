# PPO

PPO 是默认的同步 on-policy 训练路径。它使用 `scripts/train_rsl_rl.py`，从
`conf/ppo/config.yaml` 组合配置，并运行 `src/unilab/algos/torch/rsl_rl_ppo.py`
和 `src/unilab/training/rsl_rl.py` 中的 RSL-RL 适配代码。

## 快速开始

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco
uv run train --algo ppo --task go2_joystick_flat --sim motrix training.no_play=true
```

## 常用 Override

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco \
  algo.num_envs=2048 \
  algo.max_iterations=300 \
  training.no_play=true
```

使用 `uv run eval` 进行检查点回放：

```bash
uv run eval --algo ppo --task go2_joystick_flat --sim mujoco --load-run -1
```

日志按 `algo.algo_log_name` 分组；`conf/ppo/config.yaml` 中的默认值为
`rsl_rl_ppo`。
