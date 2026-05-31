# HORA

已提交的 HORA 路径是 Sharpa 手内（in-hand）teacher/student 流程。teacher owner 位
于 PPO 与 APPO 的 task 树下，通过 `sharpa_inhand` 的 `7-hora` profile 选择；student
蒸馏使用 `scripts/train_hora_distill.py` 和
`conf/hora_distill/task/sharpa_inhand/mujoco.yaml`。

## Teacher

```bash
uv run train --algo ppo --task sharpa_inhand --sim mujoco --profile hora
uv run train --algo appo --task sharpa_inhand --sim mujoco --profile hora training.no_play=true
```

HORA PPO owner 设置 `algo.algo_log_name=hora_ppo`，并通过
`unilab.algos.torch.hora.rsl_rl:resolve_hora_ppo_runtime` 解析运行时。APPO 变体设置
`algo.algo_log_name=hora_appo`。

## Student 蒸馏

student 蒸馏由 `scripts/train_hora_distill.py` 实现，并由
`conf/hora_distill/task/sharpa_inhand/mujoco.yaml` 配置。顶层 CLI 目前没有声明独立的
HORA 蒸馏 `--algo` 路由，因此本页的公开 CLI 示例仍保持在上面的 teacher 路径上。

teacher 检查点的解析在 `src/unilab/algos/torch/hora/distill_config.py` 中实现。
student 日志族为 `hora_distill`。
