# HORA

The committed HORA path is the Sharpa in-hand teacher/student flow. Teacher
owners live under the PPO and APPO task trees through the `7-hora` profile for
`sharpa_inhand`; student distillation uses `scripts/train_hora_distill.py` and
`conf/hora_distill/task/sharpa_inhand/mujoco.yaml`.

## Teacher

```bash
uv run train --algo ppo --task sharpa_inhand --sim mujoco --profile hora
uv run train --algo appo --task sharpa_inhand --sim mujoco --profile hora training.no_play=true
```

The HORA PPO owner sets `algo.algo_log_name=hora_ppo` and resolves the runtime
through `unilab.algos.torch.hora.rsl_rl:resolve_hora_ppo_runtime`. The APPO
variant sets `algo.algo_log_name=hora_appo`.

## Student Distillation

Student distillation is implemented by `scripts/train_hora_distill.py` and
configured by `conf/hora_distill/task/sharpa_inhand/mujoco.yaml`. The top-level
CLI does not currently declare a separate HORA distillation `--algo` route, so
the public CLI examples on this page stay on the teacher path above.

Teacher checkpoint resolution is implemented in
`src/unilab/algos/torch/hora/distill_config.py`. The student log family is
`hora_distill`.
