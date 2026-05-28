# HORA

The committed HORA path is the Sharpa in-hand teacher/student flow. Teacher
owners live under the PPO and APPO task trees as `sharpa_inhand/mujoco_hora`;
student distillation uses `scripts/train_hora_distill.py` and
`conf/hora_distill/task/sharpa_inhand/mujoco.yaml`.

## Teacher

```bash
uv run scripts/train_rsl_rl.py task=sharpa_inhand/mujoco_hora
uv run scripts/train_appo.py task=sharpa_inhand/mujoco_hora training.no_play=true
```

The HORA PPO owner sets `algo.algo_log_name=hora_ppo` and resolves the runtime
through `unilab.algos.torch.hora.rsl_rl:resolve_hora_ppo_runtime`. The APPO
variant sets `algo.algo_log_name=hora_appo`.

## Student Distillation

```bash
uv run scripts/train_hora_distill.py task=sharpa_inhand/mujoco
uv run scripts/train_hora_distill.py task=sharpa_inhand/mujoco \
  teacher.algo_family=appo \
  teacher.task=sharpa_inhand/mujoco_hora
```

Teacher checkpoint resolution is implemented in
`src/unilab/algos/torch/hora/distill_config.py`. The student log family is
`hora_distill`.
