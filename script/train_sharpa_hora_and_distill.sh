#!/usr/bin/env bash

set -euo pipefail

# Run Sharpa MuJoCo HORA teacher training first so the following distillation
# step can resolve the latest teacher checkpoint from the standard log layout.
echo "[train_sharpa_hora_and_distill] start teacher training"
uv run --active --no-sync \
  scripts/train_rsl_rl.py \
  task=sharpa_inhand/mujoco_hora

# Run HORA distillation only after teacher training succeeds; keeping the steps
# sequential avoids a missing-checkpoint failure in train_hora_distill.py.
echo "[train_sharpa_hora_and_distill] start distillation"
uv run --active --no-sync \
  scripts/train_hora_distill.py \
  task=sharpa_inhand/mujoco \
  teacher.algo_family=ppo
