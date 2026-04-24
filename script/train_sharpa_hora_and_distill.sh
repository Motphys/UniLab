#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

TEACHER_TASK_NAME="SharpaInhandRotation"
TEACHER_LOG_SESSION="$(date -u +%Y-%m-%d_%H-%M-%S)_mujoco_teacher_$$"
TEACHER_LOG_ROOT="${SHARPA_HORA_TEACHER_LOG_ROOT:-${REPO_ROOT}/logs/hora_ppo_isolated/${TEACHER_LOG_SESSION}}"
TEACHER_TASK_LOG_ROOT="${TEACHER_LOG_ROOT}/${TEACHER_TASK_NAME}"

# Keep the teacher run under an isolated log root so later distillation never
# depends on whichever shared hora_ppo run happens to be "latest".
echo "[train_sharpa_hora_and_distill] start teacher training"
echo "[train_sharpa_hora_and_distill] isolated_teacher_log_root=${TEACHER_LOG_ROOT}"
uv run scripts/train_rsl_rl.py \
  task=sharpa_inhand/mujoco_hora \
  reward.scales.rotate=1.25 \
  reward.scales.torque=-0.5 \
  reward.scales.work=-2.5 \
  "training.log_root=${TEACHER_LOG_ROOT}"

if [[ ! -d "${TEACHER_TASK_LOG_ROOT}" ]]; then
  echo "[train_sharpa_hora_and_distill] teacher task log root not found: ${TEACHER_TASK_LOG_ROOT}" >&2
  exit 1
fi

mapfile -t teacher_runs < <(find "${TEACHER_TASK_LOG_ROOT}" -mindepth 1 -maxdepth 1 -type d | sort)
if [[ "${#teacher_runs[@]}" -eq 0 ]]; then
  echo "[train_sharpa_hora_and_distill] no teacher run directories found under ${TEACHER_TASK_LOG_ROOT}" >&2
  exit 1
fi
TEACHER_RUN_DIR="${teacher_runs[${#teacher_runs[@]} - 1]}"

mapfile -t teacher_checkpoints < <(
  find "${TEACHER_RUN_DIR}" -mindepth 1 -maxdepth 1 -type f -name 'model_*.pt' | sort -V
)
if [[ "${#teacher_checkpoints[@]}" -eq 0 ]]; then
  echo "[train_sharpa_hora_and_distill] no teacher checkpoints found under ${TEACHER_RUN_DIR}" >&2
  exit 1
fi
TEACHER_CHECKPOINT="${teacher_checkpoints[${#teacher_checkpoints[@]} - 1]}"

# Run HORA distillation only after teacher training succeeds, and pass the
# concrete teacher checkpoint path so concurrent trainings cannot change it.
echo "[train_sharpa_hora_and_distill] resolved_teacher_run=${TEACHER_RUN_DIR}"
echo "[train_sharpa_hora_and_distill] resolved_teacher_checkpoint=${TEACHER_CHECKPOINT}"
echo "[train_sharpa_hora_and_distill] start distillation"
uv run scripts/train_hora_distill.py \
  task=sharpa_inhand/mujoco \
  teacher.algo_family=ppo \
  "algo.load_run=${TEACHER_CHECKPOINT}"
