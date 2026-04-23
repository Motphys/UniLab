#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <scale1> [scale2 ...]"
  exit 1
fi

for scale in "$@"; do
  echo "[sharpa_collect_grasps] collecting scale=${scale}"
  # Use the caller's active UniLab environment and skip lockfile syncing here,
  # because grasp collection is a long-running job and the current uv.lock cannot
  # be parsed in this workspace.
  uv run --active --no-sync \
    scripts/train_rsl_rl.py \
    task=sharpa_inhand_grasp/mujoco \
    "env.scale_list=[${scale}]"
done
