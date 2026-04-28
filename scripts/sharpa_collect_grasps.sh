#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <scale1> [scale2 ...]"
  exit 1
fi

for scale in "$@"; do
  echo "[sharpa_collect_grasps] collecting scale=${scale}"

  uv run scripts/train_rsl_rl.py \
    task=sharpa_inhand_grasp/mujoco \
    "env.domain_rand.scale_list=[${scale}]"
done
