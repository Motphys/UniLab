#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <scale1> [scale2 ...]"
  echo "Environment:"
  echo "  SHARPA_GRASP_BACKEND=mujoco|motrix   default: mujoco"
  echo "  SHARPA_GRASP_CACHE_PATH=<prefix>     optional env.grasp_cache_path override"
  echo "  SHARPA_GRASP_TARGET=<count>          optional env.grasp_collection_target override"
  echo "  SHARPA_GRASP_NUM_ENVS=<count>        optional algo.num_envs override"
  exit 1
fi

backend="${SHARPA_GRASP_BACKEND:-mujoco}"
case "${backend}" in
  mujoco|motrix) ;;
  *)
    echo "Unsupported SHARPA_GRASP_BACKEND=${backend}; expected mujoco or motrix"
    exit 1
    ;;
esac

extra_args=(training.no_play=true)
if [ -n "${SHARPA_GRASP_CACHE_PATH:-}" ]; then
  extra_args+=("env.grasp_cache_path=${SHARPA_GRASP_CACHE_PATH}")
fi
if [ -n "${SHARPA_GRASP_TARGET:-}" ]; then
  extra_args+=("env.grasp_collection_target=${SHARPA_GRASP_TARGET}")
fi
if [ -n "${SHARPA_GRASP_NUM_ENVS:-}" ]; then
  extra_args+=("algo.num_envs=${SHARPA_GRASP_NUM_ENVS}")
fi

for scale in "$@"; do
  echo "[sharpa_collect_grasps] collecting backend=${backend} scale=${scale}"

  uv run scripts/train_rsl_rl.py \
    "task=sharpa_inhand_grasp/${backend}" \
    "env.domain_rand.scale_list=[${scale}]" \
    "${extra_args[@]}"
done
