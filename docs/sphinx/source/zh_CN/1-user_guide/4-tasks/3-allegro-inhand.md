# Allegro Inhand

语言: 简体中文

## 任务

- rotation：`allegro_inhand`
- grasp cache：`allegro_inhand_grasp`

## 典型流程

1. 先生成 grasp cache
2. 再训练 rotation policy

## 配置入口

- PPO：`conf/ppo/task/allegro_inhand/`、`conf/ppo/task/allegro_inhand_grasp/`
- APPO：`conf/appo/task/allegro_inhand/`
- 默认 grasp cache：`cache/allegro_grasp_50k.npy`

## 常用命令

```bash
uv run train --algo ppo --task allegro_inhand_grasp --sim mujoco training.no_play=true
uv run train --algo ppo --task allegro_inhand --sim mujoco
uv run train --algo appo --task allegro_inhand --sim mujoco
```

Motrix owner 也存在：

```bash
uv run train --algo ppo --task allegro_inhand --sim motrix
uv run train --algo appo --task allegro_inhand --sim motrix
```

## 回放和自定义 cache

```bash
uv run eval --algo ppo --task allegro_inhand --sim mujoco --load-run -1
uv run eval --algo appo --task allegro_inhand --sim mujoco --load-run -1
```

```bash
uv run train --algo ppo --task allegro_inhand --sim mujoco \
  env.grasp_cache_path=cache/my_allegro_grasp.npy
```

## 边界

- backend 选择通过统一 CLI 的 `--sim`
- 不要单独 override `training.sim_backend`

## Navigation

- Index: [Documentation](../../0-index.md)
- Previous: [G1 Motion Tracking](2-g1-motion-tracking.md)
- Next: [Sharpa Inhand](4-sharpa-inhand.md)
