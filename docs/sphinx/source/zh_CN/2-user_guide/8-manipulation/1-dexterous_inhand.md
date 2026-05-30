# 灵巧手内操作

语言: 简体中文

本页介绍已提交的 Allegro 和 Sharpa 手内操作路径。通过 `--task` 和 `--sim` 选择后端；不要单独覆盖 `training.sim_backend`。owner YAML 始终是哪些组合被配置的内部证据。

## Allegro

Allegro 旋转使用已注册的 env `AllegroInhandRotation`。旋转 owner 是 `allegro_inhand`，抓取缓存生成使用 `allegro_inhand_grasp`。

Owner 证据：

- `conf/ppo/task/allegro_inhand/mujoco.yaml`
- `conf/ppo/task/allegro_inhand/motrix.yaml`
- `conf/ppo/task/allegro_inhand_grasp/mujoco.yaml`
- `conf/ppo/task/allegro_inhand_grasp/motrix.yaml`
- `conf/appo/task/allegro_inhand/mujoco.yaml`
- `conf/appo/task/allegro_inhand/motrix.yaml`

先生成抓取缓存，然后训练旋转：

```bash
uv run train --algo ppo --task allegro_inhand_grasp --sim mujoco training.no_play=true
uv run train --algo ppo --task allegro_inhand --sim mujoco training.no_play=true
```

PPO Allegro 路径也存在 Motrix owner YAML：

```bash
uv run train --algo ppo --task allegro_inhand_grasp --sim motrix training.no_play=true
uv run train --algo ppo --task allegro_inhand --sim motrix training.no_play=true
```

## Sharpa

Sharpa 旋转使用已注册的 env `SharpaInhandRotation`。当前已提交的训练路径是 MuJoCo owner 路径。

Owner 证据：

- `conf/ppo/task/sharpa_inhand/mujoco.yaml`
- `conf/ppo/task/sharpa_inhand/mujoco_hora.yaml`
- `conf/ppo/task/sharpa_inhand_grasp/mujoco.yaml`
- `conf/appo/task/sharpa_inhand/mujoco.yaml`
- `conf/appo/task/sharpa_inhand/mujoco_hora.yaml`
- `conf/hora_distill/task/sharpa_inhand/mujoco.yaml`

按 scale 生成缓存，训练一个 teacher，然后蒸馏出一个 student：

```bash
uv run train --algo ppo --task sharpa_inhand_grasp --sim mujoco \
  'env.domain_rand.scale_list=[0.5]' \
  training.no_play=true

uv run train --algo ppo --task sharpa_inhand --sim mujoco --profile hora training.no_play=true
```

Student 蒸馏由 `conf/hora_distill/task/sharpa_inhand/mujoco.yaml` 配置；顶层 CLI 目前没有暴露单独的 HORA 蒸馏路由。

关于分类级别的任务页面，参见 {doc}`../4-tasks/3-manipulation`。

## Navigation

- Index: [文档](0-index.md)
