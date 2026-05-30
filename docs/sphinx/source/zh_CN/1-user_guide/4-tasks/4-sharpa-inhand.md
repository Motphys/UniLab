# Sharpa Inhand

语言: 简体中文

## 任务

- rotation：`sharpa_inhand`
- grasp cache：`sharpa_inhand_grasp`
- MuJoCo HORA teacher：`sharpa_inhand/mujoco_hora`

## 典型流程

1. 生成 grasp cache
2. 训练 teacher policy
3. 需要时再训练 student policy

## 配置入口

- PPO grasp / rotation：`conf/ppo/task/sharpa_inhand_grasp/`、`conf/ppo/task/sharpa_inhand/`
- PPO MuJoCo HORA teacher：`conf/ppo/task/sharpa_inhand/mujoco_hora.yaml`
- PPO Motrix phase-1：`conf/ppo/task/sharpa_inhand/motrix.yaml`
- APPO teacher：`conf/appo/task/sharpa_inhand/mujoco_hora.yaml`
- student distill：`conf/hora_distill/task/sharpa_inhand/mujoco.yaml`

完整 HORA teacher / student 流程仍以 MuJoCo owner 为主；Motrix 当前承担的是 phase-1 PPO rotation 和 grasp cache 采集，不是完整能力等价路径。

## Grasp cache 与 scale

MuJoCo 采集：

```bash
uv run train --algo ppo --task sharpa_inhand_grasp --sim mujoco 'env.domain_rand.scale_list=[0.8]' training.no_play=true
uv run train --algo ppo --task sharpa_inhand_grasp --sim mujoco 'env.domain_rand.scale_list=[1.0]' training.no_play=true
uv run train --algo ppo --task sharpa_inhand_grasp --sim mujoco 'env.domain_rand.scale_list=[1.2]' training.no_play=true
```

Motrix 采集：

```bash
uv run train --algo ppo --task sharpa_inhand_grasp --sim motrix \
  'env.domain_rand.scale_list=[1.0]' \
  env.grasp_collection_target=1000 \
  training.no_play=true
```

默认 rotation 读取 `cache/sharpa_grasp_linspace`；Motrix phase-1 读取按 scale 切分的 `<prefix>_<scale>.npy`。
批量采集时可直接用 `bash scripts/sharpa_collect_grasps.sh 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6`。

自定义 cache：

```bash
uv run train --algo ppo --task sharpa_inhand --sim mujoco \
  env.grasp_cache_path=cache/my_sharpa_grasp_cache

uv run train --algo ppo --task sharpa_inhand --sim motrix \
  env.grasp_cache_path=cache/my_sharpa_grasp_cache \
  training.no_play=true
```

Motrix 自定义 cache 前缀需要满足 `<prefix>_<scale>.npy` 命名规则，例如 `cache/my_sharpa_grasp_cache_1.0.npy`。

## Teacher / student

```bash
uv run train --algo ppo --task sharpa_inhand --sim mujoco --profile hora
uv run train --algo appo --task sharpa_inhand --sim mujoco --profile hora
```

HORA student distill 由 `scripts/train_hora_distill.py` 和
`conf/hora_distill/task/sharpa_inhand/mujoco.yaml` 实现；当前没有独立的顶层
`--algo` 路由。需要从 APPO teacher 蒸馏时，使用该低层入口的
`teacher.algo_family=appo` 配置；HORA distill 不在顶层 CLI 的
`SUPPORTED_ALGOS` 中。

回放：

```bash
uv run eval --algo ppo --task sharpa_inhand --sim mujoco --profile hora --load-run -1
uv run eval --algo appo --task sharpa_inhand --sim mujoco --profile hora --load-run -1
```

固定 teacher run：

```bash
uv run eval --algo ppo --task sharpa_inhand --sim mujoco --profile hora \
  --load-run "2026-04-28_12-00-00_mujoco"
```

Student distill 的固定 teacher run 仍属于 `scripts/train_hora_distill.py`
低层配置范围，统一 CLI 暂未暴露。

常见日志目录：

- `logs/hora_ppo/SharpaInhandRotation/`
- `logs/hora_appo/SharpaInhandRotation/`
- `logs/hora_distill/SharpaInhandRotation/`

## 边界

- 完整 HORA 流程仍以 MuJoCo owner 为主
- Motrix 路径不是完整 HORA 能力等价路径
- Sharpa 的 scale / grasp cache / DR 边界需要结合 [05 域随机化](../5-domain-randomization.md)

## Navigation

- Index: [Documentation](../../0-index.md)
- Previous: [Allegro Inhand](3-allegro-inhand.md)
- Next: [Go2 Rough Terrain](5-go2-rough-terrain.md)
