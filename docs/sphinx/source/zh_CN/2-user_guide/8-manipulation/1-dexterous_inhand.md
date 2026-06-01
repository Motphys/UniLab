# 灵巧手内操作

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

典型流程分两个阶段：先生成抓取缓存，然后训练旋转策略。

```bash
uv run train --algo ppo --task allegro_inhand_grasp --sim mujoco training.no_play=true
uv run train --algo ppo --task allegro_inhand --sim mujoco training.no_play=true
```

PPO Allegro 路径也存在 Motrix owner YAML：

```bash
uv run train --algo ppo --task allegro_inhand_grasp --sim motrix training.no_play=true
uv run train --algo ppo --task allegro_inhand --sim motrix training.no_play=true
```

旋转 owner 默认使用 `cache/allegro_grasp_50k.npy` 处的抓取缓存。要使用自定义缓存，override `env.grasp_cache_path`：

```bash
uv run train --algo ppo --task allegro_inhand --sim mujoco \
  env.grasp_cache_path=cache/my_allegro_grasp.npy
```

用 `eval` 回放已训练的 checkpoint（`--load-run -1` 取最新的 run）：

```bash
uv run eval --algo ppo --task allegro_inhand --sim mujoco --load-run -1
uv run eval --algo appo --task allegro_inhand --sim mujoco --load-run -1
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

完整 HORA 流程分三个阶段：

1. 生成 grasp cache。
2. 训练 teacher policy。
3. 需要时再蒸馏出 student policy。

完整 HORA teacher / student 流程以 MuJoCo owner 为主。Motrix 路径当前只承担 phase-1 PPO rotation 和 grasp cache 采集，不是完整 HORA 能力等价路径。

### Grasp cache 与 scale

默认 cache 托管在 Hugging Face (`unilabsim/unilab-caches`)，首次训练时自动下载到 `src/unilab/assets/caches/`，无需手动操作。

Sharpa rotation 从按 scale 区分的 grasp cache 中采样，因此多 scale cache 通过对每个 scale 分别运行 grasp 任务来采集（cache 文件命名为 `<prefix>_<scale>.npy`）：

```bash
uv run train --algo ppo --task sharpa_inhand_grasp --sim mujoco 'env.domain_rand.scale_list=[0.8]' training.no_play=true
uv run train --algo ppo --task sharpa_inhand_grasp --sim mujoco 'env.domain_rand.scale_list=[1.0]' training.no_play=true
uv run train --algo ppo --task sharpa_inhand_grasp --sim mujoco 'env.domain_rand.scale_list=[1.2]' training.no_play=true
```

或使用辅助脚本，按顺序采集每个 scale：

```bash
bash scripts/sharpa_collect_grasps.sh 0.8 1.0 1.2
```

Motrix 也可以采集 grasp cache（仅 phase-1 范围）：

```bash
uv run train --algo ppo --task sharpa_inhand_grasp --sim motrix \
  'env.domain_rand.scale_list=[1.0]' \
  env.grasp_collection_target=1000 \
  training.no_play=true
```

要使用自定义 cache 前缀，override `env.grasp_cache_path`：

```bash
uv run train --algo ppo --task sharpa_inhand --sim mujoco \
  env.grasp_cache_path=cache/my_sharpa_grasp_cache
```

### Teacher 与 student

用 `hora` profile 训练 HORA teacher（PPO 或 APPO）：

```bash
uv run train --algo ppo --task sharpa_inhand --sim mujoco --profile hora training.no_play=true
uv run train --algo appo --task sharpa_inhand --sim mujoco --profile hora training.no_play=true
```

用 `eval --profile hora --load-run -1` 回放 teacher run：

```bash
uv run eval --algo ppo --task sharpa_inhand --sim mujoco --profile hora --load-run -1
uv run eval --algo appo --task sharpa_inhand --sim mujoco --profile hora --load-run -1
```

Student 蒸馏由 `conf/hora_distill/task/sharpa_inhand/mujoco.yaml` 配置，由 `scripts/train_hora_distill.py` 实现；顶层 CLI 目前没有暴露单独的 HORA 蒸馏路由（不在 CLI `SUPPORTED_ALGOS` 中）。需要从 APPO teacher 蒸馏时，在该低层配置中设置 `teacher.algo_family=appo`。

常见日志目录：

- `logs/hora_ppo/SharpaInhandRotation/`
- `logs/hora_appo/SharpaInhandRotation/`
- `logs/hora_distill/SharpaInhandRotation/`

这里的 scale / grasp cache / DR 边界比较敏感；生命周期规则参见 {doc}`../5-domain_randomization/0-index`。

关于分类级别的任务页面，参见 {doc}`../4-tasks/3-manipulation`。
