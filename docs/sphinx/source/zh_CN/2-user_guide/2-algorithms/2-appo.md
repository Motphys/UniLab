# APPO

APPO 是 UniLab 的异步 PPO 路径。它使用 `scripts/train_appo.py`、
`conf/appo/config.yaml` 以及 `src/unilab/algos/torch/appo/` 下的运行时。该配置暴露
了 `algo.steps_per_env`、`training.collector_device` 和
`training.replay_queue_size`；算法配置中包含 V-trace 裁剪字段。

## 快速开始

```bash
uv run train --algo appo --task go2_joystick_flat --sim mujoco
uv run train --algo appo --task g1_motion_tracking --sim motrix training.no_play=true
```

## 常用 Override

```bash
uv run train --algo appo --task go2_joystick_flat --sim mujoco \
  algo.num_envs=2048 \
  algo.max_iterations=300 \
  training.replay_queue_size=2
```

回放与检查点选择使用 `uv run eval`：

```bash
uv run eval --algo appo --task go2_joystick_flat --sim mujoco --load-run -1
```

## 运行模型

- collector 负责 CPU 仿真，learner 负责 GPU 训练。
- rollout 会先进入 replay queue，再由 learner 消费。
- APPO 内部带 V-trace importance-sampling 修正，更新语义不同于同步 PPO。
- collector / learner 流水线由一个 4 槽 ring buffer 支撑。

单次迭代的计时时序（各指标含义见[日志页](../1-training/3-logging.md)）：

```{mermaid}
gantt
    title 一次 Learner 迭代的时间线（APPO）
    dateFormat x
    axisFormat %S

    section Collector（进程）
    rollout N · env interaction（mlp_infer + env_step）×steps_per_env :active, c0, 0, 12000
    rollout N+1（与 learner 并行采集）                                :active, c1, 13000, 30000

    section Ring Buffer（4 槽）
    rollout N 就绪    :milestone, r0, 12000, 12000
    rollout N+1 就绪  :milestone, r1, 30000, 30000

    section Learner（GPU）
    Collector Wait（缓冲满则约 0）    :done,   l0, 12000, 13000
    H2D Copy（ring 进 staging）       :        l1, 13000, 16000
    Train（V-trace + PPO SGD）        :active, l2, 16000, 28000
    Weight Sync 写回 collector        :crit,   l3, 28000, 30000

    section Iter Wall
    perf/iter_ms（仅 learner 这圈）   :        l4, 12000, 30000
```

> 横轴为示意相对时长（非真实 ms 比例）。collector 子进程经 4 槽 ring buffer 与 learner 并行产出 rollout，稳态下 **Collector Wait ≈ 0**。`perf/iter_ms` 仅计 learner 这一圈（含 Collector Wait，但不含 collector 的并行采集计算）；红色 Weight Sync 标志该轮迭代结束、向 collector 发布新权重。各字段含义见[日志页](../1-training/3-logging.md)。

## 关键字段

- `algo.steps_per_env`：单个环境的 rollout 长度。
- `training.replay_queue_size`：learner 侧缓存深度。
- `training.collector_device`：collector 设备；默认跟随 learner。
- `algo.save_interval`：checkpoint 保存间隔。

默认日志根目录为 `logs/appo/<task>/`，来自 `conf/appo/config.yaml` 中的
`algo.algo_log_name=appo`。
