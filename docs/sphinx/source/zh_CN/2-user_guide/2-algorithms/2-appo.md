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
sequenceDiagram
    participant C as Collector
    participant R as Ring Buffer
    participant L as Learner
    participant G as GPU
    loop 采集一条 rollout（steps_per_env 步）
        Note over C: mlp_infer_ms — 单步策略推理选动作
        Note over C: env_step_total_ms — 单次 env.step() 耗时
    end
    C->>R: 写入 rollout（Sync Collect）
    Note over L: Collector Wait — 阻塞等 ring buffer 产出新 rollout
    R->>L: 读取可用 rollout（Rollouts Read / Available On Arrive）
    L->>G: 暂存到 staging pool（Staging Pool 滑动窗口）
    Note over L,G: H2D Copy — host→device 批次拷贝
    L->>G: V-trace 校正 + PPO 更新（Appo/Updates Executed）
    Note over L,G: Train — 纯 SGD 计算
    L->>C: 写共享内存新权重
    Note over L: Weight Sync — 发布权重给 collector
    Note over L: Iter Wall — 该 learner 迭代整圈墙钟（含以上各项）
```

## 关键字段

- `algo.steps_per_env`：单个环境的 rollout 长度。
- `training.replay_queue_size`：learner 侧缓存深度。
- `training.collector_device`：collector 设备；默认跟随 learner。
- `algo.save_interval`：checkpoint 保存间隔。

默认日志根目录为 `logs/appo/<task>/`，来自 `conf/appo/config.yaml` 中的
`algo.algo_log_name=appo`。
