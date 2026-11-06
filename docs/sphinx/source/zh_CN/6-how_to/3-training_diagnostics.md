# 训练诊断

本页汇总 NaN/Inf 失败、reward 异常、训练不收敛、后端行为不一致与性能问题的
排查路径。文中所有键与命令都能在仓库中核实；需要安装仿真器、GPU 或显示器的
步骤均已标注。

## NaN 诊断链

NaN guard 在设备侧检查观测、reward 与控制量中的非有限行，并在显式的主机
边界写出一份状态 dump。链路如下：

1. 在 `training.nan_guard` 下配置——以下为 `src/unilab/conf/ppo/config.yaml`
   中 PPO 的默认值（SAC/FlashSAC owner 带有相同的块）：

   ```yaml
   training:
     nan_guard:
       enabled: true
       buffer_size: 100
       max_envs_to_dump: 5
       output_dir: null        # null -> /tmp/unilab/nan_dumps
       defer_validation: true
       check_interval: 32
       capture_interval: 32
   ```

   字段由 `src/unilab/training/tensor_diagnostics.py` 的 `NanGuardCfg` 定义。
   `capture_interval` 控制物理状态快照的保留频率（0 表示不快照）；
   `check_interval` 是 deferred 模式下的轮询周期。
2. 训练入口把 guard 挂到 env 上（`src/unilab/training/run.py` 的
   `apply_env_nan_guard`），`TorchEnv.step` 每步调用 `NanGuard.check(...)`。
   检出时 `NanGuard.dump(...)` 写出 `nan_dump_<时间戳>_step<N>.npz`
   （状态环形缓冲加 `meta_*` 字段，如 `detection_step` 与 `nan_env_ids`），
   把模型文件复制到旁边，并更新 `nan_dump_latest.npz` 软链。
3. 在 MuJoCo viewer 中回放 dump（需要 `mujoco` extra 与显示器）：

   ```bash
   uv run unilab-viz-nan /tmp/unilab/nan_dumps/nan_dump_latest.npz --env-index 0
   ```

   `unilab-viz-nan` 是 `src/unilab/utils/nan_viz.py` 注册的 console 入口
   （`pyproject.toml`）；它先打印检出步数与 NaN env id，再回放缓冲状态。
   不支持物理状态回放的后端 dump 中没有状态——viewer 只打印元数据。

细节与 viewer 操作见 {doc}`../2-user_guide/7-tooling/3-nan_visualizer`。往返加载覆盖：
`tests/training/test_tensor_diagnostics.py`。

## 读懂 `info["log"]`

Manager-Based 环境把结构化指标放进 `info["log"]`；训练 logger 把带斜杠前缀的
条目转发到 TensorBoard / W&B。完整语义表见 {doc}`../2-user_guide/1-training/3-logging`；键族与生产者：

| 键 | 生产者 | 含义 |
| --- | --- | --- |
| `reward/<term>` | `RewardManager.step_reward_extras()`（`src/unilab/managers/reward_manager.py`） | 各 term 的 `raw_value * weight` 跨环境均值，未乘 `dt` |
| `Episode_Termination/<term>` | `TerminationManager.reset()`（`src/unilab/managers/termination_manager.py`） | 因该终止 term 而 reset 的环境行数 |
| `Episode_Metrics/<key>` | `MetricsManager`（`src/unilab/managers/metrics_manager.py`） | 已完成 episode 的指标聚合 |

整体进展看 `Train/mean_reward` 与 `Train/mean_episode_length`，归因看
`reward/<term>`；两者区别见 {doc}`../2-user_guide/1-training/3-logging`。

## 常见失败模式

### Reward 爆炸或出现非有限值

- 症状：训练以 `RewardManager term '<name>' returned NaN/Inf for
  environments ...` 中止（`src/unilab/managers/reward_manager.py`），或 NaN
  guard 写出 dump。
- 查：逐项的 `reward/<term>` 曲线——发散的 term 通常最先分离。对照 owner
  YAML `reward:` 块中它的原始量级与 `weight`。
- 改：用 CLI 调权，例如 `uv run train --algo ppo --task <task> --sim mujoco
  reward.<term>.weight=-0.01`（override 机制见
  {doc}`../2-user_guide/2-algorithms/1-ppo` 的 “Reward term weights” 小节）。保留
  平滑/正则项（`action_rate` 类惩罚）；删掉它们常常直接导致动作流不稳定。

### 策略不收敛

- 症状：`Train/mean_reward` 持平但迭代在推进。
- 对照算法参考表查超参（PPO 见 {doc}`../2-user_guide/2-algorithms/1-ppo`）：
  `algo.algorithm.learning_rate`、`algo.algorithm.entropy_coef`、
  `algo.algorithm.lam`。检查 `algo.num_envs`——它是按 rank 计的数量，过小会
  让梯度估计失真。检查观测条件：`algo.actor.obs_normalization`（Go2 owner
  开启了它），以及是否有观测 term 输出了异常量纲的值。
- 检查 reward 配比：如果跟踪项（`reward/tracking_*`）一直接近零而惩罚项主导
  `reward/<term>` 曲线，问题是权重失衡而不是优化器。

### 两个后端行为不一致

- 症状：在后端 A 训练的 checkpoint 在后端 B 上表现变差或拒绝加载。
- 查：sim2sim 契约。训练把决定策略的字段快照进 `run_config.json`；play 入口
  在构造 env 之前用该快照校验目标配置（`src/unilab/utils/sim2sim.py` 的
  `resolve_sim2sim_config`）。DENYLIST 字段（observations、actions、网络
  形状、`ctrl_dt` 等）必须一致——默认 `training.sim2sim_strict: true` 下不一致
  会抛 `CrossBackendIncompatibleError`；WARNING_LIST 字段只打印警告。审计已发布
  的组合用 `uv run scripts/audit_sim2sim_contracts.py`。
- 查后端能力差异：{doc}`../3-deployment/2-sim_to_sim/6-capability_gaps`；
  精确支持等级：{doc}`../5-reference/5-support_matrix`。

### 训练中途出现 NaN

- 走上面的 NaN 链：dump 中的 `meta_detection_step` 与 `meta_nan_env_ids`
  缩小时间窗；回放状态可以看出是姿态发散（资产/keyframe 问题）还是控制量爆炸
  （reward/动作尺度问题）。
- 对照检出步之前的 `reward/<term>` 曲线，找最先发散的 term。

### 性能与 trace 分析

- Off-policy 配置（SAC/FlashSAC）暴露 `training.trace_enabled`、
  `training.trace_output_dir`、`training.trace_thread_time` 与
  `training.trace_cuda_events`（`src/unilab/conf/sac/config.yaml`），另有
  CUDA 上 NVTX 标注用的 `training.nvtx_profile_ranges`。trace 是 Perfetto
  JSON；标量计时键见 {doc}`../2-user_guide/1-training/3-logging`。
- 在主机侧分析已捕获的 trace（不需要 GPU）：

  ```bash
  uv run scripts/analyze_offpolicy_trace.py <trace.json> --training-e2e \
    --event collector/env_step --gap-event learner/training_e2e
  ```

  脚本按事件名报告 mean/median/p90/p95/p99 时长；`--steps-per-cycle` 与
  `--drop-first` 调整吞吐量汇总。
- 生产环境长 soak 的设置与失败分类字段见
  {doc}`../2-user_guide/1-training/7-tensor_runtime_production`。
