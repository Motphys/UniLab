# MicroDuck PPO 对比报告（roadmap #1452 / issue #1457）

生成：`uv run scripts/microduck_alignment_compare.py`（数据见下表 Runs）；UniLab 侧为 child 2/3/4 合入后的对齐配方（audit GAP 0），上游为 microduck_rl @ 29e887ec 默认 BAM 配置。按 maintainer 决策，上游 seed43 未执行，上游统计为 n=1。

终段窗口 = 每 run 末尾 20% iterations；收敛速度 = mean_reward 首次达到自身终段均值 80% 的 iteration。

## Runs

**UniLab**
| run | seed | max_iterations |
| --- | --- | --- |
| `logs/rsl_rl_ppo/MicroduckVelocityFlat/2026-09-03_18-54-37_mjwarp` | 42 | 2000 |
| `logs/rsl_rl_ppo/MicroduckVelocityFlat/2026-09-03_19-20-43_mjwarp` | 43 | 2000 |

**Upstream (microduck_rl)**
| run | seed | max_iterations |
| --- | --- | --- |
| `/home/user/ws/simulator/microduck_rl/logs/rsl_rl/velocity/2026-09-03_19-48-28_seed42` | 42 | 2000 |

> ⚠ upstream 侧只有 1 个 run，std 不可用，统计仅供参考。

## 总览指标

| 指标 | UniLab | 上游 | 相对差 |
| --- | --- | --- | --- |
| mean_reward 终段均值 | 83.751 ± 5.422 | 113.833 (n=1) | -26.4% |
| episode length 终段均值 | 976.211 ± 13.477 | 944.644 (n=1) | +3.3% |
| 收敛 iteration (80% final reward) | 784.500 ± 132.229 | 128.000 (n=1) | +512.9% |

## Reward term 终段对比

| term (UniLab / 上游) | UniLab 终段 | 上游终段 | 相对差 |
| --- | --- | --- | --- |
| action_rate / action_rate_l2 | -0.801 ± 0.127 | -1.039 (n=1) | +22.9% |
| air_time | 1.391 ± 0.129 | 1.024 (n=1) | +35.8% |
| angular_momentum | -0.000 ± 0.000 | -0.000 (n=1) | -463.0% |
| body_ang_vel | -0.349 ± 0.073 | -0.031 (n=1) | -1030.8% |
| body_pose_tracking | 0.000 ± 0.000 | 0.000 (n=1) | n/a |
| dof_pos_limits | -0.025 ± 0.017 | -0.001 (n=1) | -2494.5% |
| foot_clearance | -0.022 ± 0.001 | -0.004 (n=1) | -401.2% |
| foot_slip | -0.001 ± 0.000 | -0.000 (n=1) | -490.6% |
| foot_swing_height | -0.077 ± 0.007 | -0.003 (n=1) | -2203.4% |
| head_pose_bias | -0.219 ± 0.000 | -0.224 (n=1) | +2.3% |
| head_pose_tracking | 1.702 ± 0.047 | 1.705 (n=1) | -0.1% |
| leg_pose / pose | 0.062 ± 0.087 | 0.608 (n=1) | -89.7% |
| self_collisions | -0.007 ± 0.007 | -0.001 (n=1) | -581.4% |
| tracking_ang_vel / track_angular_velocity | 0.023 ± 0.006 | 0.679 (n=1) | -96.6% |
| tracking_lin_vel / track_linear_velocity | 0.903 ± 0.171 | 1.267 (n=1) | -28.7% |
| upright | 1.605 ± 0.079 | 1.713 (n=1) | -6.4% |

## Termination 构成对比

| term (UniLab / 上游) | UniLab 终段 | 上游终段 | 相对差 |
| --- | --- | --- | --- |
| nan_state | 0.000 ± 0.000 | 0.000 (n=1) | n/a |
| tilt / fell_over | 0.214 ± 0.125 | 0.501 (n=1) | -57.4% |
| time_out | 4.038 ± 0.066 | 3.914 (n=1) | +3.1% |
上游独有 term: `out_of_terrain_bounds`

## Termination 终段占比

| term | UniLab 占比 | 上游占比 |
| --- | --- | --- |
| nan_state | 0.0% | 0.0% |
| tilt | 5.0% | 11.3% |
| time_out | 95.0% | 88.7% |
| out_of_terrain_bounds (上游独有) | — | 0.0% |


## 结论与归因

**统计口径是否一致：部分一致。** 两侧在相同预算（2000 iter × 24 × 4096 envs）下都收敛到稳定行走（episode length 终段均 >940/1000，time_out 占比 88.7% vs 95.0%，head_pose_tracking、head_pose_bias、upright 等头部/姿态项几乎相同），说明对齐后的 UniLab 配方在任务结构、reward 构成与学习动态上与上游同族。但整体 return 仍有 -26.4% 差距，且上游收敛快得多（128 vs 785 iteration 达 80%）。

**剩余差距的定量归因**（按观察到的模式与 roadmap 决策 A 的已知剩余项对照）：

1. **执行器模型（PD vs BAM，决策 A 保留项）——与最大差异项的模式一致**：UniLab 侧的平滑性/约束类惩罚显著更大——body_ang_vel -0.349 vs -0.031（约 11 倍）、foot_swing_height -0.077 vs -0.003、foot_clearance -0.022 vs -0.004、dof_pos_limits -0.025 vs -0.001、self_collisions -0.007 vs -0.001；tracking_ang_vel 0.023 vs 0.679（几乎没学会跟踪 yaw 指令）、pose 0.062 vs 0.608。XML PD（kp=50/kv=0.5，100→200Hz 已对齐）与 BAM 电压伺服（电流限幅、每 substep 摩擦预算、15–30ms 命令延迟）的力矩响应特性不同，PD 侧策略以更大的姿态扰动和更糙的步态换取主要追踪项，符合"驱动动力学差异主导剩余差距"的预期。注意本报告未做消融（上游未改动、UniLab 无 BAM 实现），归因基于模式一致性而非受控实验。
2. **环境项**：mujoco-warp 3.10.0.3 vs 上游 3.8.1（contact solver 实现有版本演进），影响量级未知，预计小于执行器项。
3. **统计功效**：上游 n=1（seed43 按 maintainer 决策跳过），上述 ± 仅来自 UniLab 双 seed；上游数值的 seed 间方差未知，结论按趋势解读。
4. **已排除项**：audit（`scripts/audit_microduck_alignment.py`）确认 PPO 超参、obs/commands/curriculum、物理积分、reset/DR、reward 栈、seed/规模全部逐项 match（184 项 MATCH / 0 GAP），差异不再来自配置漂移。

**后续建议**：若需进一步收敛剩余差距，启动 roadmap #1452 决策项①的路线 C（复刻 BAM 执行器，独立 roadmap 规模）；或将本报告作为路线 A 的验收终点归档。

## 数据与复现

- 生成命令见 `logs/alignment_1457/run_all.sh`（训练驱动，未入库）；完整曲线摘要 `logs/alignment_1457/report.json`（未入库，可由 compare 脚本重新生成）。
- UniLab runs：`logs/rsl_rl_ppo/MicroduckVelocityFlat/2026-09-03_18-54-37_mjwarp`（seed 42）、`2026-09-03_19-20-43_mjwarp`（seed 43）。
- 上游 run：microduck_rl `logs/rsl_rl/velocity/2026-09-03_19-48-28_seed42`。
