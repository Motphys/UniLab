# EXP-005：只收紧前向速度跟踪宽度

状态：配置已安装到当前运行副本，单变量解析检查与短训练入口已验证；
两组正式 3000 次更新尚未执行，尚无性能结论。

## 先理解这个变量

把奖励理解成“速度答卷的评分标准”：偏离目标多少，就扣掉多少分。
当前 1.3.3 实现 `track_lin_vel_xy_exp` 返回：

`r_tracking = exp(-((vx_command-vx)^2 + (vy_command-vy)^2) / std^2)`。

实验仅将 std 从 0.5 改为 0.35 m/s，weight 保持 2.0。
下面是尚未乘 weight/dt 的单个分项，假设 vy 误差为 0：

| vx 误差 (m/s) | std=0.5 | std=0.35 |
|---|---:|---:|
| 0 | 1.0000 | 1.0000 |
| 0.1 | 0.9608 | 0.9216 |
| 0.2 | 0.8521 | 0.7214 |
| 0.5 | 0.3679 | 0.1299 |

候选假设：更严格的速度奖励能改善跟踪。
相反结果也可能发生：误差大时奖励很小，策略更难获得有区分度的信号。
所以 std 小不是天然更好。这不是 PPO 的动作采样标准差，也不是动作缩放或奖励权重。

## 前置条件与两组定义

两组均使用 EXP-004 修复后的相同源码，从头初始化、种子 42、64 环境 × 24 rollout ×
3000 updates = 4,608,000 transitions，设备均 CUDA（物理仍为 CPU MuJoCo）。
基线重新使用原前进任务，pose 惩罚为 -0.1；不继承 EXP-003 的 no_pose 配置。
本次不叠加新的奖励权重、学习率、scale 或训练预算调整。

| 组 | task | std | 新输出文件 |
|---|---|---:|---|
| 修复后基线 | go2_joystick_flat/rma_forward | 0.5 | teacher_mapping_v1.pt |
| 收紧宽度 | go2_joystick_flat/rma_forward_tight_tracking | 0.35 | teacher_mapping_std035_v1.pt |

旧的 teacher_forward_v1 / teacher_forward_no_pose_v1 在错误映射下训练，
不作为本次 std 对照的基线。不要通过加载旧权重伪装成从头训练。
一个训练种子的结果只用于初步比较，三个评估种子不是三次独立训练。

## 到底改哪个文件

当前训练机器的文件：
`/home/abc/my_unilab/src/unilab/conf/ppo/task/go2_joystick_flat/rma_forward_tight_tracking.yaml`。

```yaml
# @package _global_
defaults:
  - /task/go2_joystick_flat/rma_forward
  - _self_

reward:
  tracking_lin_vel:
    params:
      std: 0.35
```

`defaults` 继承基线，`_self_` 让本文件最后覆盖它。
只写 std，其他参数继承；不要修改 base.yaml，也不要在 train_rma.py 中改 reward。
训练和评估通过 `--task` 选择配置；改 YAML 不会修改已保存网络的行为。

以后学习修改时，在 WSL 执行：

```bash
nano /home/abc/my_unilab/src/unilab/conf/ppo/task/go2_joystick_flat/rma_forward_tight_tracking.yaml
```

只修改 `std: 0.35` 这一行；Ctrl+O、Enter 保存，Ctrl+X 退出。
本轮先保持 0.35 不变。下一轮若换成 0.4，先保存旧配置、另起实验编号和模型名，
不能悄悄修改本轮协议后继续把结果标成 EXP-005。标准实验配置和命令应提交 Git。

### 验证有没有误改第二个变量

```bash
cd /home/abc/my_unilab
uv run --no-sync python scripts/inspect_rma_task.py \
  --task go2_joystick_flat/rma_forward_tight_tracking \
  --compare-with go2_joystick_flat/rma_forward \
  --expect-only reward.tracking_lin_vel.params.std
```

应看到 changes 只有 std：before=0.5、after=0.35。多改了一项或完全没改都会返回非零状态。
这个命令不训练、不创建物理环境；只解析实际 Hydra 配置。
已实测的比较结果保存在 `runs/rma/experiments/EXP-005/config_comparison.json`。

## 训练：两组依次运行，不同时占用电脑

1. 先运行 EXP-004 的修复后基线训练，并完成下面评估。
2. 基线有明确评估记录后，仍想测试这个假设，再运行实验组。

```bash
cd /home/abc/my_unilab
uv run --no-sync python scripts/train_rma.py \
  --phase teacher --task go2_joystick_flat/rma_forward_tight_tracking \
  --output runs/rma/teacher_mapping_std035_v1.pt \
  --num-envs 64 --iterations 3000 --device cuda
```

两组正式训练都不加 `--checkpoint`，均从头训练。输出已存在时脚本会拒绝覆盖；
重做实验应明确改输出名，不能删除旧模型来掩盖重跑。
日志分别在 `teacher_mapping_v1_logs/<unique>/` 与
`teacher_mapping_std035_v1_logs/<unique>/`，包括 metrics.csv / tensorboard。
不要在高温下继续长训；本轮没有启动这两组长训练。

## 每组训练完成后：用共同任务评估

为避免不同奖励定义误导总 reward，正式比较统一在原前进任务评估：
即实验组网络也选择 `--task go2_joystick_flat/rma_forward`。
这种做法不会改变网络，且两组的评估奖励、指令、动力学一致。
主要比较实测速度与 RMSE，而非只比较各自的训练 reward。

基线训练结束后执行；实验组训练完再把 model 一行改为 `teacher_mapping_std035_v1`：

```bash
cd /home/abc/my_unilab
model=teacher_mapping_v1
for seed in 123 456 789; do
  uv run --no-sync python scripts/evaluate_rma.py \
    --policy teacher --checkpoint "runs/rma/${model}.pt" \
    --task go2_joystick_flat/rma_forward \
    --num-envs 8 --steps 1000 --seed "$seed" \
    --command 0.5 0 0 --diagnostics \
    --trace-dir "runs/rma/evaluations/EXP-005/${model}/seed${seed}" || break
done
```

结果：终端 JSON + 每次独立 trace 目录中的 summary.json / transitions.csv / run_config.json。
逐项比较 `diagnostics.mean_measured_velocity[0]`、
`diagnostics.tracking_rmse_by_axis[0]`、侧向/转向误差及提前终止。
`terminations` 不是独立测量的跌倒率。
每回合排除前 100 步的 trace 窗口另行对照，同时保留全程指标。
训练任务源码应在改动前存档；trace 快照表示评估时配置，不能用它倒推以前训练配置。

画图：对评估输出的实际 trace.directory 执行：

```bash
uv run --no-sync python scripts/plot_evaluation_trace.py /实际的/trace.directory
```

## 两个版本不要混用

本记录和上述命令针对当前 1.3.3 的 `std`，运行副本安装的源文件来自
`experiments/rma/runtime_1_3_3/rma_forward_tight_tracking.yaml`。
GitHub 1.3.4 的同名任务使用 `tracking_sigma`，公式为 exp(-error/tracking_sigma)：
新宽度 0.35 对应分母 0.1225，且该版本的原基线分母是 0.5，等效 std 约 0.7071。
因此 1.3.4 的 .5 → .1225 与本记录的 std .5 → .35 不是同一个基线变化；
1.3.4 测试仅保护该版本单字段变更，不算本轮运行数据。
不要把两个版本完整覆盖或混为同一实验。新版本需独立编号、重新建基线。

## 本轮已经执行与尚未执行

- 已修复并备份控制映射，残差验证为 0。
- 已解析实际 1.3.3 配置，确认仅 std .5 → .35；reward weight 等其他项一致。
- 两组各 2 环境 × 1 次更新的 CPU 短训练用于接口验证，模型名为
  `smoke_mapping_baseline_20261023.pt` 与 `smoke_tight_std_20261023.pt`。
- 正式新基线和 std 对照未运行，没有效果结论；不预填成功数值。
