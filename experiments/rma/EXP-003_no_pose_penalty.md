# EXP-003：关闭默认站姿偏离惩罚的单变量对照

状态：配置与命令已准备；正式 3000 次更新训练尚未执行，不含性能结果。

入口验证：原 1.3.3 运行目录中，2 环境、1 次更新、CPU 冒烟训练已完成并保存日志和模型。
对应文件 `runs/rma/smoke_no_pose_20261022_2305.pt`，日志目录
`runs/rma/smoke_no_pose_20261022_2305_logs/20261022_230512_v5z9y6n2/`。
这 48 条训练转移用于检查链路，不作为正式对照组或行走性能证据。
随后用同一实验任务完成 2 环境 × 3 步评估，惩罚分项为 0，原基线模型 SHA256 不变。

## 假设与唯一变化

问题：已有 `teacher_forward_v1.pt` 在 0.5 m/s 指令下几乎不前进。
假设：默认站姿偏离惩罚可能抑制学习有效步态；这是待检验的解释，不是已确认根因。

基线任务 `go2_joystick_flat/rma_forward` 的 `reward.similar_to_default.weight` 为 -0.1。
实验任务 `go2_joystick_flat/rma_forward_no_pose` 继承基线，只将该值改为 0.0。
Hydra 完整解析后的配置测试要求除这一项外完全一致。
基线配置文件保持原值；不要为了本实验直接修改 `base.yaml`。

## 训练协议

从头训练，初始化种子 42，64 环境，采样长度 24，更新 3000 次，总转移 4,608,000。
与基线使用同一设备设置 CUDA，其余 PPO、网络、控制、随机化与指令设置不变。
不传 `--checkpoint`；该选项会加载旧权重，改变实验起点。
当前入口也不恢复 optimizer/RNG，因此不能把加载权重称为精确续训。

在原可运行 WSL 目录执行：

```bash
cd /home/abc/my_unilab
uv run --no-sync python scripts/train_rma.py \
  --phase teacher \
  --task go2_joystick_flat/rma_forward_no_pose \
  --output runs/rma/teacher_forward_no_pose_v1.pt \
  --num-envs 64 --iterations 3000 --device cuda
```

原基线：`/home/abc/my_unilab/runs/rma/teacher_forward_v1.pt`。
新模型：`/home/abc/my_unilab/runs/rma/teacher_forward_no_pose_v1.pt`。
新训练日志：`/home/abc/my_unilab/runs/rma/teacher_forward_no_pose_v1_logs/<unique>/`，
包含 `metrics.csv`、`run_config.json` 与 `tensorboard/`。
入口会拒绝已经存在的输出路径；重复正式实验请使用新文件名。
同一次训练中周期性更新本次的新 checkpoint 属于正常保存行为。

## 评估协议

正式训练完成后再执行；评估同样选择实验任务，确保环境与奖励配置对应。

```bash
for seed in 123 456 789; do
  uv run --no-sync python scripts/evaluate_rma.py \
    --policy teacher \
    --checkpoint runs/rma/teacher_forward_no_pose_v1.pt \
    --task go2_joystick_flat/rma_forward_no_pose \
    --num-envs 8 --steps 1000 --seed "$seed" \
    --command 0.5 0 0 --diagnostics \
    --trace-dir "runs/rma/evaluations/EXP-003/seed${seed}" || break
done
```

主要指标是平均前向速度与前向 RMSE；侧向/转向误差、提前终止及动作范围辅助判断。
两组总奖励不可直接比较：取消负项，即使运动不变也可能提高 reward。
第一轮只匹配基线的一个训练种子，结论限于初步诊断；多个评估种子不能替代独立训练种子。
若无改善应记录结果，不同时再调整动作缩放或训练预算。
