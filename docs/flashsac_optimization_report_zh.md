# FlashSAC 优化简报

## 改了什么

之前的 FlashSAC 优化把编译边界从单独的 loss 扩大到完整 objective，让 Inductor 可以跨越网络前向、categorical target 和 loss 进行融合；同时保留 graph-safe 的固定输出与延迟 metrics 拷贝，避免 CUDA Graph replay 覆盖正在使用的地址。

## 为什么有效

原路径会产生多个小 kernel 和中间 tensor。完整 objective compile 减少了 launch 和中间读写，固定输出则避免编译器临时 buffer 与 metrics buffer 复用同一地址。collector 仍然使用真实环境进程，learner 继续在 GPU 上更新。

## 怎么看真实训练耗时

```bash
uv run scripts/benchmark/rl/benchmark_flashsac_training.py \
  --backend mujoco --iterations 20 --num-envs 256 \
  --uni-rl-src ../unilab_rl-flashsac-optimization/src \
  --output scripts/benchmark/outputs/flashsac_training/summary.json
```

脚本会显示每轮的：

```text
step    learner(ms)  collector(ms)  iter(ms)  reward
...
```

结尾输出三组 mean、median、p90、p95。这里的 collector 时间来自真实 MuJoCo/Motrix `env.step()`，不是预生成 action 的离线测试；learner 时间来自真实 FlashSAC update。

## 结果解释

总轮时间不是 learner 和 collector 的简单串行相加，因为 double-buffer runner 会让两者部分重叠。判断优化是否有效，应比较同一 backend、task、num_envs、batch、iterations 下的完整 `iter_ms`，同时观察 learner 和 collector 两列，避免只看其中一个阶段。

本机用 G1WalkFlat + MuJoCo 做了短跑验证（16 env，batch 16，1 update/step）。eager 路径的 5 个有效样本结果为：learner `6.572 / 6.219 / 7.338 / 7.520 ms`，collector `49.814 / 9.006 / 123.028 / 140.249 ms`，总轮 `7.446 / 7.048 / 8.324 / 8.545 ms`（依次为 mean / median / p90 / p95）。开启 full-objective compile 后，去掉前 4 个编译/预热样本，3 个 steady-state 样本为：learner `2.217 / 2.215 / 2.265 / 2.271 ms`，collector `4.110 / 3.914 / 4.529 / 4.606 ms`，总轮 `3.015 / 3.020 / 3.061 / 3.066 ms`。样本很少，这组数字用于验证计时链路，不作为正式跨配置结论。

## 复现注意事项

- 需要安装 UniLab 对应的物理 backend extra；MuJoCo 是最容易复现的选择。
- 测试前先确认 `unilab_rl` 版本，推荐使用 `--uni-rl-src` 指向包含 FlashSAC 优化提交的 checkout。
- 首轮可能包含 torch.compile 编译开销；比较 steady state 时应增加 iterations，并丢弃最初的编译/预热轮次。
- 脚本可用 `--skip-first N` 丢弃前 N 条 timing 样本，JSON 仍保留完整的 `rows_all`。
- 本报告不包含 Triton categorical-target 实验；Triton 仍是单独的可选实验目录。
