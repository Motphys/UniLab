# EXP-004：修复目标写入，重建训练基线

状态：1.3.3 运行副本已修复目标映射，真实短诊断通过；正式训练尚未执行。

## 问题与范围

自然关节顺序为 FL/FR/RL/RR，后端执行器存储顺序为 FR/FL/RR/RL。
原 `Entity.set_full_width_joint_position_target` 直接 `control.copy_(target)`，
未做重排。普通 `set_joint_position_target` 已正确解析关节到执行器。

把 GitHub 1.3.4 已有的 full-width 映射实现与懒加载索引缓存定点移植到当前
`/home/abc/my_unilab/src/unilab/base/entity.py`；没有覆盖整个框架或升级依赖。
源补丁见 [mapping.patch](patches/unilab_1_3_3_full_width_mapping.patch)。
该实现缓存解析后的索引，没有每步读取 XML、文件或 CPU 往返；
正确的重排有计算开销，本轮未做吞吐基准，不承诺与错误直接复制路径完全同速。

原文件备份：`runs/rma/backups/EXP-004_mapping_20261023_171400/entity.py`。
修改前 SHA256：`546487c77ea7ab8e288ac63eb7dddea08fad6de845f256a436e2151724b7e1d6`。
修改后 SHA256：`54f5126f62e71acfce6c87c051fbc5e2411264f396dd39161bb7376556810d4d`。
这些是运行副本的文件哈希，不是 GitHub 1.3.4 文件哈希。

## 验收

无物理步进的不同非零目标检查：快速/普通路径映射最大误差现均为 0（修复前快速路径为 0.0300000 rad）。
同一个旧教师、种子 123、500 步的真实控制诊断：
`runs/rma/control_diagnostics/EXP-004_mapping/eval_k8559ph8`，
`max_abs_mapping_residual_rad = 0.0`（修复前为 0.47117）。
该检查读取实际控制缓冲，不通过修改观测顺序掩盖错误。

旧策略是在错误映射下训练的，修复后直接播放的表现不能代表修复后训练的效果。
没有以旧模型不跌倒或突然走起来作为本修复的验收标准。
两组新初始化的教师入口冒烟测试见 EXP-005；它们不构成学习成功证据。

## 下一步只做这件事

从头训练修复后的原前进任务，保留 pose 惩罚 -0.1、std 0.5、scale 0.25、
初始化种子 42、64 环境、3000 次更新，不加载旧 checkpoint。
这次是工程修复后的新基线，不与 EXP-003 混为一个奖励对照。

```bash
cd /home/abc/my_unilab
uv run --no-sync python scripts/train_rma.py \
  --phase teacher --task go2_joystick_flat/rma_forward \
  --output runs/rma/teacher_mapping_v1.pt \
  --num-envs 64 --iterations 3000 --device cuda
```

评估与后续 std 实验的完整命令见 [EXP-005](EXP-005_tracking_width.md)。
大训练仍由用户自行启动；本轮仅执行短检查，没有启动后台长训。
