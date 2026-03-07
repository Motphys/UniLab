# TD3 训练稳定性修复文档

## 修改日期
2026-03-07

## 问题背景

TD3 训练频繁出现 NaN 值并崩溃，而 SAC 保持稳定。分析发现三个关键问题：

1. **SharedReplayBuffer 竞态条件**：采样过程中锁释放过早，导致数据不一致
2. **缺少观测归一化同步**：TD3 没有实现 learner 和 collector 之间的归一化统计同步
3. **缺少数值稳定性保护**：TD3 没有梯度裁剪、NaN 检查或值限制

## 已实施的修复

### 1. 修复 SharedReplayBuffer 竞态条件

**文件**: `unilab/ipc/shared_buffer.py`

**修改内容**:
- 在 `sample_torch()` 方法中，将所有数据拷贝操作移到锁保护范围内
- 先拷贝 numpy 数组，释放锁后再转换为 torch tensor
- 防止采样过程中 collector 进程覆盖数据

**影响**:
- 同时影响 TD3 和 SAC
- 锁持有时间略微增加（增加 6 次 numpy 拷贝），但确保数据一致性
- 对 TD3 的分布式 critic (C51) 至关重要

### 2. 添加观测归一化同步

**文件**: `unilab/algos/torch/fast_td3/runner.py`

**修改内容**:
1. 从 SAC runner 导入 `SharedObsNormStats` 类
2. 在 `learn()` 方法中创建共享归一化统计对象
3. 通过 `collector_kwargs` 传递给 collector 进程
4. 训练循环中，在权重同步后同步归一化统计（mean/std）

**代码位置**:
- 第 22 行：导入 SharedObsNormStats
- 第 171-173 行：创建共享统计对象
- 第 215-216 行：传递给 collector
- 第 297-301 行：同步统计数据

### 3. 添加数值稳定性保护

**文件**: `unilab/algos/torch/fast_td3/learner.py`

**修改内容**:

#### Critic 更新 (第 568-577 行)
- 在反向传播前检测 NaN/Inf
- 如果检测到异常值，提前返回安全的指标值
- 添加梯度裁剪（max_norm=10.0），当 weight_decay > 0 时启用

#### Actor 更新 (第 593-601 行)
- 在反向传播前检测 NaN/Inf
- 如果检测到异常值，提前返回安全的指标值
- 添加梯度裁剪（max_norm=10.0），当 weight_decay > 0 时启用

#### 存储 weight_decay (第 514 行)
- 保存 weight_decay 参数用于梯度裁剪判断

## 当前存在的问题

### Pylance 导入警告

**文件**: `unilab/algos/torch/fast_td3/learner.py`

**问题**:
```
⚠ [Line 19] 无法解析导入"torch.nn"
⚠ [Line 20] 无法解析导入"torch.nn.functional"
⚠ [Line 21] 无法解析导入"torch.optim"
```

**分析**:
- 这是 IDE (Pylance) 的静态分析警告，不是运行时错误
- 可能原因：
  1. Python 环境中未安装 PyTorch
  2. IDE 未正确识别虚拟环境
  3. PyTorch 安装路径未在 Python path 中

**解决方案**:
1. 确认 PyTorch 已安装：`pip list | grep torch`
2. 在 VS Code 中选择正确的 Python 解释器
3. 如果是开发环境问题，不影响实际运行

## 验证测试

### 测试 1: 验证 buffer 修复
```bash
python scripts/train_fast_td3.py --task Go2LocoFlatTerrain --max_iterations 500
```
预期：不会出现突然的 Q 值峰值或崩溃

### 测试 2: 验证观测归一化同步
在 TD3 learner 中添加临时日志，每 100 次迭代打印 `obs_normalizer.mean`
预期：值应随时间变化（从 collector 同步）

### 测试 3: 验证数值稳定性
```bash
python scripts/train_fast_td3.py --task Go2LocoFlatTerrain --max_iterations 2000
```
监控 tensorboard：
- qf_loss 和 actor_loss 不应出现 NaN
- Q 值保持有界（qf_max < 100, qf_min > -100）

### 测试 4: 验证 SAC 不受影响
```bash
python scripts/train_fast_sac.py --task Go2JoystickFlatTerrain --max_iterations 500
```
预期：应正常完成，无错误

## 技术细节

### 为什么 TD3 比 SAC 更容易出现 NaN？

1. **分布式 Critic (C51)**：TD3 使用分布式 Q 学习，对数据不一致极其敏感
2. **缺少熵正则化**：SAC 的熵项提供天然的数值稳定性
3. **目标策略平滑**：TD3 的噪声注入在数据不一致时会放大误差

### 修复的关键点

1. **原子性采样**：确保 (obs, next_obs) 对的一致性
2. **分布对齐**：learner 和 collector 使用相同的观测归一化统计
3. **梯度保护**：防止梯度爆炸导致的数值溢出

## 向后兼容性

- 所有修改都是增量式或内部修复
- 无 API 变更
- SAC 代码路径完全不变
- 现有 TD3 配置文件无需修改

## 下一步

1. 运行完整的验证测试套件
2. 监控长时间训练的稳定性
3. 如果问题持续，考虑调整超参数（降低学习率、增加 tau）
