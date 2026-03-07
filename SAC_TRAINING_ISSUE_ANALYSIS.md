# SAC 训练问题分析

## 改动总结

### 对 SAC 的改动（c771df6 → HEAD）

#### 1. learner.py 改动
**新增导入**：
```python
from unilab.algos.torch.common.normalization import EmpiricalNormalization
from unilab.algos.torch.common.stability import safe_tensor
```

**修改的代码**（第 94-98 行）：
```python
# 原代码
mean = torch.clamp(mean, -10.0, 10.0)
mean = torch.nan_to_num(mean, nan=0.0)
log_std = torch.nan_to_num(log_std, nan=self.log_std_min)

# 新代码
mean = safe_tensor(mean, nan_value=0.0, clamp_range=(-10.0, 10.0))
log_std = safe_tensor(log_std, nan_value=self.log_std_min, clamp_range=(self.log_std_min, self.log_std_max))
```

**删除的代码**：
- 删除了 EmpiricalNormalization 类定义（52 行）

#### 2. runner.py 改动
**修改导入**：
```python
# 原代码
from unilab.ipc import SharedReplayBuffer, SharedWeightSync

# 新代码
from unilab.ipc import SharedReplayBuffer, SharedWeightSync, SharedObsNormStats
```

**删除的代码**：
- 删除了 SharedObsNormStats 类定义（20 行）

---

## 潜在问题分析

### 🔴 问题 1: log_std 的 clamp 范围变化

**原代码行为**：
```python
log_std = torch.nan_to_num(log_std, nan=self.log_std_min)
# 只替换 NaN，不做 clamp
```

**新代码行为**：
```python
log_std = safe_tensor(log_std, nan_value=self.log_std_min,
                      clamp_range=(self.log_std_min, self.log_std_max))
# 先 clamp 到 [log_std_min, log_std_max]，再替换 NaN
```

**问题**：
- 原代码中 `log_std` 已经在前面通过 `tanh` 压缩到 `[log_std_min, log_std_max]`（第 90-91 行）
- 新代码又做了一次 clamp，这是**冗余的**
- 但更严重的是：`safe_tensor` 先 clamp 再 nan_to_num，而原代码只做 nan_to_num
- 如果 `log_std` 在 tanh 后已经在正确范围内，这个改动**不应该**影响训练

### 🟡 问题 2: safe_tensor 的实现细节

让我检查 `safe_tensor` 的实现：

```python
def safe_tensor(tensor, nan_value=0.0, clamp_range=(-10.0, 10.0)):
    tensor = torch.clamp(tensor, clamp_range[0], clamp_range[1])
    return torch.nan_to_num(tensor, nan=nan_value)
```

**与原代码的差异**：
- 原代码：`clamp` → `nan_to_num` → `nan_to_num`（分别处理 mean 和 log_std）
- 新代码：`safe_tensor(mean)` → `safe_tensor(log_std)`

**行为一致性**：
- 对于 `mean`：完全一致 ✅
- 对于 `log_std`：新增了 clamp，但由于前面已经 tanh 压缩，理论上不应该有影响 ⚠️

### 🟢 问题 3: 其他改动

**EmpiricalNormalization 和 SharedObsNormStats**：
- 只是代码移动，功能完全相同 ✅
- 不应该影响训练 ✅

---

## 可能的根本原因

### 假设 1: log_std clamp 导致探索不足

**分析**：
```python
# SACActor.__init__
self.log_std_max = log_std_max  # 默认 0.0
self.log_std_min = log_std_min  # 默认 -5.0

# forward 中
log_std = torch.tanh(log_std)
log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)
# 此时 log_std 已经在 [-5.0, 0.0] 范围内

# 新代码
log_std = safe_tensor(log_std, nan_value=self.log_std_min,
                      clamp_range=(self.log_std_min, self.log_std_max))
# 再次 clamp 到 [-5.0, 0.0]，理论上无影响
```

**结论**：不太可能是这个原因 ❌

### 假设 2: 导入顺序或模块初始化问题

**可能性**：
- 新增的导入可能影响模块加载顺序
- 但这种情况极少见 ❌

### 假设 3: 训练脚本或环境配置变化

**需要检查**：
- 训练脚本是否有变化？
- 环境配置是否有变化？
- 随机种子是否固定？

---

## 建议的回滚策略

### 方案 1: 完全回滚 SAC 改动（最安全）

```bash
# 回滚到 c771df6（TD3 修复后，重构前）
git checkout c771df6 -- unilab/algos/torch/fast_sac/
```

### 方案 2: 只回滚 safe_tensor 改动（保留代码移动）

修改 `fast_sac/learner.py` 第 94-98 行：
```python
# 恢复原始代码
mean = torch.clamp(mean, -10.0, 10.0)
mean = torch.nan_to_num(mean, nan=0.0)
log_std = torch.nan_to_num(log_std, nan=self.log_std_min)
```

同时删除 `safe_tensor` 导入。

---

## 需要收集的信息

1. **训练日志对比**：
   - 重构前的训练曲线
   - 重构后的训练曲线
   - 具体在哪个 iteration 开始出问题？

2. **错误信息**：
   - 是否有报错？
   - 是否有 NaN？
   - 还是只是收敛变慢/不收敛？

3. **超参数**：
   - 使用的任务是什么？
   - batch_size, learning_rate 等参数

4. **环境差异**：
   - 是否在同一台机器上测试？
   - Python/PyTorch 版本是否一致？

---

## 立即行动建议

**最快验证方法**：

1. 回滚 SAC 的 `safe_tensor` 改动（方案 2）
2. 运行快速测试（50 iterations）
3. 对比训练曲线

如果回滚后正常，说明问题确实在 `safe_tensor` 的使用上。
