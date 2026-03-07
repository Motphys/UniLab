# UniLab 代码重构 TODO List

## 执行日期
开始：2026-03-07

---

## 阶段 1：提取共享组件（优先级 1）

### ✅ Task 1.1: 移动 SharedObsNormStats 到 IPC 层

**目标**：将 SharedObsNormStats 从 SAC runner 移到独立的 IPC 模块

**步骤**：
- [ ] 1.1.1 创建 `unilab/ipc/shared_obs_stats.py`
- [ ] 1.1.2 从 `fast_sac/runner.py` 复制 SharedObsNormStats 类
- [ ] 1.1.3 在 `unilab/ipc/__init__.py` 中导出 SharedObsNormStats
- [ ] 1.1.4 修改 `fast_sac/runner.py`：删除类定义，改为导入
- [ ] 1.1.5 修改 `fast_td3/runner.py`：更新导入路径
- [ ] 1.1.6 测试：运行 SAC 和 TD3 训练确保功能正常

**影响文件**：
- 新建：`unilab/ipc/shared_obs_stats.py`
- 修改：`unilab/ipc/__init__.py`
- 修改：`unilab/algos/torch/fast_sac/runner.py`
- 修改：`unilab/algos/torch/fast_td3/runner.py`

**预期收益**：消除跨算法模块依赖

---

### ✅ Task 1.2: 提取 EmpiricalNormalization 到 common

**目标**：将重复的观测归一化类提取到通用模块

**步骤**：
- [ ] 1.2.1 创建 `unilab/algos/torch/common/normalization.py`
- [ ] 1.2.2 从 `fast_td3/learner.py` 复制 EmpiricalNormalization 类（254-305 行）
- [ ] 1.2.3 在 `unilab/algos/torch/common/__init__.py` 中导出
- [ ] 1.2.4 修改 `fast_sac/learner.py`：删除类定义，改为导入
- [ ] 1.2.5 修改 `fast_td3/learner.py`：删除类定义，改为导入
- [ ] 1.2.6 测试：运行 SAC 和 TD3 训练确保归一化正常工作

**影响文件**：
- 新建：`unilab/algos/torch/common/normalization.py`
- 修改：`unilab/algos/torch/common/__init__.py`
- 修改：`unilab/algos/torch/fast_sac/learner.py`
- 修改：`unilab/algos/torch/fast_td3/learner.py`

**预期收益**：减少 50 行重复代码

---

### ✅ Task 1.3: 提取分布式 Critic 网络到 common

**目标**：将 SAC 和 TD3 共享的 C51 Critic 提取到通用模块

**步骤**：
- [ ] 1.3.1 创建 `unilab/algos/torch/common/networks.py`
- [ ] 1.3.2 从 `fast_td3/learner.py` 复制以下类：
  - DistributionalQNetwork (28-111 行)
  - Critic (117-170 行)
- [ ] 1.3.3 在 `unilab/algos/torch/common/__init__.py` 中导出
- [ ] 1.3.4 修改 `fast_sac/learner.py`：删除类定义，改为导入
- [ ] 1.3.5 修改 `fast_td3/learner.py`：删除类定义，改为导入
- [ ] 1.3.6 测试：运行 SAC 和 TD3 训练确保 Q 值计算正确

**影响文件**：
- 新建：`unilab/algos/torch/common/networks.py`
- 修改：`unilab/algos/torch/common/__init__.py`
- 修改：`unilab/algos/torch/fast_sac/learner.py`
- 修改：`unilab/algos/torch/fast_td3/learner.py`

**预期收益**：减少 130 行重复代码

---

## 阶段 2：统一数值稳定性（优先级 3，提前执行）

### ✅ Task 2.1: 创建数值稳定性工具模块

**目标**：统一 SAC 和 TD3 的数值稳定性策略

**步骤**：
- [ ] 2.1.1 创建 `unilab/algos/torch/common/stability.py`
- [ ] 2.1.2 实现 `check_nan_loss()` 函数
- [ ] 2.1.3 实现 `clip_gradients()` 函数
- [ ] 2.1.4 实现 `safe_tensor()` 函数
- [ ] 2.1.5 在 `unilab/algos/torch/common/__init__.py` 中导出
- [ ] 2.1.6 修改 `fast_td3/learner.py`：使用统一工具
- [ ] 2.1.7 修改 `fast_sac/learner.py`：使用统一工具
- [ ] 2.1.8 测试：验证 NaN 检测和梯度裁剪正常工作

**影响文件**：
- 新建：`unilab/algos/torch/common/stability.py`
- 修改：`unilab/algos/torch/common/__init__.py`
- 修改：`unilab/algos/torch/fast_sac/learner.py`
- 修改：`unilab/algos/torch/fast_td3/learner.py`

**预期收益**：统一数值稳定性策略，便于调试

---

## 阶段 3：抽象训练循环（优先级 2，暂缓）

### ⏸️ Task 3.1: 创建 OffPolicyRunner 基类

**状态**：暂缓执行（风险较高，需要更多测试）

**原因**：
- 涉及核心训练逻辑重构
- 需要完整的回归测试套件
- 建议在阶段 1-2 完成并稳定后再执行

---

## 阶段 4：配置管理（优先级 4，暂缓）

### ⏸️ Task 4.1: 创建配置类

**状态**：暂缓执行

**原因**：
- 非紧急优化
- 可以在后续迭代中实现

---

## 执行计划

### 今日执行（2026-03-07）

**执行顺序**：
1. Task 1.1: 移动 SharedObsNormStats（15 分钟）
2. Task 1.2: 提取 EmpiricalNormalization（20 分钟）
3. Task 1.3: 提取分布式 Critic（25 分钟）
4. Task 2.1: 创建数值稳定性工具（30 分钟）

**总预计时间**：~90 分钟

**测试计划**：
- 每个 Task 完成后运行快速测试（50 iterations）
- 所有 Task 完成后运行完整测试（500 iterations）

---

## 风险控制

### 回滚策略
- 每个 Task 开始前创建 git commit
- 如果测试失败，立即回滚到上一个 commit
- 保留原始代码作为注释，便于对比

### 测试命令
```bash
# 快速测试 TD3
python scripts/train_fast_td3.py --task Go2LocoFlatTerrain --max_iterations 50

# 快速测试 SAC
python scripts/train_fast_sac.py --task Go2JoystickFlatTerrain --max_iterations 50

# 完整测试
python scripts/train_fast_td3.py --task Go2LocoFlatTerrain --max_iterations 500
python scripts/train_fast_sac.py --task Go2JoystickFlatTerrain --max_iterations 500
```

---

## 进度跟踪

- [ ] 阶段 1 完成（Task 1.1-1.3）
- [ ] 阶段 2 完成（Task 2.1）
- [ ] 所有测试通过
- [ ] 文档更新
- [ ] Code review

