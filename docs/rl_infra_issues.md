# RL Infra 代码审查问题记录

涵盖范围：`unilab/ipc/` 和 `unilab/algos/torch/fast_sac/`
审查重点：Linux/macOS 跨平台一致性、并发安全性

---

## Critical

### C1. `fused=True` AdamW — macOS MPS/CPU 必然崩溃

**文件:** `unilab/algos/torch/fast_sac/learner.py:398-417`

**问题:** `fused=True` 仅支持 CUDA backend。macOS 使用 MPS 或 CPU 时直接抛出：
```
RuntimeError: Fused AdamW is not supported for this device
```
三个 optimizer（q_optimizer、actor_optimizer、alpha_optimizer）均受影响。

**修复:** 根据 device 动态决定是否启用 fused：
```python
_fused = isinstance(device, str) and device.startswith("cuda")
```

**状态:** 已修复 ✓

---

### C2. `_get_default_device` 未实现 — device=None 时崩溃

**文件:** `unilab/algos/torch/fast_sac/runner.py`, `unilab/ipc/async_runner.py:43-46`

**问题:** `AsyncRunner` 将 `_get_default_device` 声明为 `@abstractmethod`，但 `FastSACRunner`
没有实现它。用户不传 `device` 参数时，`super().__init__()` 中 `device or self._get_default_device()`
触发 `TypeError`。`FastTD3Runner` 绕过了该问题（未继承 `AsyncRunner`）。

**修复:** 在 `FastSACRunner` 中实现 `_get_default_device()`，优先 CUDA → MPS → CPU。

**状态:** 已修复 ✓

---

## High — 并发竞争条件

### H1. `SharedWeightSync` 无锁并发读写 — 可读到半写数据

**文件:** `unilab/ipc/weight_sync.py:51-70`

**问题:** `write_weights` 写入 buffer 全程无锁，`read_weights_into` 读取也无锁。
版本号在数据写完后才递增，但在 shared memory 场景下缺少 memory barrier，
CPU/OS 可能重排序内存写操作，导致读端看到 version 已更新但 buffer 数据仍为旧值。

时序示例（危险路径）：
1. Learner: 开始写 buffer（写了一半参数）
2. OS 重排: version++ 先被其他核心可见
3. Collector: 发现 version 变化，读到半写的 actor 参数

**修复:** 增加一个 `spawn` context 的 `Lock`，write/read 各自持锁。

**状态:** 已修复 ✓

---

### H2. `SharedReplayBuffer.add_batch` 数据写入在锁外 — torn reads

**文件:** `unilab/ipc/shared_buffer.py:98-118`

**问题:** `start` 指针从无锁的 `_meta[0]` 读取，后续所有数组写入（obs、actions 等）
均在锁外进行，只有最后的 meta 更新在锁内。`sample` 在锁外读数组数据，
可能读到某次 `add_batch` 写了一半的转换数据（尤其 wrap-around 路径）。

**修复:** 整个 `add_batch` 的指针读取 + 数据写入 + meta 更新统一在锁内完成。

**状态:** 已修复 ✓

---

## Medium — 跨平台行为不一致

### M1. `multiprocessing` start method 混用

**文件:** `unilab/ipc/shared_buffer.py:9,44`, `unilab/ipc/shared_storage.py:38-40`

**问题:**
- `shared_buffer.py` 用 `_SPAWN_CTX.Lock()` — spawn context
- `shared_storage.py` 用 `mp.Value/Event` — 默认 context（Linux 下是 fork，macOS 是 spawn）

Linux 上两种 context 混用时，fork context 的同步原语传入 spawn 子进程行为不可预期。

**修复:** 在 `SharedOnPolicyStorage` 中统一改为 `_SPAWN_CTX = mp.get_context("spawn")`，
`Value` 和 `Event` 均通过该 context 创建。

**状态:** 已修复 ✓

---

### M2. `torch._foreach_mul_` / `_foreach_add_` 使用私有 API

**文件:** `unilab/algos/torch/fast_sac/learner.py:530-531`

**问题:** `torch._foreach_*` 是私有 API，macOS 上通常使用更新的 nightly 版 PyTorch
支持 MPS，私有 API 接口稳定性更差，跨版本可能 break。

**修复:** 改为显式参数循环，使用公开 API：
```python
for tgt, src in zip(self.qnet_target.parameters(), self.qnet.parameters()):
    tgt.data.mul_(1.0 - self.tau).add_(src.data, alpha=self.tau)
```

**状态:** 已修复 ✓

---

## Low — 设计缺陷

### L1. `TrainingLogger._refresh()` 是空操作

**文件:** `unilab/algos/torch/common/logger.py:360-362`

`log_buffer_fill`、`log_save`、`log_status` 均调用 `_refresh()`，但该方法体内
`if self._status != "Training": pass` 什么都不做。buffer fill 进度、checkpoint
通知在 warmup 期间不会显示。

**修复:** `_refresh()` 改为打印当前 `_status` 状态行（单行，避免在训练期间重复打印完整 panel）。

**状态:** 已修复 ✓

---

### L2. `update_actor` 多余的第二次前向传播

**文件:** `unilab/algos/torch/fast_sac/learner.py:491-495`

`get_actions_and_log_probs` 后紧接着又调用 `self.actor(obs)` 获取 `log_std`，
做了两次完整 forward pass。

**修复:** `get_actions_and_log_probs` 改为返回三元组 `(action, log_prob, log_std)`，
`update_actor` 直接复用第一次 forward 的 `log_std`，两处调用方同步更新。

**状态:** 已修复 ✓

---

### L3. `SharedReplayBuffer._lock` 靠外部赋值修补，非常脆弱

**文件:** `unilab/ipc/shared_buffer.py:47-48`, `unilab/algos/torch/common/worker.py:107`

Worker 侧重建 buffer 时 `_lock=None`，通过 `replay_buffer._lock = buffer_lock`
补丁修复。如果其他地方忘记这步，`add_batch` 中 `with self._lock:` 会抛
`TypeError: 'NoneType' object does not support the context manager protocol`。

**修复:** `SharedReplayBuffer.__init__` 增加 `lock=` 参数；`create=False` 时要求必须
传入 lock（带 `assert` 防呆）；worker 改为构造时传入，删除补丁赋值行。

**状态:** 已修复 ✓
