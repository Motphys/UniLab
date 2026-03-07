# Off-Policy RL 统一基础设施设计

## 设计目标

1. **模块化**：解耦数据收集、存储、采样、训练
2. **可复用**：SAC/TD3 共享核心组件
3. **数据安全**：统一处理多进程数据同步,避免 NaN
4. **易扩展**：新算法只需实现 Learner 接口

**注**: APPO 使用 on-policy storage 和 V-trace，架构差异大，不纳入此框架

## 核心架构

```
┌─────────────────────────────────────────────────────────────┐
│                    OffPolicyRunner (Base)                   │
│  - 管理 Collector/Learner 进程生命周期                      │
│  - 统一的 learn() 训练循环                                  │
│  - 统一的 checkpoint/logging                                │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ DataCollector│   │ ReplayBuffer │   │   Learner    │
│              │──▶│              │──▶│  (Interface) │
│ - 环境交互   │   │ - 线程安全   │   │ - update()   │
│ - 策略采样   │   │ - 零拷贝     │   │ - save/load  │
└──────────────┘   └──────────────┘   └──────────────┘
                                              │
                            ┌─────────────────┴─────────────────┐
                            ▼                                   ▼
                      ┌─────────┐                         ┌─────────┐
                      │   SAC   │                         │   TD3   │
                      │ Learner │                         │ Learner │
                      └─────────┘                         └─────────┘
```

## 模块设计

### 1. SafeReplayBuffer (统一数据安全)

**职责**：封装所有多进程数据同步逻辑,确保数据安全

```python
class SafeReplayBuffer:
    """Thread-safe replay buffer with guaranteed data integrity."""

    def sample_torch(self, batch_size: int, device: str):
        """Sample with automatic sync to prevent race conditions."""
        with self._lock:
            # Copy data in lock
            data_copies = self._copy_batch(batch_size)

        # Convert to torch with non-blocking + explicit sync
        tensors = self._to_torch_safe(data_copies, device)
        return tensors

    def _to_torch_safe(self, data, device):
        """Safe conversion: non-blocking + explicit sync."""
        result = {k: torch.from_numpy(v).to(device, non_blocking=True)
                  for k, v in data.items()}

        # Single sync point for all transfers
        if device != "cpu":
            self._synchronize(device)

        return result
```

### 2. OffPolicyLearner (统一接口)

**职责**：定义所有 off-policy 算法的统一接口

```python
class OffPolicyLearner(ABC):
    """Base interface for off-policy RL algorithms."""

    @abstractmethod
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step. Returns metrics."""
        pass

    @abstractmethod
    def get_actor_state_dict(self) -> Dict:
        """Get actor weights for collector sync."""
        pass

    @abstractmethod
    def load_actor_state_dict(self, state_dict: Dict):
        """Load actor weights."""
        pass

    def save_checkpoint(self, path: str):
        """Default checkpoint implementation."""
        pass

    def load_checkpoint(self, path: str):
        """Default checkpoint loading."""
        pass
```

### 3. OffPolicyRunner (统一训练流程)

**职责**：管理训练循环,适配所有 off-policy 算法

```python
class OffPolicyRunner:
    """Unified runner for all off-policy algorithms."""

    def __init__(self, learner: OffPolicyLearner, config: OffPolicyConfig):
        self.learner = learner
        self.config = config
        self.buffer = SafeReplayBuffer(...)
        self.weight_sync = SharedWeightSync(...)

    def learn(self, max_iterations: int):
        """Unified training loop."""
        # Start collector process
        self._start_collector()

        # Training loop
        for iteration in range(max_iterations):
            # Wait for buffer warmup
            if self.buffer.size < self.config.warmup_steps:
                continue

            # Sample and update
            for _ in range(self.config.updates_per_step):
                batch = self.buffer.sample_torch(
                    self.config.batch_size,
                    self.config.device
                )
                metrics = self.learner.update(batch)

            # Sync weights to collector
            if iteration % self.config.sync_frequency == 0:
                self._sync_weights()

            # Logging
            self._log_metrics(metrics)
```

### 4. OffPolicyConfig (统一配置)

**职责**：标准化配置参数

```python
@dataclass
class OffPolicyConfig:
    """Unified config for off-policy algorithms."""

    # Environment
    env_name: str
    num_envs: int = 4096

    # Training
    batch_size: int = 8192
    replay_buffer_capacity: int = 1_000_000
    warmup_steps: int = 0
    updates_per_step: int = 8

    # Devices
    device: str = "auto"
    collector_device: str = "cpu"

    # Synchronization
    sync_collection: bool = True
    env_steps_per_sync: int = 1
    sync_frequency: int = 1

    # Common hyperparameters
    gamma: float = 0.97
    tau: float = 0.1
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
```

## 使用示例

### SAC

```python
# Create learner
learner = SACLearner(
    obs_dim=obs_dim,
    action_dim=action_dim,
    config=sac_config,
)

# Create runner
runner = OffPolicyRunner(
    learner=learner,
    config=OffPolicyConfig(
        env_name="Go1JoystickFlatTerrain",
        batch_size=8192,
    )
)

# Train
runner.learn(max_iterations=1000)
```

### TD3

```python
# Create learner
learner = TD3Learner(
    obs_dim=obs_dim,
    action_dim=action_dim,
    config=td3_config,
)

# Create runner (same interface!)
runner = OffPolicyRunner(
    learner=learner,
    config=OffPolicyConfig(
        env_name="Go1JoystickFlatTerrain",
        batch_size=8192,
    )
)

# Train
runner.learn(max_iterations=5000)
```

## 数据安全保证

### 问题根源

多进程环境下,异步数据传输可能导致:
1. Collector 修改共享内存时,Learner 正在读取
2. 异步 GPU 传输未完成就返回数据

### 解决方案

**SafeReplayBuffer 三层防护**:

1. **Lock 内复制**: 在锁内完成 numpy copy,确保数据快照一致
2. **Non-blocking 传输**: 使用 non_blocking=True 提升性能
3. **显式同步**: 单次 synchronize() 确保所有传输完成

```python
def sample_torch(self, batch_size, device):
    # Layer 1: Copy in lock
    with self._lock:
        data = self._copy_batch(batch_size)

    # Layer 2: Non-blocking transfer
    tensors = {k: torch.from_numpy(v).to(device, non_blocking=True)
               for k, v in data.items()}

    # Layer 3: Explicit sync
    if device != "cpu":
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

    return tensors
```

## 迁移路径

### Phase 1: 创建基础组件
- [ ] SafeReplayBuffer
- [ ] OffPolicyLearner 接口
- [ ] OffPolicyConfig

### Phase 2: 重构 SAC
- [ ] SACLearner 实现 OffPolicyLearner
- [ ] 使用 SafeReplayBuffer
- [ ] 验证性能和稳定性

### Phase 3: 重构 TD3
- [ ] TD3Learner 实现 OffPolicyLearner
- [ ] 复用 SafeReplayBuffer
- [ ] 验证性能和稳定性

### Phase 4: 统一 Runner
- [ ] 创建 OffPolicyRunner
- [ ] 迁移 SAC 到新 Runner
- [ ] 迁移 TD3 到新 Runner
- [ ] 删除重复代码

## 预期收益

1. **代码减少**: ~40% 重复代码消除
2. **维护性**: 数据安全逻辑集中管理
3. **扩展性**: 新算法只需实现 Learner
4. **可靠性**: 统一的数据安全保证
