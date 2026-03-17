# Reward Config Injection System - Complete Implementation

## Overview

A type-driven reward configuration system that moves reward parameters from hardcoded dataclasses to Hydra YAML files, supporting backend-specific (MuJoCo/Motrix) and algorithm-specific (SAC/TD3/APPO/RSL-RL) configurations.

## Core Architecture

### Type-Driven Automatic Conversion

**Location**: `src/unilab/base/registry.py` (15 lines)

```python
from typing import get_type_hints

# In registry.make():
if env_cfg_override is not None:
    type_hints = get_type_hints(env_cfg.__class__)
    for key, value in env_cfg_override.items():
        if hasattr(env_cfg, key):
            if isinstance(value, dict) and key in type_hints:
                target_type = type_hints[key]
                if hasattr(target_type, "__dataclass_fields__"):
                    value = target_type(**value)
            setattr(env_cfg, key, value)
```

**Key Innovation**: Uses Python's type annotation system to automatically convert dict → dataclass. No manual type mapping needed.

## Supported Algorithms

| Algorithm | Training Script | Runner | Worker | Configs | Status |
|-----------|----------------|--------|--------|---------|--------|
| SAC | ✅ | ✅ | ✅ | 6 YAML | ✅ Verified |
| TD3 | ✅ | ✅ | ✅ | 6 YAML | ✅ Verified |
| APPO | ✅ | ✅ | ✅ | 3 YAML | ✅ Verified |
| RSL-RL (PPO) | ✅ | ✅ | ✅ | 3 YAML | ✅ Verified |

## Backend Support

- **MuJoCo**: All algorithms (Go1, G1)
- **Motrix**: SAC/TD3 (Go1, G1)

## Configuration Flow

```
Hydra YAML (cfg.reward)
    ↓
train_*.py: OmegaConf.to_container(cfg.reward)
    ↓ Convert to plain dict
Runner.__init__(env_cfg_override={"reward_config": reward_dict})
    ↓
OffPolicyRunner.start_collectors(env_cfg_override)
    ↓ Pass through multiprocessing
off_policy_collector_fn(env_cfg_override)
    ↓
registry.make(env_cfg_override=env_cfg_override)
    ↓ Type-driven conversion: dict → RewardConfig dataclass
Env uses env_cfg.reward_config
```

## File Modifications

### Core Files (8 modified)

1. **`src/unilab/base/registry.py`**
   - Added type-driven override logic (15 lines)

2. **`scripts/train_offpolicy.py`**
   ```python
   env_cfg_override = None
   if hasattr(cfg, "reward") and cfg.reward:
       from omegaconf import OmegaConf
       reward_dict = OmegaConf.to_container(cfg.reward, resolve=True)
       env_cfg_override = {"reward_config": reward_dict}

   return FastSACRunner(
       env_name=cfg.training.task_name,
       env_cfg_override=env_cfg_override,
       # ...
   )
   ```

3. **`scripts/train_appo.py`**
   ```python
   def main(cfg: DictConfig) -> None:
       ensure_registries()
       from omegaconf import OmegaConf  # CRITICAL: Import at function top

       env_cfg_override = {}
       if hasattr(cfg, "reward") and cfg.reward:
           reward_dict = OmegaConf.to_container(cfg.reward, resolve=True)
           env_cfg_override["reward_config"] = reward_dict

       rl_cfg = OmegaConf.to_container(cfg.algo, resolve=True)
       # ... pass env_cfg_override to APPORunner
   ```

4. **`scripts/train_rsl_rl.py`**
   ```python
   def main(cfg: DictConfig) -> None:
       ensure_registries()
       from unilab.base import registry
       from omegaconf import OmegaConf  # CRITICAL: Import at function top

       env_cfg_override = {}
       if hasattr(cfg, "reward") and cfg.reward:
           reward_dict = OmegaConf.to_container(cfg.reward, resolve=True)
           env_cfg_override["reward_config"] = reward_dict

       # ... pass to registry.make()
   ```

5. **`src/unilab/algos/torch/fast_sac/runner.py`**
   - Added `env_cfg_override: dict[str, Any] | None = None` parameter

6. **`src/unilab/algos/torch/fast_td3/runner.py`**
   - Added `env_cfg_override: dict | None = None` parameter

7. **`src/unilab/algos/torch/offpolicy/runner.py`**
   - Added `env_cfg_override: dict | None = None` parameter
   - Stored as `self.env_cfg_override`
   - Passed to worker subprocess

8. **`src/unilab/algos/torch/offpolicy/worker.py`**
   - Added `env_cfg_override: dict | None = None` to both functions
   - Passed to `registry.make()`

### Configuration Files (12 YAML)

**SAC/TD3** (`conf/offpolicy/reward/`):
- `default.yaml` - Empty (backward compatibility)
- `go1_sac_mujoco.yaml` - Go1 + SAC + MuJoCo
- `go1_sac_motrix.yaml` - Go1 + SAC + Motrix
- `g1_sac_mujoco.yaml` - G1 + SAC + MuJoCo
- `g1_sac_motrix.yaml` - G1 + SAC + Motrix
- `g1_td3_mujoco.yaml` - G1 + TD3 + MuJoCo

**APPO** (`conf/appo/reward/`):
- `default.yaml`
- `go1_appo_mujoco.yaml`
- `g1_appo_mujoco.yaml`

**RSL-RL/PPO** (`conf/ppo/reward/`):
- `default.yaml`
- `go1_ppo_mujoco.yaml`
- `g1_ppo_mujoco.yaml`

## Test Coverage

**11/11 tests passing**

### Unit Tests

1. **`tests/config/test_reward_injection.py`** (3 tests)
   - Hydra config loading
   - Config composition
   - Override merging

2. **`tests/base/test_reward_override.py`** (2 tests)
   - Registry dict override
   - Registry dataclass override

### Integration Tests

3. **`tests/integration/test_reward_injection_integration.py`** (4 tests)
   - SAC runner integration
   - TD3 runner integration
   - SAC with Motrix backend
   - TD3 with Motrix backend

4. **`tests/integration/test_appo_rsl_reward.py`** (2 tests)
   - APPO reward override
   - RSL-RL reward override

### End-to-End Verification

All algorithms verified with actual training runs:

```bash
# SAC
uv run python scripts/train_offpolicy.py task=g1_sac algo.max_iterations=2

# TD3
uv run python scripts/train_offpolicy.py task=g1_td3 algo.max_iterations=2

# APPO
uv run python scripts/train_appo.py task=go1_appo algo.max_iterations=2

# RSL-RL
uv run python scripts/train_rsl_rl.py task=go1_ppo algo.max_iterations=2
```

## Usage Examples

### Basic Usage

```bash
# Use default config
uv run python scripts/train_offpolicy.py task=g1_sac

# Override specific scale
uv run python scripts/train_offpolicy.py task=g1_sac reward.scales.alive=20.0

# Different backend
uv run python scripts/train_offpolicy.py task=g1_sac training.sim_backend=motrix
```

### YAML Config Example

**`conf/offpolicy/reward/g1_sac_mujoco.yaml`**:
```yaml
# @package _global_
reward:
  scales:
    tracking_lin_vel: 2.0
    tracking_ang_vel: 1.5
    penalty_ang_vel_xy: -1.0
    penalty_orientation: -10.0
    penalty_action_rate: -2.0
    pose: -0.5
    penalty_feet_ori: -25.0
    feet_phase: 5.0
    alive: 10.0
  tracking_sigma: 0.25
  base_height_target: 0.754
  min_base_height: 0.3
  max_tilt_deg: 65.0
  gait_frequency: 1.5
  feet_phase_swing_height: 0.09
  feet_phase_tracking_sigma: 0.008
```

## Design Principles

1. **Zero Hardcoding**: No manual type mapping tables
2. **Automatic Extension**: New environments automatically supported
3. **Type Safety**: Leverages Python's type system
4. **Clear Responsibility**: Registry handles all conversion
5. **Backward Compatible**: All existing code works without modification

## Critical Implementation Details

### Import Scope Bug (Fixed)

**Problem**: `UnboundLocalError` in APPO and RSL-RL training scripts

**Root Cause**: `from omegaconf import OmegaConf` was inside `if hasattr(cfg, "reward")` block, but `OmegaConf.to_container(cfg.algo, resolve=True)` was used outside.

**Solution**: Move import to top of `main()` function:
```python
def main(cfg: DictConfig) -> None:
    ensure_registries()
    from omegaconf import OmegaConf  # Must be here, not in if block
```

### Multiprocess Serialization

Reward configs are passed as plain dicts through multiprocessing, then converted to dataclasses in worker processes. This avoids pickle issues with dataclass instances.

### No Redundant Computation

Reward functions are computed once per step in `_compute_reward()`. Config injection only affects initialization, not runtime computation.

## Statistics

- **Core code**: 15 lines (registry.py)
- **Modified files**: 8
- **Configuration files**: 12 YAML
- **Test files**: 4 (11 tests)
- **Deleted temporary code**: 1 module (`src/unilab/utils/reward_config.py`)

## Verification Checklist

- ✅ All 11 unit/integration tests pass
- ✅ SAC training runs successfully
- ✅ TD3 training runs successfully
- ✅ APPO training runs successfully
- ✅ RSL-RL training runs successfully
- ✅ MuJoCo backend works
- ✅ Motrix backend works
- ✅ Reward scales correctly applied
- ✅ No redundant computation
- ✅ Backward compatible
- ✅ Type-safe conversion

## Future Extensions

To add new robot/backend/algo combinations:

1. Create YAML file: `conf/{algo}/reward/{robot}_{algo}_{backend}.yaml`
2. Define reward scales and parameters
3. No code changes needed - type-driven system handles it automatically

Example:
```bash
# New robot "H1" with SAC on MuJoCo
# Just create: conf/offpolicy/reward/h1_sac_mujoco.yaml
uv run python scripts/train_offpolicy.py task=h1_sac
```
