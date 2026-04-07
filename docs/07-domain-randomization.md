# Domain Randomization

这页说明 UniLab 当前的域随机化设计、支持范围，以及新任务应该怎么接入。

它回答四件事：

1. 域随机化现在统一放在哪里
2. reset-time 和 interval-time 随机化分别怎么走
3. 用户侧当前能改哪些随机化项
4. 开发者新增任务时应该怎么接

## Design Goals

当前 DR 设计有几个明确约束：

- `startup` 已经并入 `reset`，初始化和 episode 结束后的重置走同一条路径
- 环境实例类里不显式按后端分支做 DR 逻辑
- 两个物理后端使用同一套调用方式
- 后端不支持的随机化能力直接报 `NotImplementedError`，不静默跳过
- 具体任务的 DR 逻辑和任务代码放在一起，不集中堆到一个中央 `providers.py`

## Architecture

抽象层在 [`src/unilab/dr/`](../src/unilab/dr/)：

- [`src/unilab/dr/types.py`](../src/unilab/dr/types.py)
  定义 `ResetPlan`、`ResetRandomizationPayload`、`IntervalRandomizationPlan`、`DomainRandomizationCapabilities`
- [`src/unilab/dr/provider.py`](../src/unilab/dr/provider.py)
  定义 `DomainRandomizationProvider` 抽象接口
- [`src/unilab/dr/manager.py`](../src/unilab/dr/manager.py)
  负责把 env、provider 和 backend 能力检查串起来
- [`src/unilab/dr/dr_utils.py`](../src/unilab/dr/dr_utils.py)
  放公共 helper，例如 base mass / COM 的通用随机化和 interval push 计划构造

具体任务实现直接放在各自任务文件里，例如：

- Go1: [`src/unilab/envs/locomotion/go1/joystick.py`](../src/unilab/envs/locomotion/go1/joystick.py)
- Go2: [`src/unilab/envs/locomotion/go2/joystick.py`](../src/unilab/envs/locomotion/go2/joystick.py)
- G1 Joystick: [`src/unilab/envs/locomotion/g1/joystick.py`](../src/unilab/envs/locomotion/g1/joystick.py)
- G1 Motion Tracking: [`src/unilab/envs/motion_tracking/g1/tracking.py`](../src/unilab/envs/motion_tracking/g1/tracking.py)
- Allegro Rotation: [`src/unilab/envs/manipulation/inhand_rot_allegro/rotation.py`](../src/unilab/envs/manipulation/inhand_rot_allegro/rotation.py)
- Allegro Rotation SAC: [`src/unilab/envs/manipulation/inhand_rot_allegro/rotation_sac.py`](../src/unilab/envs/manipulation/inhand_rot_allegro/rotation_sac.py)

统一入口在 [`src/unilab/base/np_env.py`](../src/unilab/base/np_env.py)：

- env 构造时调用 `self._init_domain_randomization(Provider())`
- `reset()` 委托给 `DomainRandomizationManager.reset(...)`
- 每步执行前调用 `apply_interval_randomization_if_due(...)`

## Runtime Flow

### Reset-Time Randomization

reset 阶段由任务 provider 产出一个 `ResetPlan`，里面包含：

- `env_ids`
- `qpos`
- `qvel`
- `info_updates`
- `randomization`

其中：

- `qpos / qvel` 负责任务自己的状态采样
- `randomization` 负责后端统一能理解的通用随机化 payload

manager 会先检查 payload 里的 term 是否被当前 backend 支持，再调用 backend 的 `set_state(...)`。

### Interval-Time Randomization

interval 阶段由任务 provider 决定当前 step 是否应该返回 `IntervalRandomizationPlan`。

目前统一出来的 interval 入口主要用于 `push_robots` 这类按固定步数触发的扰动。

如果任务没有 interval DR，实现可以直接返回 `None`。

## Current Common Capabilities

目前统一抽象出来、由后端声明支持能力的通用项有：

- `base_mass_delta`
- `base_com_offset`
- `supports_interval_push`

当前两个后端都声明支持：

- MuJoCo: [`src/unilab/base/backend/mujoco_backend.py`](../src/unilab/base/backend/mujoco_backend.py)
- Motrix: [`src/unilab/base/backend/motrix_backend.py`](../src/unilab/base/backend/motrix_backend.py)

这几个 common term 只解决“后端怎么执行”的问题，不负责“任务怎么采样”。

当前没有把 `kp/kd` 纳入统一 DR 能力。

更具体地说，下面这些控制参数现在都不是 `ResetRandomizationPayload` 的一部分：

- `actuator_gainprm`
- `actuator_biasprm`
- `dof_damping`

也就是说，文档这一版不要把 `kp/kd` DR 视为“已统一支持”的能力。如果后面要支持这类随机化，需要先在 `src/unilab/dr/types.py` 里增加新的 reset term，再由各 backend 显式声明并实现。

## Task-Specific Randomization

### Locomotion

Go1 / Go2 / G1 joystick 这类 locomotion 任务通常在 reset 时一起采样：

- base xy 位置
- yaw
- base velocity
- velocity commands
- gait phase
- 可选的 base mass / COM 随机化
- 可选的 interval push

但 `Kp / Kd` 本身目前不是 reset DR 项。它们仍然是在环境初始化阶段直接写入 backend model，例如：

- [`src/unilab/envs/locomotion/go1/base.py`](../src/unilab/envs/locomotion/go1/base.py)
- [`src/unilab/envs/locomotion/go2/base.py`](../src/unilab/envs/locomotion/go2/base.py)

配置主要在各任务文件的 `domain_rand` dataclass 里，例如：

- [`src/unilab/envs/locomotion/go1/joystick.py`](../src/unilab/envs/locomotion/go1/joystick.py)
- [`src/unilab/envs/locomotion/go2/joystick.py`](../src/unilab/envs/locomotion/go2/joystick.py)
- [`src/unilab/envs/locomotion/g1/joystick.py`](../src/unilab/envs/locomotion/g1/joystick.py)

### G1 Motion Tracking

G1 motion tracking 除了 common DR 外，还会在 reset 时额外处理：

- motion frame 采样
- pose randomization
- velocity randomization
- joint position noise / clip

相关配置在：

- `pose_randomization`
- `velocity_randomization`
- `domain_rand`

定义位置见 [`src/unilab/envs/motion_tracking/g1/tracking.py`](../src/unilab/envs/motion_tracking/g1/tracking.py)。

### Allegro In-Hand Rotation

Allegro 任务的 reset 逻辑不走 base mass / COM 这一套，而是处理：

- grasp cache 采样
- hand joint noise
- ball initial position / quaternion
- ball velocity noise
- ball z offset

同样，Allegro 的 `kp/kd` 目前也不是统一 DR 能力，而是在环境初始化时直接设置：

- [`src/unilab/envs/manipulation/inhand_rot_allegro/base.py`](../src/unilab/envs/manipulation/inhand_rot_allegro/base.py)

相关配置在：

- [`src/unilab/envs/manipulation/inhand_rot_allegro/rotation.py`](../src/unilab/envs/manipulation/inhand_rot_allegro/rotation.py)
- [`src/unilab/envs/manipulation/inhand_rot_allegro/rotation_sac.py`](../src/unilab/envs/manipulation/inhand_rot_allegro/rotation_sac.py)

## User Usage

### Training Scripts

当前训练脚本默认只会把 reward 和少量 backend-specific override 提给 `registry.make(...)`。

也就是说，DR 参数并没有统一暴露成一套稳定的 Hydra CLI override 接口。

如果你只是使用现成训练脚本：

- 优先改任务默认配置里的 DR dataclass
- 或者在脚本里额外加自己的 `env_cfg_override` plumbing

### Direct Env Construction

如果你直接调用 `registry.make(...)`，可以显式传 `env_cfg_override`：

```python
from unilab.base import registry

env = registry.make(
    "Go2JoystickFlatTerrain",
    sim_backend="mujoco",
    num_envs=1024,
    env_cfg_override={
        "domain_rand": {
            "randomize_base_mass": True,
            "added_mass_range": [-1.0, 1.0],
            "random_com": True,
            "com_offset_x": [-0.03, 0.03],
            "push_robots": True,
            "push_interval": 500,
            "max_force": [1.0, 1.0, 0.5],
        },
    },
)
```

对 G1 motion tracking，可以额外覆盖：

```python
env_cfg_override={
    "pose_randomization": {
        "yaw": (-0.1, 0.1),
    },
    "velocity_randomization": {
        "x": (-0.2, 0.2),
        "yaw": (-0.4, 0.4),
    },
}
```

对 Allegro，可以覆盖：

```python
env_cfg_override={
    "domain_rand": {
        "joint_noise": 0.05,
        "ball_vel_noise": 0.2,
        "ball_z_offset": 0.01,
    },
}
```

## Developer Usage

新增一个任务时，推荐按下面做：

1. 在任务文件里定义该任务自己的 DR provider 类
2. provider 实现 `validate(...)`
3. provider 实现 `build_reset_plan(...)`
4. provider 实现 `build_reset_observation(...)`
5. 如果任务需要 interval DR，再实现 `build_interval_randomization_plan(...)`
6. 在环境构造函数里调用 `self._init_domain_randomization(YourProvider())`

最小骨架：

```python
from unilab.dr import (
    DomainRandomizationCapabilities,
    DomainRandomizationProvider,
    IntervalRandomizationPlan,
    ResetPlan,
)


class MyTaskDomainRandomizationProvider(DomainRandomizationProvider):
    def validate(self, env, capabilities: DomainRandomizationCapabilities) -> None:
        ...

    def build_reset_plan(self, env, env_ids) -> ResetPlan:
        ...

    def build_reset_observation(self, env, env_ids, info_updates):
        ...

    def build_interval_randomization_plan(
        self, env, step_counter: int
    ) -> IntervalRandomizationPlan | None:
        return None
```

如果任务只需要 common base mass / COM / push 逻辑，优先复用：

- `validate_common_reset_randomization(...)`
- `build_common_reset_randomization(...)`
- `validate_interval_push_support(...)`
- `build_interval_push_plan(...)`

这些 helper 在 [`src/unilab/dr/dr_utils.py`](../src/unilab/dr/dr_utils.py)。

## Backend Rules

新增 DR 能力时，保持下面几条：

- backend capability 由 backend 自己声明，不由环境类猜测
- provider 只描述任务要什么，不直接操纵后端内部字段
- 不能在实例化后的任务环境里显式按 backend 分支决定 DR 行为
- 如果 backend 暂时不支持该项，直接抛 `NotImplementedError`

## Related Docs

- 后端支持范围见 [Simulation Backends](02-simulation-backends.md)
- G1 motion tracking 任务本身见 [G1 Motion Tracking](05-g1-motion-tracking.md)

## Navigation

- Previous: [Collaboration Workflow](06-collaboration.md)
- Next: [Contributing](../CONTRIBUTING.md)
