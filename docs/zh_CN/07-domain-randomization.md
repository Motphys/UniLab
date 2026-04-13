# 域随机化现状

这页只描述当前仓库里已经注册、且已经接入 DR provider 的任务现状。结论全部来自代码，不按设计意图推断。

当前统一入口在 [`NpEnv._init_domain_randomization()`](../../src/unilab/base/np_env.py) 和 [`DomainRandomizationManager`](../../src/unilab/dr/manager.py)：

- reset 路径：task provider 产出 `ResetPlan`，manager 校验 capability 后调用 backend `set_state(..., randomization=...)`
- interval 路径：task provider 产出 `IntervalRandomizationPlan`，manager 在 step 前按需调用 backend `apply_interval_randomization(...)`

## 现状结论

1. 当前 7 个已注册任务全部使用统一 DR 入口，没有任务绕开 `DomainRandomizationManager` 直接在 `reset()` 里做另一套 DR 流程。
2. 形式上基本都是结构化的：任务文件内定义 `domain_rand` 配置 dataclass、`DomainRandomizationProvider`、`ResetPlan`，`G1WalkTaskMjSAC` 复用 `G1Joystick` 的 provider。
3. 现在“统一”的主要是入口和执行流程，不是所有随机项本身。公共 helper [`build_common_reset_randomization()`](../../src/unilab/dr/dr_utils.py) 目前只生成 `base_mass_delta` 和 `base_com_offset`；公共 interval helper 目前只生成 push。
4. [`ResetRandomizationPayload`](../../src/unilab/dr/types.py) 已经能表达 `body_iquat`、`body_inertia`、`kp`、`kd`，且 [`MuJoCoBackend`](../../src/unilab/base/backend/mujoco_backend.py) 已声明支持；但当前没有任何任务 provider 真正采样并下发这些项。对任务侧来说，它们还没有形成统一可配置的 DR 项。
5. [`MotrixBackend`](../../src/unilab/base/backend/motrix_backend.py) 当前只支持 `base_mass_delta`、`base_com_offset` 和 interval push。

## 统一性评估表

| 任务 | 是否使用统一 DR 入口 | 是否为结构化形式 | reset 形式 | interval 形式 | 代码 |
| --- | --- | --- | --- | --- | --- |
| `Go1JoystickFlatTerrain` | 是 | 是：`Domain_Rand + Provider + ResetPlan` | 任务状态采样 + common payload | push | [`go1/joystick.py`](../../src/unilab/envs/locomotion/go1/joystick.py) |
| `Go2JoystickFlatTerrain` | 是 | 是：`Domain_Rand + Provider + ResetPlan` | 任务状态采样 + common payload | push | [`go2/joystick.py`](../../src/unilab/envs/locomotion/go2/joystick.py) |
| `G1JoystickFlatTerrain` | 是 | 是：`Domain_Rand + Provider + ResetPlan` | 任务状态采样 + common payload | push | [`g1/joystick.py`](../../src/unilab/envs/locomotion/g1/joystick.py) |
| `G1WalkTaskMjSAC` | 是 | 是：复用 [`G1JoystickDomainRandomizationProvider`](../../src/unilab/envs/locomotion/g1/joystick.py) | 任务状态采样 + common payload | push | [`g1/joystick_sac.py`](../../src/unilab/envs/locomotion/g1/joystick_sac.py) |
| `G1MotionTracking` | 是 | 是：`Domain_Rand + Provider + ResetPlan` | 大量任务特有 reset 采样 + common payload | push | [`motion_tracking/g1/tracking.py`](../../src/unilab/envs/motion_tracking/g1/tracking.py) |
| `AllegroInhandRotation` | 是 | 是：`DomainRandConfig + Provider + ResetPlan` | 纯任务特有 reset 采样，`randomization=None` | 无 | [`inhand_rot_allegro/rotation.py`](../../src/unilab/envs/manipulation/inhand_rot_allegro/rotation.py) |
| `AllegroInhandRotationSac` | 是 | 是：`DomainRandConfigSac + Provider + ResetPlan` | 纯任务特有 reset 采样，`randomization=None` | 无 | [`inhand_rot_allegro/rotation.py`](../../src/unilab/envs/manipulation/inhand_rot_allegro/rotation.py) |

## 任务域随机化清单

| 任务 | 当前实现的 reset 域随机 | 当前实现的 interval 域随机 | 默认状态 |
| --- | --- | --- | --- |
| `Go1JoystickFlatTerrain` | base xy；base yaw；base qvel；command 采样；`current_actions/last_actions` 清零；可选 `base_mass_delta`；可选 `base_com_offset` | `push_robots` | 默认开启 `base_mass_delta`、`base_com_offset`、push |
| `Go2JoystickFlatTerrain` | base xy；base yaw；base qvel；command 采样；`current_actions/last_actions` 清零；可选 `base_mass_delta`；可选 `base_com_offset` | `push_robots` | 默认全部关闭 |
| `G1JoystickFlatTerrain` | base xy；base yaw；按 `reset_base_qvel_limit` 采样 base qvel；command 采样；`gait_phase` 采样；`current_actions/last_actions` 清零；可选 `base_mass_delta`；可选 `base_com_offset` | `push_robots` | 默认 common payload 和 push 关闭 |
| `G1WalkTaskMjSAC` | 与 `G1JoystickFlatTerrain` 相同，直接复用同一个 provider | `push_robots` | 默认 common payload 和 push 关闭 |
| `G1MotionTracking` | motion frame 采样；root pose 扰动 `x/y/z/roll/pitch/yaw`；root velocity 扰动 `x/y/z/roll/pitch/yaw`；joint position noise；MuJoCo 下按 joint range clip；`current_actions/last_actions` 清零；可选 `base_mass_delta`；可选 `base_com_offset` | `push_robots` | `pose_randomization`、`velocity_randomization`、`joint_position_range` 默认有非零扰动；common payload 和 push 默认关闭 |
| `AllegroInhandRotation` | 若有 grasp cache 则随机采样 grasp；否则对 hand joints 加 `joint_noise`、对球加 `ball_z_offset`；始终对球线速度加 `ball_vel_noise`；不下发 backend randomization payload | 无 | grasp cache 路径可用时默认会采样；`joint_noise`、`ball_vel_noise`、`ball_z_offset` 默认 0 |
| `AllegroInhandRotationSac` | 与 `AllegroInhandRotation` 相同：grasp cache 采样或 hand joint noise、ball z offset、ball velocity noise；不下发 backend randomization payload | 无 | grasp cache 路径可用时默认会采样；`joint_noise`、`ball_vel_noise`、`ball_z_offset` 默认 0 |

## 当前统一 DR 能力与边界

### 1. 统一入口已经完成

统一入口由 [`NpEnv`](../../src/unilab/base/np_env.py) 和 [`DomainRandomizationManager`](../../src/unilab/dr/manager.py) 保证：

- 任务只需要注册 provider
- manager 统一做 capability 校验
- backend 统一负责真正落地 randomization payload

所以从执行路径看，当前各任务已经统一。

### 2. 公共 helper 还比较窄

[`dr_utils.py`](../../src/unilab/dr/dr_utils.py) 当前只有两类公共 helper：

- reset common payload：`base_mass_delta`、`base_com_offset`
- interval common payload：push

这意味着：

- locomotion 任务虽然都走统一入口，但它们的 base xy、yaw、qvel、command、gait phase 仍然是各自 provider 里直接采样
- `G1MotionTracking` 的 pose / velocity / joint noise 也是任务特有逻辑
- Allegro 的 grasp / object 初始状态采样完全是任务特有逻辑

所以“统一形式”目前更多体现在 contract 和调用方式，而不是“所有任务都共用同一套随机项 schema”。

### 3. backend 能力已经超过当前任务使用范围

[`ResetRandomizationPayload`](../../src/unilab/dr/types.py) 现在包含：

- `base_mass_delta`
- `base_com_offset`
- `body_iquat`
- `body_inertia`
- `kp`
- `kd`

backend capability 当前是：

- [`MuJoCoBackend`](../../src/unilab/base/backend/mujoco_backend.py)：支持上面 6 个 reset term，且支持 interval push
- [`MotrixBackend`](../../src/unilab/base/backend/motrix_backend.py)：只支持 `base_mass_delta`、`base_com_offset`，且支持 interval push

但任务侧当前实际情况是：

- 没有任何 provider 构造 `body_iquat`
- 没有任何 provider 构造 `body_inertia`
- 没有任何 provider 构造 `kp`
- 没有任何 provider 构造 `kd`

也就是说，backend contract 已经先扩了，但任务配置层和 provider 层还没有统一迁移到这些项。

### 4. `kp/kd` 目前仍不是任务 reset DR 的统一项

虽然 MuJoCo backend 已支持 `kp/kd` reset payload，但当前任务里并没有把它们接到 `domain_rand` 配置和 provider 采样逻辑里。

现有 locomotion / Allegro 任务里，控制增益仍然主要是在环境初始化阶段通过 `create_backend(..., position_actuator_gains=...)` 设置，而不是在每次 reset 时通过 provider 下发：

- [`go1/joystick.py`](../../src/unilab/envs/locomotion/go1/joystick.py)
- [`go2/joystick.py`](../../src/unilab/envs/locomotion/go2/joystick.py)
- [`inhand_rot_allegro/rotation.py`](../../src/unilab/envs/manipulation/inhand_rot_allegro/rotation.py)
- [`inhand_rot_allegro/rotation.py`](../../src/unilab/envs/manipulation/inhand_rot_allegro/rotation.py)

因此从“当前任务 DR 状态”角度，`kp/kd` 还不能算各任务已经统一接入的域随机项。

## 新任务接入时的最低标准

如果要保持和当前代码风格一致，新任务至少应满足：

1. 在任务文件里定义自己的 DR config dataclass
2. 在任务文件里定义 `DomainRandomizationProvider`
3. reset 通过 `ResetPlan` 返回 `qpos`、`qvel`、`info_updates`、`randomization`
4. 如需 interval 扰动，通过 `IntervalRandomizationPlan`
5. 在 env 构造函数里调用 `self._init_domain_randomization(...)`

如果某个随机项要做成“统一 DR 项”，还需要同时满足三层一致：

1. [`ResetRandomizationPayload`](../../src/unilab/dr/types.py) 里有明确字段
2. backend capability 明确声明支持，并在 backend 内真正落地
3. 任务 config / provider 真正采样并下发该字段

缺任何一层，都只能算“底层有能力”或“任务里自己做了随机”，还不能算仓库层面的统一 DR 项。
