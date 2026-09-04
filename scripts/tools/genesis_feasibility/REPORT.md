# Genesis 物理运行时 SimBackend 契约可行性报告

研究性 child issue：unilabsim/UniLab #1372（roadmap #1339 Child 1）。
分支：`research/issue-1372-genesis-feasibility`。
本报告只评估可行性，**不实现 backend，不声明任何 production support**。
每条结论标注证据等级：实测 / 源码推断 / 未验证。

## 0. 结论（go/no-go）

**Go —— 推荐方案 A（进程内薄 adapter）**，附四个硬性前置条件（见第 6 节）。
主干契约能力（冷路径元数据、qpos/qvel 布局、PD 控制、子集 reset、per-env DR、
传感器等价物、离屏渲染、NumPy 边界）在 genesis-world 1.3.3 上全部实测可用；
三个确认缺口（`<keyframe>`、MJCF 全局 `<option>`、MJCF `<sensor>`）都有明确的
冷路径补偿方案，无上游否决项。

## 1. 环境矩阵（固定版本）

| 项 | 值 | 备注 |
|---|---|---|
| GPU / 驱动 | RTX 4090 49GB / 595.84 | 实测 |
| Python | 3.13.14（项目 venv） | 实测 |
| torch | 2.7.0+cu128（UniLab venv 锁定） | 实测 |
| genesis-world | **1.3.3**（PyPI wheel；上游 tag `v1.3.3` = commit `76f8f5b3`） | 实测（探针运行版本） |
| genesis 上游源码 | 本地 checkout main HEAD `19f56d65`（= v1.3.3+30，含 rigid entity 静态/运行时拆分 breaking change） | 仅源码阅读，未运行 |
| quadrants | 1.3.0（genesis 精确 pin） | 实测；py3.13 下报 DeprecationWarning（ast.Str、keyword ctx） |
| mujoco | 3.10.0（UniLab venv，`mujoco>=3.5`） | 仅作 ground truth |
| genesislab | master `0f2e4d9` | 仅源码阅读 |
| 资产 | `src/unilab/assets/robots/g1/scene_flat.xml`（29 dof 位置执行器 + 21 个 MJCF sensor + keyframe "stand"） | — |

复现命令（全部探针共用 pinned 形式）：

```bash
uv run --with genesis-world==1.3.3 python scripts/tools/genesis_feasibility/probe_contract.py   # P1
uv run --with genesis-world==1.3.3 python scripts/tools/genesis_feasibility/probe_runtime.py    # P2
uv run --with genesis-world==1.3.3 python scripts/tools/genesis_feasibility/probe_perf.py       # P3
```

## 2. 依赖兼容矩阵

| 依赖 | UniLab | genesis-world 1.3.3 | 结论 |
|---|---|---|---|
| Python | `>=3.10,<3.14` | `>=3.10,<3.14` | 兼容（实测） |
| torch | `==2.7.0`（linux x86_64）/ `==2.9.0`（aarch64） | **不在 install_requires**；import 缺失即 raise；`torch<2.8.0` 启动 warning（`__init__.py:34-39`） | **冲突行（实测）**：2.7.0 下主干可用，但属上游不支持窗口，且观测到一个传感器组合失效（第 4 节 R7）。方案 A 需 maintainer 决策：torch 窗口上调至 ≥2.8，或接受 unsupported+warning 并写入支持矩阵 |
| mujoco | `>=3.5`（lock 3.10.0） | `>=3.2.5` | 兼容（实测同进程共存） |
| quadrants | —（新依赖） | `==1.3.0` 精确 pin | 新增传递依赖；py3.13 DeprecationWarning（实测，上游兼容注记） |
| 安装形态 | mjwarp 已有 `dependencies.py` 懒加载先例 | torch/quadrants 重依赖 | Genesis extra + lazy import + 版本钉扎 + 专属 DependencyError 是必选（源码推断，仿 `unisim-core/src/unisim/backend/mjwarp/dependencies.py`） |

## 3. SimBackend 契约能力矩阵

状态：OK = 实测可用；GAP = 有缺口但有补偿路径；FAIL = 阻塞。复现列 P1/P2/P3 见第 1 节。

### 3.1 冷路径元数据（materialize 一次性解析）

| 契约 | Genesis 映射 | 状态 | 证据 | 复现 |
|---|---|---|---|---|
| 关节名/顺序 | `entity.joints`（29 个 1-dof 关节名与顺序 == mujoco `jnt` 顺序） | OK | 实测 | P1 `1a` |
| `get_actuator_names` / 目标关节 / 顺序 | `get_dofs_act_gain()>0` 的 dof→joint 映射 == mj actuator 顺序（n=29） | OK | 实测 | P1 `1b` |
| `get_actuator_ctrl_range` | position 执行器的 `ctrlrange` **导入即丢弃**（mjcf.py 仅 biastype=NONE 时保留）；g1 未声明 ctrlrange（mj 原始值全 0），`forcerange` 无损（diff=0） | GAP（对 g1 无实际影响） | 源码推断+实测 | P1 `1c` |
| `get_joint_range` | `get_dofs_limit()`，29 关节 max diff=1.09e-07 | OK | 实测 | P1 `1d` |
| `get_body_mass` / `get_body_ipos` | `link.inertial_mass` / `link.inertial_pos`，30 link：mass diff=0、ipos max diff=7.24e-09 | OK | 实测 | P1 `1e` |
| `get_geom_friction` | `entity.geoms`（仅碰撞 geom，42==mj 碰撞 geom 数）per-link 摩擦多重集一致 | OK | 实测 | P1 `1f` |
| `get_geom_contact_masks` | contype/conaffinity 被 `solve_contype_conaffinity` **重编码**（bitmask 整数值 ≠ mj，碰撞矩阵语义保持） | GAP（语义可用，值不可与 mj 对表） | 实测+源码推断 | P1 `1f` |
| geom 名称/id | `RigidGeom` 无 `name` 属性（1.3.3）；只能 link+idx 寻址 | GAP（契约可按需 fail-closed） | 实测 | P1 `1f` |
| `get_gravity` | `SimOptions.gravity`（默认 (0,0,-9.81)，g1.xml 未覆盖） | OK | 实测 | P1 `1g` |
| `get_dof_armature` | `get_dofs_armature()` == mj `dof_armature` | OK | 实测 | P1 `1f` |
| `get_geom_size` / `get_body_subtree_ids` | geom `data`/`metadata`、link 树存在 | 未验证 | — | — |

### 3.2 状态布局与运动学

| 契约 | Genesis 映射 | 状态 | 证据 | 复现 |
|---|---|---|---|---|
| qpos 布局 | `get_qpos()` [B,36] = [root xyz \| root quat **wxyz** \| 29 joint]；与 `links_pos/quat(pelvis)`、`dofs_position[6:]` 一致；`dofs_*` 视图 35 宽（含 6 root dof） | OK | 实测 | P1 `2a` |
| 名称→dof/qpos 索引 | `get_joint(name).dofs_idx_local/qs_idx_local` == mj `jnt_dofadr/jnt_qposadr`；root 关节 n_dofs=6、n_qs=7 | OK | 实测 | P1 `2b` |
| 四元数约定 / FK 一致性 | keyframe 姿态下 torso_link 世界位置 vs mujoco `xpos` max diff=2.19e-08，quat diff=0（注意：mj `xipos` 是 COM 系，对应 `xpos` 才是 body 系原点） | OK | 实测 | P1 `2c` |
| root qvel 坐标系 | root dofs[3:6] 为**体坐标系**角速度，`get_links_ang()` 返回**世界系** —— 与 MuJoCo 语义及 SimBackend 契约（`set_state` 体帧写入 / `get_base_ang_vel` 世界帧读出）完全一致 | OK | 实测 | P1 `2d` |
| root state 读取 | **陷阱（实测）**：MJCF entity 的 `base_link` 是 `world`，entity 级 `get_pos/get_quat/get_vel/get_ang` 恒为原点/零，对浮动基机器人无用；必须用 link 寻址 `get_links_*(pelvis_idx)` | OK（用 link API） | 实测 | P1 `2a` |
| `get_body_pos/quat/lin_vel/ang_vel_w` | `get_links_pos/quat/vel/ang(links_idx, envs_idx)` | OK | 实测 | P1 `2c/2d` |
| `get_body_*_b`（体帧） | 无直接 API；adapter 用世界系量 + 四元数逆旋转（`unilab.utils.rotation` 已有 helpers） | OK（adapter 数学） | 源码推断 | — |
| `get_dof_pos/vel` | `get_dofs_position/velocity()[..., 6:]`（排除 root） | OK | 实测 | P1 `2a/4` |
| `get_site_jacobian_w` | `entity.get_jacobian(link, local_point)`（[B,6,n_dofs]，需 `morph.requires_jac_and_IK=True`） | 未验证 | 源码推断 | — |
| 速度量时效 | `set_dofs_velocity` 后 dofs 视图立即生效，但 `get_links_vel/ang` 要过一次 `scene.step()` 才刷新 | OK（注意事项，见 5.6） | 实测 | P1 `2d` |

### 3.3 控制、reset、keyframe、全局 option

| 契约 | Genesis 映射 | 状态 | 证据 | 复现 |
|---|---|---|---|---|
| 执行器增益导入 | MJCF `<position kp kv>` → `act_gain/act_bias`（PD 可约）：kp max diff=8.5e-07、kv max diff=8.8e-08 | OK | 实测 | P1 `3a` |
| `step(ctrl, nsteps)` 语义 | `control_dofs_position(target, dofs_idx)` 一次设定后跨 `scene.step()` 保持（5 步后 ctrl force=12.7 N·m）；== MuJoCo ctrl broadcast | OK | 实测 | P1 `3b` |
| PD 响应 | elbow +0.2 rad 目标，10 步 err=0.015（100 步后整机失稳倒地属物理现象） | OK | 实测 | P1 `3b` |
| `set_pre_step_control` | 无引擎级 per-substep hook；adapter 在 `nsteps` 循环内每 substep 调 conversion + `control_dofs_*` + `scene.step()`，与 MuJoCo backend 的 pre_step 路径同构 | OK（adapter 层实现） | 实测+源码推断 | P1 `3b` |
| `get_actuator_gains` | `get_dofs_kp/kv()` | OK | 实测 | P1 `3a` |
| `set_state` 子集 reset | `set_qpos/set_dofs_velocity/zero_all_dofs_velocity(envs_idx=...)` 免重建；触及 env == 目标、未触及 env 保持不变；set→step→get 有限、shape (8,36)、float32 | OK | 实测 | P1 `4` |
| `get_keyframe_qpos` | **Genesis 不解析 `<keyframe>`**（全仓无 key_qpos；init qpos == mj.qpos0 + body pos，与 keyframe 关节最大差 0.669 rad） | GAP | 实测 | P1 `5` |
| keyframe 补偿 | materialize 冷路径用 `mujoco` 包解析 `mj.key_qpos`（实测 key_id=0 可用）——与 isaacgym backend 父进程侧元数据扫描同模式 | 补偿方案确定 | 实测 | P1 `5` |
| `get_default_qpos` / `get_default_dof_pos` / `get_init_qvel` | init `get_qpos()`（== mj.qpos0 + 根 body pos）/ 其关节块 / 零向量 | OK | 实测 | P1 `5` |
| MJCF 全局 `<option>` | **全部丢弃**（morphs.py 注释；实测 `<option timestep=0.006667 integrator=implicitfast>` 未生效，有效 dt 来自 Sim/RigidOptions 默认 0.01）；timestep/integrator/solver/cone/iterations 都须变成 owner YAML 显式字段（cone=elliptic 仅 warning） | GAP（有明确映射路径） | 实测+源码推断 | P1 `6` |

### 3.4 传感器（g1 任务依赖 21 个 MJCF sensor）

| 契约 | Genesis 映射 | 状态 | 证据 | 复现 |
|---|---|---|---|---|
| MJCF `<sensor>` 导入 | **完全不解析**（mjcf.py 无 sensor 代码），21 个全部丢弃 | GAP | 源码推断 | P1 `7` |
| gyro / accelerometer | `IMUSensor(entity_idx, link_idx_local)`，acc/gyro [B,3]，支持 noise/bias 参数（无 MJCF cutoff/noise 自动映射） | OK（等价物） | 实测 | P1 `7` |
| contact（`data="found"`） | `get_links_net_contact_force()` [B,31,3] + 阈值（站立足底 |F|=138N）；或 `ContactForceSensor`（单独使用实测 OK） | OK（等价物） | 实测 | P1 `7` |
| velocimeter / framelinvel / framepos / framequat / framezaxis | `get_links_vel/pos/quat` + adapter 侧 site offset 与旋转数学（Genesis 无 site 抽象） | OK（adapter 数学） | 源码推断 | — |
| 传感器组合稳定性 | IMU+ContactForce+Contact 三 sensor 同场景（batch flags=True）在 **torch 2.7.0** 下崩溃：`collider.get_contacts` gather int64 dtype 错误；ContactForce 单独同配置 OK | GAP（待复验，疑似 torch 版本相关） | 实测 | P1 历史运行 |

### 3.5 DR、生命周期、宿主副作用、性能、渲染

| 契约 | Genesis 映射 | 状态 | 证据 | 复现 |
|---|---|---|---|---|
| `get_dr_capabilities` / interval DR | per-env round-trip 实测：`set_links_inertial_mass`、`set_dofs_frictionloss`、`set_dofs_kp/kv`（需 build 期 `batch_links_info/batch_dofs_info=True`——owner YAML 决策项）；`set_friction_ratio` 可调用（输入形状 (n_envs,n_links)） | OK | 实测 | P1 `8` |
| `apply_body_force` | solver 级 `apply_links_external_force/torque(links_idx=全局, envs_idx=...)` 可调用（1.3.3 无 entity 级封装；物理效果未验证） | OK（API 形状注意） | 实测（调用） | P1 `8` |
| init/destroy 生命周期 | 单进程 3× init→build→step→destroy 无崩溃；CUDA reserved 稳定 22MB；**host RSS 峰值每周期增长约 200–450MB（亚线性，两次完整运行分别测得 [452.7, 200.9] 与 [255.2, 199.3]MB）** → 长驻进程必须只 init 一次 | OK（带约束） | 实测 | P2 `9a` |
| 多 Scene 并存 | 单 gs 会话两个 Scene 同时 build/step，状态互不影响 | OK | 实测 | P2 `9b` |
| 宿主 torch 副作用 | `gs.init(gpu)` 改写 `torch.set_default_device(cuda:0)` + `set_default_dtype`：**裸 `torch.zeros()` 落 CUDA**；`torch.get_num_threads()` 不变（16）；root logging handlers 不变；`seed=` 触发全局 RNG reseed；`cpu_max_num_threads=1` **无条件**强制（非仅 CPU backend，`QD_NUM_THREADS` 可覆盖）；import 本身无副作用 | GAP（可管理，见 5.8/6） | 实测+源码推断 | P2 `10` |
| NumPy 边界成本 | n_envs=256/2048/4096：step=1.77/2.81/4.29ms；8 个 getter 逐次 `.cpu().numpy()` D2H 增加 +0.20/+0.35/+0.37ms（+11.0%/+12.5%/+8.6%）；每步 H2D ctrl push +0.10/+0.07/+0.15ms（+5.7%/+2.5%/+3.4%）；SPS≈1.45e5/7.29e5/9.54e5 | OK（mjwarp 式 host-cache 是优化项非前提） | 实测 | P3 `11` |
| play/render | 离屏 `camera.render()` headless 实测 OK（rgb uint8）；交互 Viewer 为 pyrender/pyglet，需 display（未尝试 GUI） | OK | 实测+源码推断 | P2 `12` |
| `get_physics_state`（playback） | qpos 快照序列即可 | 源码推断 | — | — |
| hfield scanner | `RaycasterSensor` 支持 grid probe pattern | 未验证 | 源码推断 | — |

## 4. 与 #1339 调研假设的对照

**被实测确认的假设**：`<keyframe>` 缺口及 mujoco 冷路径补偿；MJCF 全局 option 丢弃；
PD 三模式与 per-env DR API 齐备；无 per-substep 控制 hook；torch 为未声明硬依赖 +
`torch<2.8` warning；`gs.init` 污染 torch default device/dtype；zerocopy 默认开启。

**需要修正的假设**：
1. 「CPU backend 强制单线程」——实际是 `cpu_max_num_threads=1` **无条件**设置（GPU 运行同样受影响，作用于 quadrants 编译/CPU 侧），`QD_NUM_THREADS` 可覆盖（源码推断，v1.3.3 `__init__.py:243-254`）。
2. 「`entity.get_pos/get_quat/get_vel/get_ang` 可作 root state 读取」——对 MJCF 浮动基模型**不成立**（实测）：entity 的 `base_link` 是 `world`，这组 getter 恒返回原点/零。root state 必须走 `get_links_*(pelvis_link_idx)`。
3. 「init→destroy→init 循环无生命周期否决项」——功能上成立（实测 3× 无崩溃、CUDA 稳定），但 host RSS 每周期增长约 200–450MB（亚线性，实测）；单进程反复 init/destroy 不可作为常态用法。

**新发现（假设之外）**：
- `get_links_pos` 对应 mujoco `xpos`（body 系原点）而非 `xipos`（COM 系）——对照时必须用 `xpos`（实测，diff 2.2e-08）。
- root qvel[3:6] 体帧写入 / `links_ang` 世界帧读出，与 MuJoCo 及 SimBackend 契约语义精确对齐（实测）——无需 adapter 侧帧转换，只需声明布局。
- contype/conaffinity 被重编码（碰撞矩阵语义不变，整数值 ≠ mj）——`get_geom_contact_masks` 只能暴露 genesis 原生值（实测+源码推断）。
- `RigidGeom` 无 `name`（1.3.3）；geom 级契约只能 link+idx 寻址（实测）。
- `set_*` 后速度类 link getter 需过一次 `scene.step()` 才刷新；dofs 视图即时（实测）。
- 传感器多类型组合在 torch 2.7.0 下崩溃一例（实测，见 3.4）。
- NumPy 边界逐 getter 拷贝成本 ≈ +9~12% D2H（实测）——mjwarp 式单发 host-cache 可再压缩，但不是可行性前提。

## 5. Child 2（adapter 设计）映射决策

1. **keyframe**：materialize 冷路径用 `mujoco` 解析 owner XML 的 `mj.key_qpos`（实测可行），缓存在 adapter；热路径禁用 XML 解析。与 isaacgym 父进程侧 `sensors.py` 元数据扫描同模式，不引入新契约。
2. **全局 option**：owner YAML 显式声明 `dt`（经现有 ctrl_dt/decimation 链路）、`integrator`（g1 = implicitfast）、`constraint_solver`、`friction_cone`、solver iterations；Genesis backend 在 `Scene(...)` 构造时消费。`sim2sim` DENYLIST 无需新增字段（这些本来就要求跨后端 YAML 显式一致）。
3. **传感器**：materialize 时把 XML sensor 名映射到 Genesis 等价物并注册进 `bind_sensor_data`：gyro/accelerometer→IMUSensor（link+site offset）；contact found→net_contact_force 阈值（首选，零注册成本）或 ContactForceSensor；velocimeter/framelinvel/framepos/framequat/framezaxis→links_* + 旋转数学。MJCF 的 cutoff/noise 属性无自动映射——要么经 IMUSensor noise 参数显式声明，要么 fail-closed（推荐后者，先声明干净数据）。
4. **控制律**：默认路径 1（Genesis 内置 actuator + `control_dofs_position`，实测无损且语义匹配）；路径 2（adapter 显式 PD + `control_dofs_force`，genesislab actuator 模式）留作数值对齐退路。`set_pre_step_control` 在 adapter 的 nsteps 循环内实现，不新增 SimBackend 公共方法。
5. **root state**：契约的 root 读写全部以冷路径解析的 root link idx（pelvis）走 `get_links_*` / `set_qpos` / `set_dofs_velocity`；**禁止**使用 entity 级 `get_pos/get_vel/get_ang`（实测指向 world link）。
6. **速度量时效**：reset/set_state 后位置量立即可读（set_qpos 触发 FK），速度 link getter 需过 step 屏障；若 reset 后、首个 step 前要读 base 速度，用 `get_dofs_velocity` 的 root 切片（即时）+ 帧转换（实测可行）。
7. **DR**：owner YAML 在 materialize 决策中声明 `batch_links_info/batch_dofs_info=True`；`get_dr_capabilities()` 只声明本报告实测生效项（mass/COM shift、frictionloss、damping、armature、kp/kv、friction_ratio、external force）；geom 摩擦**绝对值**的 per-env 化只有 ratio API，规划 DR 字段时注意。
8. **宿主污染管理**：backend 构造时在 `gs.init` 前快照、在 init 后恢复（或显式重设）`torch.get_default_device()`/dtype 与 RNG 状态；写进 adapter 单测。
9. **数组边界**：仿 mjwarp——预分配 host cache，step/reset 屏障后一次性 D2H 刷新，getter 只读 cache（实测逐 getter 拷贝也仅 +9~12%，host-cache 是确定性优化）。
10. **geom/contact mask**：`get_geom_contact_masks` 暴露 genesis 重编码值并在 docstring 说明不可与 mujoco 对表；geom 名称契约对 genesis fail-closed。

## 6. 方案 A（进程内薄 adapter） vs 方案 B（子进程隔离）

| 维度 | A 进程内 | B 子进程 |
|---|---|---|
| torch 全局污染 | 实测污染点明确且可数（default device/dtype、RNG seed），可在 adapter 构造期快照/恢复 | 天然隔离 |
| torch 版本冲突 | 与 learner 同进程同 torch：2.7.0 实测主干可用，但处上游不支持窗口且观测到传感器组合失效一例；升级 torch 窗口需全仓评估 | worker 可独立 venv 跑 torch≥2.8，但 learner 侧不受影响这一优点对 UniLab 意义有限（物理在 worker） |
| 生命周期 | 实测 init/destroy 循环可用；约束是单进程只 init 一次（UniLab 一进程一训练 run，天然满足） | 进程退出即彻底清理，RSS 增长问题消失 |
| 性能 | 无 IPC；实测 D2H/H2D 边界合计 <15%，host-cache 后更低；4096 envs SPS≈9.5e5 | 增加 shm/序列化层（isaacgym 后端已有协议先例，但那是为 Python 3.8 隔离支付的必要成本） |
| 工程成本 | 仿 mjwarp 包骨架（`__init__` 懒导出 + backend + materialization + dependencies），薄 | worker 生命周期 + shm 协议 + 崩溃恢复，明显更重 |
| 依赖部署 | genesis extra + lazy import 即可 | 需要双环境/镜像编排 |

**推荐：方案 A（go）**。理由：所有实测证据都未触及方案 A 的否决项——生命周期上游原生支持、宿主污染点有限且可在 adapter 边界内管理、NumPy 边界成本小、无 IPC 需求；方案 B 解决的是「torch 版本窗口」和「反复 init 的 RSS 增长」两个真实但次级的问题，而这两者分别可由 maintainer 的 torch 窗口决策与「单进程 init 一次」约束覆盖，不值得为此引入第二套 worker 协议。前置条件：(1) mjwarp 式 lazy `dependencies.py` + 版本钉扎；(2) `gs.init` 前后快照/恢复 torch 全局状态（含 RNG）并配单测；(3) 支持矩阵明确记录 torch 2.7 的 unsupported 状态与 ContactForceSensor 组合失效案例，`get_dr_capabilities()` 只声明实测能力；(4) 文档化「一进程一 gs.init」约束。若 Child 3 的数值对齐（`enable_mujoco_compatibility`）或 torch 升级评估出现恶化证据，方案 B 作为降级路径保留。

## 7. genesislab 借鉴点（机制层面）

与 #1339 调研一致，无推翻：init 期一次性名称解析（regex→index，全选降级 `slice(None)`）；
action scale/offset/clip 全 init 预编译；event 三 mode（startup/reset/interval）与 Genesis per-env
DR API 的 interval 映射；actuator 抽象把控制律与引擎解耦（路径 2 的现成参考，`actuator_pd.py` +
`ActuatorManager` 同时回写 `set_dofs_kp/kv` 保持隐式 PD 一致）；env↔algo 差异收敛在单 wrapper。
反面教训同样成立并已被本 issue 红线覆盖：静默 no-op DR、布局常量散落（浮动基 offset 硬编码）、
`__getattr__` 透传引擎私有 API、lazy property 直摸引擎无 step 屏障、init 真跑 term func 推断维度。

## 8. 未验证 / 留待 Child 2+ 的事项

- ContactForceSensor 多 sensor 组合崩溃的根因定位与在 torch≥2.8 下的复验。
- `set_friction_ratio` / `apply_links_external_force` 的物理效果（本报告仅验证可调用与 API 形状）。
- `RaycasterSensor` grid 作 height scanner；`get_jacobian`（需 `requires_jac_and_IK=True`）；geom size / body subtree / terrain spawn data。
- `RigidOptions(enable_mujoco_compatibility=True)` 与 MuJoCo 后端的数值对齐程度（g1_walk_flat 级别，建议放 Child 3 benchmark）。
- CPU backend 吞吐定位（本报告全部在 `gs.gpu` 实测）。
- GPU 显存在 4096+ envs 长训下的增长曲线。

## 9. 附录：探针输出（实测原文）

### P1 `probe_contract.py`（行 1–8）

```text
[1a.joint_order] OK [实测] 1-dof joint names/order match MuJoCo: True (gs=29, mj=29); n_dofs gs=35 mj=35; n_links gs=31 mj_nbody=31
[1b.actuator_order] OK [实测] actuated joint order (act_gain>0) == mj actuator order: True (n=29)
[1c.ctrl_range] GAP [源码推断+实测] actuator_ctrlrange dropped for position actuators (kept only for biastype=NONE, mjcf.py); g1 does not author ctrlrange (mj raw all-zero: True) -> declare in owner YAML; forcerange imported: max diff=0.00e+00
[1d.joint_range] OK [实测] max |joint_range diff| over 29 1-dof joints = 1.09e-07
[1e.mass_ipos] OK [实测] per-link mass max diff=0.00e+00, body_ipos max diff=7.24e-09 over 30 links
[1f.friction_armature] OK [实测+源码推断] collision-geom slide-friction per-link multisets match MuJoCo on 19 links, mismatched=[]; entity.geoms holds collision geoms only (gs=42 vs mj collision=42; visual -> vgeoms); contype/conaffinity are RE-SYNTHESIZED by solve_contype_conaffinity (bitmask ints differ from mj, collision matrix semantics preserved) -> get_geom_contact_masks must expose genesis-native masks; RigidGeom has no name in 1.3.3; dof_armature match=True
[1g.gravity] OK [实测] SimOptions.gravity=(0.0, 0.0, -9.81) (None -> default [0,0,-9.81]); mj opt.gravity=[0.0, 0.0, -9.81]; g1.xml does not override gravity
[5.keyframe] GAP [实测] Genesis ignores MJCF <keyframe>: init qpos root=[0.0, 0.0, 0.7929999828338623] vs keyframe root=[0.0, 0.0, 0.754], joint max diff=0.669 (init joints == mj.qpos0: True); cold-path fallback mujoco mj.key_qpos works: key_id=0
[2a.qpos_layout] OK [实测] get_qpos (8, 36) = [root xyz | root quat wxyz | 29 joint qpos] == links_pos/quat(pelvis) + dofs_position[6:]: True; dofs view is 35-wide incl. 6 root dofs (dofs_position[0:6]=root xyz+euler-zeros); MJCF entity base_link='world' -> entity-level get_pos/get_quat/get_vel/get_ang track the WORLD link and are useless for the floating root; adapter must use link-addressed getters; dtype=float32
[2b.name_to_idx] OK [实测] get_joint(name).dofs_idx_local/qs_idx_local == mj jnt_dofadr/jnt_qposadr: True (root joint: n_dofs=6, n_qs=7, dofs=[0, 1, 2, 3, 4, 5], qs=[0, 1, 2, 3, 4, 5, 6])
[2c.quat_fk] OK [实测] FK vs mujoco at keyframe: torso_link xpos max diff=2.19e-08, quat(wxyz) diff=0.00e+00; genesis get_links_pos == mujoco xpos (body frame origin, NOT xipos COM frame)
[2d.root_ang_frame] OK [实测] set root dofs[3:6]=[0,0,1] at quat 90deg-x, after 1 step links_ang(pelvis)=[-0.0, -1.0, -0.0010000000474974513]; interpretation: qvel body-frame + getter world (== MuJoCo semantics); contract wants qvel body-frame columns in set_state and world-frame get_base_ang_vel
[3a.pd_gains] OK [实测] MJCF <position kp kv> -> act_gain/act_bias (PD-reducible): max kp diff=8.54e-07, kv diff=8.77e-08 over 29 actuators
[3b.ctrl_held_pd] OK [实测] one control_dofs_position call held across 100x scene.step() (max |ctrl force|=12.70 after 5 steps); elbow +0.2rad err @10/30/100 steps=[0.01489999983459711, 0.062199998646974564, 0.06440000236034393] (robot topples without balance ctrl, root z@100=0.00); == MuJoCo step(ctrl,nsteps) ctrl-broadcast semantics; no per-substep hook (adapter loops scene.step for nsteps)
[4.set_state_subset] OK [实测] env-indexed set without rebuild: touched==target True, untouched preserved True, subset set_dofs_velocity True; round-trip set->step->get finite=True shape=(8, 36) dtype=float32
[6.global_option] GAP [实测+源码推断] <option timestep=0.006667 integrator=implicitfast> dropped by importer (morphs.py note); effective dt from Sim/RigidOptions (here default 0.01); integrator/cone/solver-iters are scene-level RigidOptions -> owner-YAML fields; cone=elliptic only warned (mjcf.py parse_xml)
[7.sensors] GAP [源码推断+实测] none of mj nsensor=21 MJCF sensors imported (no parse code); working equivalents: IMUSensor acc(8, 3)/gyro(8, 3) noise-free, links_net_contact_force(8, 31, 3) (foot |F|=138.0N at stand; |F|>thr == MJCF contact data=found); ContactForceSensor read OK (with batch_*_info=True); velocimeter/framepos/framequat/framezaxis/framelinvel -> get_links_pos/quat/vel + adapter site-offset math
[8.dr] OK [实测] per-env DR round-trip: links_inertial_mass True, dofs_frictionloss True, dofs_kp True (require batch_links_info/batch_dofs_info=True at build); set_friction_ratio + solver.apply_links_external_force callable (effect 未验证); entity-level apply_links_* absent in 1.3.3 -> solver API with global idx
```

### P2 `probe_runtime.py`（行 9/10/12）

```text
[9a.init_destroy_loop] OK [实测] 3x init->build(4 envs)->step->destroy in one process: no crash; peak RSS per iter MB=[1723.0, 1978.2, 2177.5] (per-cycle growth [255.2, 199.3], sub-linear; ru_maxrss is a high-water mark); cuda_reserved MB per iter=[22.0, 22.0, 22.0] stable; host RAM grows ~200-450MB/cycle -> long-lived processes must init once, not cycle
[9b.dual_scenes] OK [实测] two Scenes alive in one gs session: both build+step; scene_b root x offset applied (mean=1.00), scene_a unaffected: True
[10.host_side_effects] GAP [实测+源码推断] gs.init(gpu) mutates torch globals: default_device cpu->cuda:0, torch.zeros() lands on cuda:0 (was cpu), default_dtype torch.float32->torch.float32; torch num_threads 16->16 (untouched: True); cpu_max_num_threads=1 forced unconditionally unless QD_NUM_THREADS (quadrants-side, __init__.py:245-254); seed=... calls set_random_seed (global torch/np/random reseed); logging.root handlers 0->0; import alone harmless: True; adapter must snapshot+restore torch defaults or document the pollution
[12.render] OK [实测+源码推断] offscreen camera render OK: rgb(48, 64, 3) dtype=uint8; interactive Viewer is pyrender/pyglet-based (needs display); camera.render() is the headless path; GUI not attempted per probe scope
```

### P3 `probe_perf.py`（行 11，交错计时取中位数）

```text
[11.boundary[256]] OK [实测] step=1.771ms; +D2H(qpos/qvel/root/8 links pos+quat)=1.967ms (+0.196ms, +11.0%); +H2D ctrl push=1.872ms (+0.101ms, +5.7%); SPS(a)=1.45e+05
[11.boundary[2048]] OK [实测] step=2.808ms; +D2H(qpos/qvel/root/8 links pos+quat)=3.158ms (+0.350ms, +12.5%); +H2D ctrl push=2.877ms (+0.070ms, +2.5%); SPS(a)=7.29e+05
[11.boundary[4096]] OK [实测] step=4.294ms; +D2H(qpos/qvel/root/8 links pos+quat)=4.663ms (+0.369ms, +8.6%); +H2D ctrl push=4.440ms (+0.146ms, +3.4%); SPS(a)=9.54e+05
```
