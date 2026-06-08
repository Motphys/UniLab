# Sim2Sim 跨后端配置契约梳理（#579 P1）

本页是 issue #579 的逐任务契约梳理结果：对每个同时拥有多个后端 owner YAML 的 task，
判断其 MuJoCo 与 Motrix 在 **DENYLIST 契约字段**（影响训练策略 I/O / 网络结构的字段）上
是否一致，从而回答「这个 task 当前能否 A 后端训练 → B 后端 play（sim2sim 迁移）」。

契约字段的单一事实源在 `src/unilab/training/sim2sim.py`；运行时守卫
`resolve_sim2sim_config` 在 play 建 env 前校验，详见 `AGENTS.md` 的 Sim2Sim 章节。

## 方法

梳理用 `scripts/audit_sim2sim_contracts.py` 产出（read-only，不改任何 config / checkpoint）：

- 对每个 `conf/<tree>/task/<task>/` 下的每个后端 YAML，用 hydra `compose` 得到**有效配置**
  （已展开 `defaults` 继承、`base.yaml`、`# @package _global_`），再用守卫同一套归一化逐字段比对。
- 复跑：`uv run scripts/audit_sim2sim_contracts.py`（默认扫 `ppo appo` 两棵树；`--trees` 可加扫，`--json` 出机器可读）。

**一个必须理解的口径**：脚本比对的是 **composed YAML**，此时「某后端没写该字段」显示为 `<absent>`。
但运行时 env dataclass 会给该路径填默认值。因此 `asymmetric-presence`（一边写、一边缺）这种分歧，
对 env 结构字段（`action_scale` / `sampling_mode`）是真实运行时差异。**P2 硬化后**（见结论 1）守卫对这类
env 结构字段的不对称出现已 **fail-closed 报错**；algo 专属字段（`empirical_normalization` /
`obs_normalization`）仍按设计在目标缺省时跳过（跨算法合法）。本页表格的「运行时真相 / 守卫是否实际拦截」
两列即在脚本结果基础上，结合 env 代码默认值复核后的结论。

判定分三档，比单纯「一致 / 发散」更准：

- ✅ **可迁移**：DENYLIST 在 composed config 上无分歧。
- ❌ **阻断（守卫拦得住）**：存在 value-diff（两边都写、值不同），或 env 结构字段不对称出现，守卫会抛 `CrossBackendIncompatibleError`。
- 🔴 **真不兼容但守卫静默**：DENYLIST 分歧曾全是 env 结构 asymmetric-presence 而被跳过（false-negative）；**P2 已修复，本档目前为空**（保留作历史记录）。

## 表一：`conf/ppo/task/`（19 对 mujoco↔motrix）

| Task | 判定 | DENYLIST 分歧（运行时真相） | 守卫实际拦？ |
|---|---|---|---|
| allegro_inhand · allegro_inhand_grasp · g1_climb_tracking · g1_motion_tracking · g1_wall_flip_tracking · go1_joystick_rough · go2_arm_manip_loco · go2_handstand · go2_joystick_flat · go2_joystick_rough · go2w_joystick_flat · go2w_joystick_rough · sharpa_inhand · sharpa_inhand_grasp（14 个） | ✅ 可迁移 | 无 | —（无需） |
| g1_box_tracking | ❌ 阻断 | `empirical_normalization` false↔true（建模）；`obs_groups` 仅 critic 组差异（装饰性，可收敛） | ✅ |
| g1_flip_tracking | ❌ 阻断 | `empirical_normalization` true↔false + `obs_groups`（value-diff）；`action_scale` 29 维 list↔默认标量 0.25（asymmetric，P2 后拦）；`sampling_mode` 运行时两边都 = `start`（无害） | ✅（emp_norm / obs_groups；P2 后 action_scale 也拦） |
| g1_walk_flat（试点） | ❌ 阻断 | `action_scale` 0.25↔0.5、`empirical_normalization` false↔true（均真发散）；`obs_groups` 对部署 actor 实为装饰性 | ✅ |
| go1_joystick_flat | ❌ 阻断 | `empirical_normalization` false↔true（建模） | ✅ |
| g1_motion_tracking_deploy | ❌ 阻断 | 仅 `action_scale` 29 维 list ↔ motrix 缺省→标量 0.25（asymmetric） | ✅（P2：env 结构 asymmetric fail-closed） |
| go2_footstand | ⚪ N/A | 仅 mujoco，无后端对 | — |

## 表二：`conf/appo/task/`（6 对）

| Task | 判定 | DENYLIST 分歧（运行时真相） | 守卫实际拦？ |
|---|---|---|---|
| allegro_inhand · g1_climb_tracking · g1_motion_tracking · go2_joystick_flat（4 个） | ✅ 可迁移 | 无 | — |
| g1_flip_tracking | ❌ 阻断 | `action_scale` 29 维↔默认 0.25（asymmetric，真）；`sampling_mode` 运行时两边都 = `start`（无害） | ✅（P2：env 结构 asymmetric fail-closed） |
| g1_wall_flip_tracking | ❌ 阻断 | `action_scale` 29 维↔默认 0.25（真）；`sampling_mode` `start`↔默认 `adaptive`（真） | ✅（P2） |

> `g1_walk_flat` / `go1_joystick_flat` 在 appo 树是单后端（仅 mujoco）→ N/A。

## 其它配置树

`conf/ppo_him/task`（仅 `go2_arm_manip_loco` mujoco）、`conf/offpolicy/task`（algo 占位，无 backend YAML）、
`conf/hora_distill/task`（`sharpa_inhand` 的 mujoco / mujoco_nodr 同后端蒸馏变体）均**无 mujoco↔motrix 对**，
sim2sim 不适用。

## 关键结论

### 1. asymmetric-presence 盲区（曾是 false-negative，P2 已修复）

**历史问题**：`resolve_sim2sim_config` 旧逻辑 `if target_value is None: continue`，加上
`extract_contract_snapshot` 跳过 None 源值，导致「源后端设了、目标后端 YAML 没设（运行时用 dataclass 默认）」
的 DENYLIST 字段**从不被比较**；且守卫跑在 hydra-composed cfg 上、早于 env dataclass 默认注入。
`action_scale` 默认是标量（如 `src/unilab/envs/locomotion/g1/base.py` 中 ControlConfig 的 `0.25`），
而相关 mujoco YAML 用 29 维 per-joint 向量，二者在 `ctrl = action*action_scale + base` 处真实不等
→ 策略被静默污染、守卫不报错。曾有 3 个 task 级 false-negative：`ppo/g1_motion_tracking_deploy`、
`appo/g1_flip_tracking`、`appo/g1_wall_flip_tracking`。

**P2 修复**：定义 `ENV_STRUCTURAL_DENYLIST = [DENYLIST 中以 `env.` 开头者]`
（= `env.control_config.action_scale`、`env.sampling_mode`）。对这些 env 结构字段，**任一方向**的
不对称出现（源有目标缺 / 源缺目标有）都 **fail-closed 报错**，要求目标 YAML 显式声明该字段才能验证。
这是**故意的 field-aware**：`target None → skip` 仍保留给合法的跨算法字段（PPO `empirical_normalization`
vs off-policy `obs_normalization`，见
`tests/training/test_sim2sim_resolver.py::test_target_missing_path_is_skipped`），不能一刀切「目标缺即报错」。
fail-closed 取向意味着即使两边运行时默认恰好相等也会报错（补 1 行显式声明即解除）；这符合 #579
「静默污染 → 显式报错」的理念。运行 `scripts/audit_sim2sim_contracts.py` 现报告 guard-blind-spot 字段为 0。

### 1b. 运行时维度最后防线（play 侧、零训练改动）

配置级 guard 只查 YAML 代理;真正决定策略 I/O 维度的是 `env.obs_groups_spec` / action space（运行时算出）。
为兜住 YAML 看不到的真实维度不匹配,五个 play 入口加载 checkpoint 时用 `policy_load_dim_guard`
（`src/unilab/training/sim2sim.py`）包裹:torch `load_state_dict` 对张量 size mismatch **无论 strict 与否都会报错**、
mlx `load_weights(strict=True)` 同理 —— guard 把这类隐晦的 size-mismatch 错误**重抛为显式 sim2sim 维度诊断**
(指明 env 的 obs/action 维度 + 指向本审计脚本)。它**只在加载本就失败时触发**,因此绝不会阻断正常 play,
也不碰训练链路。完整的逐算法 obs 维度 assert(含 HORA-SAC priv 拼接、state-dependent std 输出翻倍等)
留待能真跑 play 的 GPU 冒烟阶段验证。

### 1c. 用户级绕过开关(应对守卫过度阻断)

配置级守卫对**任何** DENYLIST 值差异都报错,包括"装饰性" `obs_groups` 差异(只差 play 不用的 critic 组)。
为避免误拦一个已知兼容的跨后端组合,新增 `training.sim2sim_strict`(默认 `true`,零行为变化):
设 `training.sim2sim_strict=false` 时,`resolve_sim2sim_config(strict=False)` 把 DENYLIST 差异**降级为 warning 并继续**,
此时 load 时的 `policy_load_dim_guard` 仍兜底真正的维度不匹配。API 同时以 `Sim2SimConfigResolver` 类
(RFC 命名)与同名模块函数双形态提供。

### 2. `empirical_normalization` 是真「建模」分歧

它把 running mean/std normalizer 作为 nn.Module 子模块烘进 actor，并将统计 buffer 存入 checkpoint
（`src/unilab/training/rsl_rl.py` 把该 flag 注入 actor/critic 的 `obs_normalization`）。ON 训练的策略
OFF 加载就是把未归一化输入喂给期望归一化输入的权重 → 崩坏；且 state_dict key 不匹配。
因此**不能只改 YAML 收敛**，统一必须重训。Motrix 多数 task 倾向开启它（`g1_flip_tracking` 方向相反）。

### 3. `obs_groups` 多为装饰性 / critic-only

相关 env 只产出 `{obs, critic}` 组，`actor:[actor]` 与 `actor:[policy]` 实际喂给 actor 同一张量；
差异往往只在 critic 组（play 不加载 critic）。这类分歧可安全收敛进 `base.yaml`（统一拼写、不动部署 actor）。

### 4. 「让某 task 可迁移」是 owner 决策，不是配置搬运

能「零成本」统一的只有无害项（`obs_groups` 拼写、appo flip 的 `sampling_mode=start`）。
`action_scale` / `empirical_normalization` 的统一都会改变某后端的训练并需重训，必须由 task owner 拍板，
不能擅自改训练值。

## owner 建议（逐 BLOCKED task）

| Task | 可零成本收敛的字段 | 需 owner 决策（改训练 / 重训）的字段 |
|---|---|---|
| ppo/g1_box_tracking | `obs_groups`（抽进 base.yaml 统一拼写） | `empirical_normalization` |
| ppo/g1_flip_tracking | `sampling_mode`（两边运行时已是 start，可显式写齐） | `empirical_normalization`、`obs_groups`、`action_scale` |
| ppo/g1_walk_flat（试点） | `obs_groups`（装饰性） | `action_scale`(0.25↔0.5)、`empirical_normalization` |
| ppo/go1_joystick_flat | — | `empirical_normalization` |
| ppo/g1_motion_tracking_deploy | — | `action_scale`（守卫已 P2 硬化，见结论 1） |
| appo/g1_flip_tracking | `sampling_mode`（运行时已都是 start） | `action_scale`（守卫已 P2 硬化） |
| appo/g1_wall_flip_tracking | — | `action_scale`、`sampling_mode`(start↔adaptive)（守卫已 P2 硬化） |

## 复跑方式

```
uv run scripts/audit_sim2sim_contracts.py
```

configs 演进后随时重扫即可刷新本表；脚本会额外列出「含守卫漏拦字段」的 task 供复核
（P2 后 env 结构 asymmetric 已 enforced，仅 algo 专属字段的跨算法缺省会被标为 blind-spot）。
