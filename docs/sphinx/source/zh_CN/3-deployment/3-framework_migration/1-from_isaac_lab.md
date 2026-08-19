# 从 Isaac Lab 迁移

把 Isaac Lab Manager-Based task 迁入 UniLab 时，应保留 manager 与 term 结构，只在各自
owner 边界适配配置、数值执行和场景访问；不要把 task 重写成单体 `NpEnv` 子类。

这是基于源码的兼容迁移，不代表任意 Isaac Lab task 都能不修改直接运行。目标路径是：

```text
Hydra owner YAML
  -> plain ManagerBasedRlEnvCfg
  -> Registry + make_manager_based_rl_env
  -> NumPy/SimBackend runtime 上的 ManagerBasedRlEnv
  -> 交给现有 training 和 IPC 路径的 NpEnvState
```

## 兼容边界

```{list-table}
:header-rows: 1
:widths: 28 24 48

* - Isaac Lab 表面
  - UniLab 状态
  - 迁移规则
* - Manager 类别、term 名称和字典顺序
  - Compatible
  - 保持 observation、action、event、reward、termination、command 与
    curriculum term 的顺序。
* - Function/class term 与 `func + params`
  - Compatible
  - 把 import 改为 `unilab.managers`；保留 term 边界和局部
    `reset(env_ids)` 语义。
* - `ManagerBasedRLEnv` / `ManagerBasedRLEnvCfg`
  - Compatible 拼写 alias
  - UniLab canonical 名称是 `ManagerBasedRlEnv` 与 `ManagerBasedRlEnvCfg`；alias
    指向同一份实现。
* - Tensor 数值与运算
  - Adapted
  - 把 `torch.Tensor` 换成 `np.ndarray` 并使用向量化 NumPy；manager-facing
    API 没有 device 接口。
* - 嵌套 `@configclass` task 配置
  - Adapted
  - 把完整 task 声明迁入唯一 Hydra owner YAML；用 `_target_` 选择具体 config
    dataclass，用 dotted `func` 选择 term。
* - `InteractiveSceneCfg`、USD 与 PhysX view
  - Adapted 或 Unsupported
  - 声明 task-owned `SceneCfg` 与 `EntityCfg`；状态和控制只通过
    `SceneEntityCfg` 与公共 entity facade 访问。不支持的能力在冷路径绑定时报错。
* - Omniverse、Isaac renderer 与 Torch/PhysX mutation
  - Unsupported
  - UniLab 不安装这些 runtime，也不提供静默模拟或回退。
```

规范边界见 {doc}`ADR-0006 </adr/ADR-0006-community-manager-api-on-numpy-runtime>`。
只有已经被 registry、配置和测试覆盖的表面才能声明为 Compatible。

## 迁移步骤

### 1. 盘点来源 task

固定 Isaac Lab revision，并列出来源 manager group、term 名称与顺序、参数、observation
维度、action 维度、reset 行为和 episode timing。写代码前逐项分类：

- 复用已有 `unilab.managers` config 或 `unilab.envs.mdp` term；
- 把 task-specific term 从 Torch 适配为 NumPy；
- 如果 term 依赖公共 entity 或 `SimBackend` contract 尚未提供的能力，立即停止。

不能用 `getattr`/`hasattr` 探测 backend 对象、返回零，或把 task 路由回 legacy env。

### 2. 在冷路径迁移 scene 与 asset

用 task-owned `SceneCfg` 代替 Isaac Lab 的 USD/`InteractiveSceneCfg` 声明，显式声明 term
需要的每个 entity 与 selector。`SceneEntityCfg` 在 materialization 时只解析一次名称和
正则表达式；reset/step 复用缓存 ID 与 NumPy view。

Cartpole fixture 使用最小 task-owned MJCF。更复杂的 asset 必须遵守
{doc}`场景组合 <../../4-developer_guide/1-architecture/4-scene_composition>`，并只使用所选
backend 的正式能力。

### 3. 迁移 term 代码，不改 manager 结构

保留每个 function/class term 及其参数，机械地把 Torch 类型与运算改为 NumPy，保持 batch
shape，并在来源 term 返回每环境数值时继续返回每环境数值。Stateful term 在构造时解析
selector、分配 buffer，热路径只更新 NumPy buffer。

Python 只拥有 term 实现和可复用 config dataclass，不能再保存一份 task-specific term
启停清单或默认 weight。

### 4. 让 Hydra 成为唯一 task 配置 owner

在 owner YAML 中声明 scene、timing、group、term、具体 config 类型、callable、参数、
weight 和 observation mapping。例如：

```yaml
env:
  observations:
    policy:
      _target_: unilab.managers.ObservationGroupCfg
      terms:
        joint_pos_rel:
          _target_: unilab.managers.ObservationTermCfg
          func: unilab.envs.mdp.joint_pos_rel
  terminations:
    time_out:
      _target_: unilab.managers.TerminationTermCfg
      func: unilab.envs.mdp.time_out
      time_out: true
  policy_observation_group: policy
  critic_observation_group: null

reward:
  alive:
    _target_: unilab.managers.RewardTermCfg
    func: unilab.envs.mdp.is_alive
    weight: 1.0
```

Hydra compose 在冷路径把这份声明物化为 plain typed config。未知字段、无法解析的
`_target_`/`func` 和错误 config 类型都会在 reset/step 之前报错。直接用 Python 构造
config 只用于 focused 底层测试。

### 5. 只注册一条通用 runtime 路径

Task module 为仓库已经实际支持的每个 backend 注册 `ManagerBasedRlEnvCfg` 与
`make_manager_based_rl_env`。Backend owner YAML 只承载 backend 身份与 tuning。用户通过
标准 CLI 选择 compose owner，例如：

```bash
uv run train --algo ppo --task <task> --sim mujoco
```

不要增加 task-specific 训练脚本分支、env factory、runner 或 IPC 路径。

### 6. 在适配风险附近验证

测试 Hydra compose 与 typed materialization、term 顺序与数学、selector 失败、
observation/action shape、局部 reset，以及至少一个真实已注册 backend 的 transition。行为
应与固定来源 task 对比；完成语义迁移后再做性能 benchmark。

## 仓库证据

`tests/fixtures/isaac_lab_cartpole/` 迁移了 Isaac Lab commit
`b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8` 的 Manager-Based Cartpole task。它保留
全部 12 个来源 term 的名称和顺序，同时把 Torch 适配为 NumPy、嵌套 config object 适配
为 Hydra YAML，并用 fixture-local MJCF 实现 scene/action/reset 边界。这只是 test-only
证据，不是 production task 或 Isaac Lab 全量支持声明。

## 另请参阅

- {doc}`Manager-Based API <../../4-developer_guide/1-architecture/6-manager_based_api>`
- {doc}`Env contract <../../4-developer_guide/2-contracts/1-env_contract>`
- {doc}`ADR-0006 </adr/ADR-0006-community-manager-api-on-numpy-runtime>`
