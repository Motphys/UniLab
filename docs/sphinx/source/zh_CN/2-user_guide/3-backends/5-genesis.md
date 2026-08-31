# Genesis 后端

[Genesis](https://github.com/Genesis-Embodied-AI/Genesis)（PyPI 分发名
`genesis-world`，仓库钉在 1.3.3）是一个 GPU 物理仿真器，UniLab 以
**进程内**方式使用它：`GenesisBackend` 在其上提供标准的 `SimBackend`
NumPy contract，物理与 learner 同进程运行——没有 worker 子进程，也没有
IPC。

当前状态：`GenesisBackend` 已实现并注册到 registry；`g1_walk_flat` 提供
PPO owner 配置（`conf/ppo/task/g1_walk_flat/genesis.yaml`），跨后端契约
审计（`scripts/audit_sim2sim_contracts.py`）覆盖 mujoco↔genesis（结论
TRANSFERABLE）。支持等级为 **experimental**：现有证据是 registry +
owner YAML + compose/contract 覆盖。尚未完成任何训练验证，因此支持
矩阵中该 cell 标记为 `Configured`，且 playback/渲染不是已声明能力。

已知缺口（fail-closed，非静默）：当前在该后端上构造
`ManagerBasedRlEnv` 会抛出 `RuntimeError: genesis backend is not
materialized`——env 构造期间的 entity 校验在 env 的 `materialize()`
钩子之前读取状态 getter，而 genesis adapter 只在 materialize 之后才
提供这些读取（其 `materialize()` 也是一次性的，不像 IsaacGym 后端那样
幂等且惰性触发）。owner YAML、registry 条目与下文的 CLI 路由均已就位；
训练命令只会在 env 构造处失败，等待 adapter 生命周期的后续修复。
adapter 的其余设计遵循
`scripts/tools/genesis_feasibility/REPORT.md` 的实测映射。

## 模型契约

Genesis 1.3.3 在 import 时丢弃三类 MJCF 特性（REPORT §3）：全局
`<option>` 块、`<keyframe>`、以及整个 `<sensor>` 块。adapter 在
materialize 冷路径补偿——热路径从不解析 XML：

- **全局 option**：MJCF `<option>` 承载的值由 owner YAML 显式重声明。
  对 g1（`<option integrator="implicitfast"
  timestep="0.006666666666666667"/>`），timestep 经现有
  `sim_dt` / `ctrl_dt` 链路传递，integrator 经
  `env.genesis_integrator: implicitfast` 声明。constraint solver、摩擦锥
  与 solver iterations 保持 Genesis 默认值（对应 `env.genesis_*` 字段为
  `None`）：MJCF option 块未声明它们，且 MuJoCo 隐式默认（PGS solver、
  pyramidal 摩擦锥）没有 Genesis 等价物。
- **keyframe**：materialize 时用 `mujoco` 包扫描一次并缓存，
  `default_keyframe_name: stand` 的 reset 行为不变。
- **传感器**：由 link 状态计算 MJCF 同名等价物，每个 accelerometer
  site 对应一个 `IMUSensor`（干净无噪声数据）。`data="found"` 的
  contact sensor 变为 per-link net contact force 阈值（1 N）——这是对
  geom 对 `found` 语义的近似，不是复现。
- **执行器**：`<position kp kv>` 执行器无损导入，用
  `control_dofs_position` 驱动；kp/kd reset 随机化受支持（DR 能力集只
  声明 per-env 往返实测的项：body mass、base mass、kp、kd，以及
  interval body force）。

## 前置条件

- Linux x86_64，装有 NVIDIA GPU 与驱动。可行性探针只验证了 `gs.gpu`
  通道；CPU backend 不是已验证的支持通道。
- 仓库钉定的 torch 窗口（x86_64 上为 torch 2.8）：探针中
  IMUSensor + contact-force 传感器组合在 torch 2.7 下崩溃（REPORT
  §3.4/§8）。
- 安装可选 extra（精确钉定 `genesis-world==1.3.3`）：

```bash
uv sync --extra genesis
```

## 训练与评估

训练通过标准 CLI 选择 genesis owner（owner YAML 可正常 compose，
registry 路由到 genesis 后端；随后在 env 构造处按上文所述的
materialize 生命周期错误 fail-closed，直到 adapter 后续修复落地）：

```bash
# PPO
uv run train --algo ppo --task g1_walk_flat --sim genesis

# 小规模 smoke：64 个环境、只跑 3 个 iteration
uv run train --algo ppo --task g1_walk_flat --sim genesis \
    algo.num_envs=64 algo.max_iterations=3
```

Playback/渲染**不受支持**：owner 设置
`training.play_render_mode: none`，训练完成后安全跳过 playback。强制
其他模式（`--render-mode auto|interactive|record`）会 fail-closed，抛出
指明不支持该模式的 `NotImplementedError`——离屏 camera 渲染是已实测但
未集成的后续项（REPORT §3.5 [12]）。`uv run eval` 在默认 `none` 模式下
仍会加载 checkpoint（含 sim2sim preflight 与维度守卫），但不执行任何
渲染 rollout：

```bash
uv run eval --algo ppo --task g1_walk_flat --sim genesis --load-run <run_dir_name>
```

## 生命周期：一进程一次 `gs.init`

Genesis 的 init/destroy 循环每轮泄漏 200–450 MB host RSS（REPORT
§3.5 [9a]），因此 adapter 允许**每个进程恰好一次 `gs.init`**：session
销毁后再构造 backend 会 fail-closed 并给出明确错误。一个训练 run 一个
进程天然满足该约束；不要在长驻宿主进程里反复构建 genesis env。

## 未支持边界

以下能力 fail-closed（显式报错），而不是静默降级：

- **geom 名称契约**（`get_geom_names` 等）：不暴露。
- **体帧运动学**（`get_body_pos_b` / `get_body_quat_b`）。
- **生成式 terrain 与 height scanner**：`scene.terrain` 被拒绝；请选择
  flat owner YAML。
- **`none` 以外的 playback/render 模式**（见上文）。
- **绝对值 geom 摩擦 DR**：上游 Genesis 只有 per-env 摩擦*比例* API，
  因此 `geom_friction` reset 随机化与 `get_geom_friction` fail-closed。
- **interval push / body-velocity-delta DR**：在构造/plan 时拒绝
  （`push_body_name`、push perturbation）。
- **同进程重复 `gs.init`**（见上文）。

## 跨后端迁移（sim2sim）

genesis owner 与 mujoco owner 在契约守卫
（`src/unilab/utils/sim2sim.py`）审计下保持 DENYLIST 一致（结论
TRANSFERABLE），同一 task 的 checkpoint 可跨后端使用。注意 playback
不对称：在 MuJoCo 上播放 genesis 训练的 checkpoint 可正常渲染，而在
genesis 上播放任何 checkpoint 都只能用 `play_render_mode=none`。
