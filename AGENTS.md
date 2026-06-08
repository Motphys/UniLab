# UniLab Agent Principles

**Always use `uv run`, not python**.

UniLab 是一个 **高性能、模块化、contract 驱动** 的 RL infrastructure 仓库。

## Core Principles

1. **Contract first**: 不为了一次通过绕过 env / backend / runner contract。
2. **Fix at owner layer**: `scripts/` 只组装流程，不承载长期业务规则。
3. **Config first**: task / reward / backend 优先通过 Hydra + registry 表达。
4. **Backend isolation**: MuJoCo / Motrix 差异留在 backend 适配层和配置层。
5. **Evidence only**: support claim 只写仓库里已有的注册、配置、测试或 benchmark 事实。
6. **Validate near risk**: 在最接近风险的边界补验证，不只跑顶层命令。
7. **Cold-path asset access only**: asset/XML/model metadata 只允许在 init / materialization / cache 等低频路径处理；热路径不能解析 asset，也不能靠 `getattr` / `hasattr` 探测 backend 私有能力。

## High-Risk Areas

| 区域 | 不可破坏的不变量 |
|------|----------------|
| Env  | `NpEnvState.obs` 必须是 dict；`reset()` 返回 `(obs_dict, info_dict)`；`obs_groups_spec` 影响 wrapper 和 learner 维度。 |
| Config / Reward | reward 通过 Hydra 注入；后端切换必须通过 `task=<task>/<backend>` 选择 owner YAML，`training.sim_backend` 只是 owner YAML 的身份字段，不能单独 override 来切后端。算法超参数直接走 YAML compose，不经 Python 层解释。 |
| Backend | backend-specific 逻辑留在 backend / env 适配层，不向训练脚本扩散。env 层只能调用 `SimBackend`（`base.py`）中已声明的方法；若某方法只在 MuJoCo 或 Motrix 中存在，必须先将其加入 `SimBackend` 抽象接口（可抛 `NotImplementedError`），禁止直接在 env 里调用 backend 子类的私有方法（即"功能泄漏/feature leakage"）。新增 backend 专有能力时，需同步更新 `SimBackend`。 |
| Asset / Metadata | `ASSETS_ROOT_PATH`、`model_file`、XML / asset 元数据只允许在 init / materialization / cache 等低频路径访问；`step/reset/domain randomization` 等热路径不得解析 asset 或基于 asset 元数据做运行时分支。 |
| Asset / XML structure | `<keyframe>` 必须放在 task-level XML（`scene_*.xml` 或 `locomotion_task.xml` 等 fragment），**禁止放进 robot.xml**。robot.xml 是纯机器人描述（body / joint / actuator / sensor），跟 task / 场景无关；keyframe 是 task 起始姿态，属于场景或 task 资源。motrix 后端需要 keyframe 时通过 `scene.fragment_files` 引用 fragment XML。 |
| Async | 不绕开 runner lifecycle，也不另起 collector / learner 同步协议。 |
| Sim2Sim 契约 | 跨后端 play（A 后端训练 → B 后端 play）要**可迁移**，则影响策略 I/O / 网络结构的字段必须跨后端一致，应放进 task 的 `base.yaml` 共享。backend YAML 若为单后端调参 override 了某契约字段，则该 task 在该后端**不可 sim2sim 迁移**：训练时写入 `run_config.json` 的 `contract_snapshot`，play 时 `resolve_sim2sim_config` 在建 env 前校验，差异即 `CrossBackendIncompatibleError`（把静默污染变成显式报错）。详见下方 Sim2Sim 章节。 |

## Sim2Sim 跨后端配置契约（#579）

A 后端训练的 checkpoint 在 B 后端 play 时，env 用**目标后端 YAML** 创建；为避免与训练配置静默不一致而污染策略，`src/unilab/training/sim2sim.py` 定义三类字段（按 dotted path 维护，单一事实源）：

- **DENYLIST**（差异即 `CrossBackendIncompatibleError`）：`algo.obs_groups`、`env.control_config.action_scale`、`algo.policy.actor_hidden_dims` / `critic_hidden_dims`、`algo.empirical_normalization`（off-policy 为 `algo.obs_normalization`）、`env.sampling_mode`。即改变策略 I/O 或网络结构的字段。其中 `env.*` 子集（`ENV_STRUCTURAL_DENYLIST` = `action_scale`、`sampling_mode`）由 env dataclass 默认兜底、composed config 里常缺省，故**任一方向的不对称出现（源有目标缺 / 源缺目标有）也 fail-closed 报错**（#579 P2），要求目标 YAML 显式声明才能验证；`algo` 专属字段（`empirical_normalization` / `obs_normalization`）目标缺省时仍按设计跳过（跨算法合法）。
- **WARNING_LIST**（允许覆盖、打 warning）：`reward.*`（play 不用 reward）、`env.control_config.simulate_action_latency`、`env.ctrl_dt`。
- **ALLOWLIST**（目标后端自由覆盖、不快照不比较）：`training.sim_backend`、`env.scene`、`training.play_steps`、`env.domain_rand`、`env.noise_config`、`env.commands.vel_limit`。

机制：训练时 `ExperimentTracker.start()` 把 `DENYLIST + WARNING_LIST` 字段写进 `run_config.json` 的 `contract_snapshot`（**不改 checkpoint 格式**，天然兼容历史 checkpoint）；五个 play 入口（rsl-rl / him-ppo / appo / offpolicy / mlx）在建 env 前调用 `resolve_sim2sim_config(source_run_dir, cfg)` 校验。旧 run（无 snapshot）fallback + warning、不中断。此外，五个 play 入口加载 checkpoint 时用 `policy_load_dim_guard` 包裹（运行时维度最后防线）：env 实际 obs/action 维度与 checkpoint 张量形状不符（YAML 级 guard 看不到的真实维度不匹配）时，把隐晦的 size-mismatch 重抛为显式 sim2sim 维度诊断；只在加载本就失败时触发，不影响正常 play，零训练侧改动。

API 以 `Sim2SimConfigResolver` 类（RFC 命名）+ 同名模块级函数双形态提供（清单常量仍是单一事实源）。**用户级绕过**：要强行跨后端 play 一个已知兼容的组合，设 `training.sim2sim_strict=false`——DENYLIST 差异降级为 warning 并继续（默认 `true`，零行为变化；load 时的维度 guard 仍兜底真实维度不匹配）。

DENYLIST 字段应通过 task 的 `base.yaml` 作为**跨后端默认契约**（范例：`conf/ppo/task/g1_walk_flat/{base,mujoco,motrix}.yaml`）：mujoco 直接继承 `base`；motrix 出于单后端调参**显式 override** 了部分契约字段，因此 `g1_walk_flat` 当前不可 mujoco↔motrix sim2sim（guard 会按设计报错），去掉这些 override 即可恢复可迁移。

## Pointers

- PPO: `scripts/train_rsl_rl.py`
- MLX PPO: `scripts/train_mlx_ppo.py`
- APPO: `scripts/train_appo.py`
- SAC / TD3: `scripts/train_offpolicy.py`
- env contract: `src/unilab/base/np_env.py`
- backend contract: `src/unilab/base/backend/base.py`
- training run helpers: `src/unilab/training/run.py`
- visualization helpers: `src/unilab/visualization/`
- env shared numeric helpers: `src/unilab/envs/common/rotation.py`, `src/unilab/envs/common/math.py`
- MLX rotation helpers: `src/unilab/algos/mlx/common/rotation.py`
- config schema: `src/unilab/structured_configs.py`
- async runner: `src/unilab/ipc/async_runner.py`
- sim2sim 跨后端契约: `src/unilab/training/sim2sim.py`

## GitHub CLI (gh) 速查

### Issue 查看
```bash
gh issue view <number>
gh api repos/<owner>/<repo>/issues/<number> --jq '.body'
```

### PR 创建与管理
```bash
gh pr create --title "标题" --body "内容" --base main
gh pr list
gh pr view
```

### PR Gate

创建或更新 PR 前必须满足：

1. 最终提交已经完成，且 `git status --short --branch` 确认工作树干净。
2. 最终提交已经通过 `make test-all`。
3. 如果用户明确说明已经跑过 `make test-all`，不要重复跑；但必须在 PR body 的 Validation 里记录 `make test-all` 已完成。
4. 如果 `make test-all` 未通过且用户没有明确 override，不要创建或更新 PR。

### CI 工作流查看
```bash
gh run list
gh run list --workflow=<workflow-name>
gh run view <run-id>
gh run list --status=failure
```

### 常用组合
```bash
gh api repos/unilabsim/UniLab/issues/174 --jq '.title, .body'
git push -u origin fix/issue-174-mlx-ppo-config-alignment
gh pr create --title "fix: xxx" --body "Fixes #174" --base main
```

## Context

- 架构标准与验证详情：[docs/sphinx/source/zh_CN/4-developer_guide/0-index.md](docs/sphinx/source/zh_CN/4-developer_guide/0-index.md)
- 协作流程与 PR 规范：[docs/sphinx/source/zh_CN/4-developer_guide/5-contributing_workflow.md](docs/sphinx/source/zh_CN/4-developer_guide/5-contributing_workflow.md)
- 开发者入口（环境、命令、提交规范）：[CONTRIBUTING.md](CONTRIBUTING.md)
- 文档本地构建与发布到 UniLab-doc：[docs/sphinx/README.md#本地发布到-unilab-doc](docs/sphinx/README.md#本地发布到-unilab-doc)
