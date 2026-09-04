# 扩展 UniLab：新算法

算法相关的工作必须保持 env、config 与 runner 契约。请从
{doc}`../2-contracts/1-env_contract`、{doc}`../2-contracts/3-task_owner` 与
{doc}`../2-contracts/5-runner_lifecycle` 开始。

## 三档扩展方式

按改动深度从低到高有三档；优先选择能满足需求的最浅一档。

### 1. 纯配置：复用现有算法

不改任何代码，只调整 `src/unilab/conf/<algo>/` 下的 Hydra 配置：

- 算法超参数内联在 `src/unilab/conf/<algo>/config.yaml`；
- 每个 task×backend 组合对应一个 owner YAML：
  `src/unilab/conf/<algo>/task/<task>/<backend>.yaml`；
- 替换策略 / 算法实现类可以走 owner YAML 中的 `class_name` dotted path
  （现网实例：`uni_rl.algos.hora:HoraActorModel`），无需新增代码路径。

### 2. `runtime_resolver`：算法代码放在自己的仓库

研究者在自己的仓库里 import `uni_rl`（distribution 名 `unilab-rl`）作为库，
实现 runner / learner / play 逻辑，**不需要 fork unilab_rl**。owner YAML
通过 `algo.runtime_resolver` 声明一个 dotted path（`module:attr`）：

```yaml
algo:
  runtime_resolver: my_pkg.my_module:resolve_my_runtime
```

约定：

- 签名为 `(rl_cfg: dict) -> Runtime | None`；返回 `None` 表示回落到该算法的
  默认 runtime；
- 返回对象必须携带 `runner_cls`；按算法族可选携带 `play_fn`（APPO 风格）
  或 `wrapper_cls`（PPO 风格）；
- 解析发生在 uni_rl 侧（`uni_rl.algos.appo.runtime` /
  `uni_rl.algos.rsl_rl_runtime`），dotted path 可以指向任何可 import 的模块。

仓库内实例：`src/unilab/conf/appo/task/sharpa_inhand/mujoco_hora.yaml` 指向
`unilab.scripts.play_hora_appo:resolve_hora_appo_runtime`；HORA 的 SAC / PPO
变体分别指向 `uni_rl.algos.hora.sac:resolve_hora_sac_runtime` 与
`uni_rl.algos.hora.rsl_rl:resolve_hora_ppo_runtime`。

### 3. fork unilab_rl：改 `uni_rl/algos/`

只有当需要改动 runner / learner / collector 的共享实现（例如新的 IPC
生命周期）时，才 fork [unilabsim/unilab_rl](https://github.com/unilabsim/unilab_rl)
并修改 `uni_rl/algos/`。异步算法应复用 `AsyncRunner`、`ReplayBuffer` /
`RolloutRingBuffer` 与 `SharedWeightSync`，而不是新建一套 IPC 生命周期。

## CLI 可路由算法的 footprint

`uv run train --algo <algo>` / `uv run eval --algo <algo>` 采用约定式路由
（`src/unilab/cli.py` 的 `available_algos` 与 `build_route`）：algo 名
`<algo>` 可路由，当且仅当以下两个文件同时存在，**不需要修改 cli.py**：

1. `src/unilab/conf/<algo>/config.yaml` —— Hydra config 根，算法超参数内联；
2. `src/unilab/scripts/train_<algo>.py` —— 入口脚本，保持为组装层薄壳：
   compose Hydra、调用 `ensure_registries()`、通过 registry 路径构造 env，
   然后把控制权交给 runner 或 trainer。薄壳先例：`train_sac.py` /
   `train_td3.py` / `train_flashsac.py` 复用 `train_offpolicy.py` 的共享实现。

除此之外，每个 task×backend 组合需要 owner YAML
`src/unilab/conf/<algo>/task/<task>/<backend>.yaml`。未知 algo 会
fail-closed，报错信息列出全部可用 algo（内置 + 约定发现的）。

注意：

- 只有 conf 目录而没有入口脚本的 config 树（如 `hora_distill`、`ppo_him`）
  不可路由——它们不是独立的 CLI algo。
- 内置算法的特殊脚本名映射保留不变：`ppo` → `train_rsl_rl.py`、
  `appo` → `train_appo.py`。
- `src/unilab/structured_configs.py` 的 dataclass 是**可选**的约定俗成镜像，
  没有 ConfigStore 强制注册；新增算法不强制要求添加对应 dataclass。

## 实现清单

1. 按上面三档选择集成路径，能纯配置就不写代码，能 `runtime_resolver`
   就不 fork。
2. 把第三方适配器命名保留在适配器边界上。不要为了迎合某个库而改动
   内部的 `obs` 加可选 `critic` 的 env 契约。
3. 新入口脚本保持为组装层，不承载长期业务规则。

## 在风险点附近验证

- CLI 路由与约定式发现：`tests/test_cli.py`
- 脚本/配置测试：`tests/scripts/test_train_script_configs.py`、
  `tests/scripts/test_train_scripts.py`

## 仓库内证据

- 结构化 config dataclass：`src/unilab/structured_configs.py`
- 训练辅助工具：`src/unilab/training/common.py`、
  `src/unilab/training/run.py`
- 现有算法包：[unilabsim/unilab_rl](https://github.com/unilabsim/unilab_rl)
  的 `uni_rl`
