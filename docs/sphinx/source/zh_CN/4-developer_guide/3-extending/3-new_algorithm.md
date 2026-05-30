# 扩展 UniLab：新算法

语言: 简体中文

算法相关的工作必须保持 env、config 与 runner 契约。请从
{doc}`../2-contracts/1-env_contract`、{doc}`../2-contracts/3-task_owner` 与
{doc}`../2-contracts/5-runner_lifecycle` 开始。

## 选择集成路径

- 同步 on-policy 示例：`scripts/train_rsl_rl.py` 与
  `scripts/train_mlx_ppo.py`。
- 异步 on-policy 示例：使用 `APPORunner` 的 `scripts/train_appo.py`。
- Off-policy 示例：`scripts/train_offpolicy.py`，配合 `conf/offpolicy/`
  下的 SAC、TD3 与 FlashSAC 配置。

## 实现清单

1. 把可复用的 learner 或 runner 代码放在 `src/unilab/algos/` 下。
2. 在归属的 config 根目录下添加 Hydra config。一个新的 off-policy 变体
   应当添加 `conf/offpolicy/algo/<algo>.yaml` 以及对应的
   `conf/offpolicy/task/<algo>/<task>/<backend>.yaml` owner YAML。
3. 如果需要新的顶层训练脚本，请让它保持为组装层：
   compose Hydra、调用 `ensure_registries()`、通过 registry 路径构造 env，
   然后把控制权交给 runner 或 trainer。
4. 把第三方适配器命名保留在适配器边界上。不要为了迎合某个库而改动
   内部的 `obs` 加可选 `critic` 的 env 契约。
5. 对于异步算法，复用 `AsyncRunner`、`ReplayBuffer` 或
   `RolloutRingBuffer` 以及 `SharedWeightSync`，而不是新建一套 IPC
   生命周期。
6. 对于 off-policy 算法，保持 CLI 的 `--algo <algo>` 选择与 owner YAML
   路径 `conf/offpolicy/task/<algo>/<task>/<backend>.yaml` 对齐；
   `assert_offpolicy_task_choice_matches_algo` 会强制这一约束。

## 在风险点附近验证

- `tests/algos/` 下的算法单元测试
- 异步路径的 IPC 测试，位于 `tests/ipc/`
- 脚本/配置测试：`tests/scripts/test_train_script_configs.py`、
  `tests/scripts/test_train_scripts.py`

## 仓库内证据

- 结构化 config dataclass：`src/unilab/structured_configs.py`
- 训练辅助工具：`src/unilab/training/common.py`、
  `src/unilab/training/run.py`
- 现有算法包：`src/unilab/algos/torch/`、
  `src/unilab/algos/mlx/`

## Navigation

- Index: [文档](0-index.md)
