# Env 契约

Env 契约由 `src/unilab/base/base.py` 与 `src/unilab/base/torch_env.py`
在代码层拥有。观测语义
记录在 {doc}`/adr/ADR-0005-unified-obs-critic-env-and-ipc-contract`，并被
{doc}`/adr/ADR-0011-torch-only-manager-based-runtime` 取代。

## 必需形状

- `TorchEnvState.obs` 是 `dict[str, torch.Tensor]`，而不是扁平的 tensor。
- 必需的 actor 观测 key 是 `obs`。
- 唯一可选的、仅供 critic 使用的观测 key 是 `critic`。
- `obs_groups_spec` 将每个观测组名映射到其扁平维度。Wrapper 与 learner 利用该
  映射来确定 actor 与 critic 路径的尺寸。
- `reset(env_indices)` 接受一维 Torch integer tensor 或 `None`，并为被 reset
  的 env 行返回 `(obs_dict, info_dict)`。
- 在 `TorchEnv` 上调用 `step(actions)` 接受 contiguous float32 Torch tensor，
  并返回 `TorchEnvState`；外部适配器只能在适配边界做 trainer device transfer。
- 不存在 legacy 环境 fallback。不支持的 device、carrier 和 backend tensor
  lifecycle 一律 fail closed。

## Owner 职责

- env 代码拥有 MDP 语义、观测构造、奖励、termination、truncation、reset 行为
  以及 final-observation 处理。
- 异步 backend 发布 `CompletionToken` 时，`TorchEnv` 通过这个公开 receipt
  在下一次 state-feedback Manager action/状态读取/控制写入之前，以及最终
  Manager state update 之前，为当前 Torch stream 建立顺序。feed-forward
  仍保持一次批量 backend 调用；无 receipt 的同步 producer 不执行 read gate。
  任务代码不得为建立该顺序覆写 physics stepping，也不得使用 backend 私有
  stream/event。
- runner 与 learner 不得在 env owner 层之外通过拼接字段来自行构造 critic 观测。
- 如果某个第三方库仍将 critic 观测称为 "privileged"，请把该名称转换保留在适配器
  内部。在 UniLab 内部，该 key 始终是 `critic`。

## 仓库中的证据

- Env base 契约：`src/unilab/base/base.py`
- Torch env 生命周期：`src/unilab/base/torch_env.py`
- RSL-RL 适配边界：`src/unilab/rl/vec_env.py`
- Final observation helper：`src/unilab/base/final_observation.py`
- 测试：`tests/base/test_torch_env.py`、
  `tests/envs/test_manager_based_rl_env.py`、
  `tests/utils/test_final_observation.py`、`tests/ipc/`
