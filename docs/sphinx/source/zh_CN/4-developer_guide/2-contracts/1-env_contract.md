# Env 契约

Env 契约由 `src/unilab/base/base.py` 与 `src/unilab/base/np_env.py` 在代码层
拥有。观测语义记录在
{doc}`/adr/ADR-0005-unified-obs-critic-env-and-ipc-contract`。

## 必需形状

- `NpEnvState.obs` 是 `dict[str, np.ndarray]`，而不是扁平的 tensor。
- 必需的 actor 观测 key 是 `obs`。
- 唯一可选的、仅供 critic 使用的观测 key 是 `critic`。
- `obs_groups_spec` 将每个观测组名映射到其扁平维度。Wrapper 与 learner 利用该
  映射来确定 actor 与 critic 路径的尺寸。
- `reset(env_indices)` 为被 reset 的 env 行返回 `(obs_dict, info_dict)`。
- 在 `NpEnv` 上调用 `step(actions)` 返回 `NpEnvState`；外部适配器可以在适配边界
  处将该状态映射为第三方 trainer 的 API。

## Owner 职责

- env 代码拥有 MDP 语义、观测构造、奖励、termination、truncation、reset 行为
  以及 final-observation 处理。
- runner 与 learner 不得在 env owner 层之外通过拼接字段来自行构造 critic 观测。
- 如果某个第三方库仍将 critic 观测称为 "privileged"，请把该名称转换保留在适配器
  内部。在 UniLab 内部，该 key 始终是 `critic`。

## 仓库中的证据

- Env base 契约：`src/unilab/base/base.py`
- Numpy env 状态：`src/unilab/base/np_env.py`
- RSL-RL 适配边界：`src/unilab/training/rsl_rl.py`
- Final observation helper：`src/unilab/base/final_observation.py`
- 测试：`tests/base/test_np_env.py`、`tests/utils/test_final_observation.py`、
  `tests/ipc/`
