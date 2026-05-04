# ADR-0005 Unified Obs Critic Env And IPC Contract

语言: 简体中文

- Status: Accepted
- Date: 2026-04-17
- Owners: Env / IPC maintainers
- Supersedes: None
- Superseded by: None

## Context

UniLab 已经收敛到 `NpEnvState.obs: dict[str, np.ndarray]`，但历史实现里仍混杂过多种观测语义：

- actor 路径消费 `obs`
- 部分 env 额外暴露所谓 `privileged`
- 部分 off-policy / asymmetric critic 路径又单独引入 `critic`

这会直接破坏 contract-first：

1. 同一个 env 在不同 learner / runner / IPC 路径里，critic 输入语义不一致。
2. actor 路径可能意外吃到 critic-only 特征。
3. 某些链路需要靠脚本层或 learner 层猜测如何把 actor obs、extra obs 重新拼成 critic 输入。

这些都不是 owner layer 的职责，也不是可接受的顶层设计。

## Decision

UniLab 运行时 observation contract 统一为且仅为两层：

1. Env observation dict
   - `obs` 是唯一必选键，表示 actor observation。
   - `critic` 是唯一可选额外键，表示 critic observation。
   - 运行时不再承认 `privileged` 是 observation contract 的一部分。

2. Actor contract
   - actor 只消费 `obs`。
   - 所谓 flat actor policy 也只等价于 `obs`；不能拼接 `critic`，更不能从别处推导 critic-only 信息。

3. Critic contract
   - critic 若存在 asymmetric 输入，则由 env owner layer 直接产出完整 `critic`。
   - critic 若不存在单独输入，则显式回退到 `obs`。
   - 不允许在 runner、learner、script、IPC 层做隐式拼接、补齐、猜测或兼容转换。

4. IPC contract
   - on-policy 与 off-policy 统一只传递：
     - `obs`
     - 可选 `critic`
     - `next_obs`
     - 可选 `next_critic`
     - `last_obs`
     - 可选 `last_critic`
   - terminal patching / timeout bootstrap 若 env 提供 `critic`，必须原样保留并传到 critic 路径。

5. Adapter boundary
   - 某些第三方库若仍使用历史字段名，例如 RSL-RL 的 `num_privileged_obs` / `get_privileged_observations()`，只允许在 adapter boundary 做名字映射。
   - 该映射不改变 UniLab 内部 contract；其语义必须仍然是 `critic`。

## Consequences

- asymmetric actor-critic 的职责被固定在 env owner layer，不再向 learner / script 扩散。
- APPO、off-policy replay、terminal bootstrap、RSL wrapper 使用同一套 `obs` + optional `critic` 语义。
- 新 env 若需要 critic-only 信息，必须在 `obs_groups_spec` 和 `_compute_obs()` 中直接声明 `critic`，而不是新增第三种历史键名。

## Alternatives Considered

- 继续保留 `privileged`、`critic`、拼接 flat obs 等多套 observation 语义。拒绝原因：runner / learner / IPC 会继续猜测 critic 输入来源，actor 路径也可能误吃 critic-only 特征。
- 在每个算法 adapter 内单独兼容 critic 输入。拒绝原因：会把 env owner layer 的职责扩散到 learner 和 script。

## Non-Goals

- 不要求所有 env 都必须提供 `critic`。
- 不要求第三方库同步改名；外部接口名兼容可以留在 adapter boundary。
- 不把 `training.sim_backend`、脚本参数或 checkpoint 兼容逻辑混入 observation contract。

## Evidence In Repo

- Env contract: `src/unilab/base/np_env.py`
- Final observation helper: `src/unilab/base/final_observation.py`
- RSL-RL adapter: `src/unilab/training/rsl_rl.py`
- IPC tests: `tests/ipc/`
- Observation tests: `tests/base/test_np_env.py`, `tests/utils/test_final_observation.py`

## Related Documents

- [ADR Index](README.md)
- [RL Infrastructure 开发标准](../zh_CN/development-standard.md)
- [协作流程](../zh_CN/collaboration.md)
