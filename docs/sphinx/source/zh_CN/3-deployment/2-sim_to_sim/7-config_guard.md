# 跨后端配置守卫

跨后端回放（在一个后端训练、到另一个后端 `eval`）时，UniLab 会自动校验目标后端配置与训练时的策略契约是否兼容，避免用错配置静默加载出行为异常的策略。整个过程无需手动干预。

## 一个可成功回放的例子

`go2_joystick_flat` 的 MuJoCo 与 Motrix owner 在守卫字段上完全一致，因此跨后端回放可以直接通过：

```bash
# 1) MuJoCo 训练，产生 checkpoint
uv run train --algo ppo --task go2_joystick_flat --sim mujoco

# 2) Motrix 跨后端回放同一个 checkpoint —— 守卫校验通过，正常播放
uv run eval  --algo ppo --task go2_joystick_flat --sim motrix --load-run -1
```

## 生效链路

1. **训练时**：`ExperimentTracker` 把决定策略 I/O 的契约字段快照进 `run_config.json` 的 `contract_snapshot`（不改动 checkpoint 格式，历史 checkpoint 天然兼容）。
2. **回放时**：`eval` 读取 `--sim` 指定的**目标后端** owner 配置（如 `conf/ppo/task/go2_joystick_flat/motrix.yaml`），并注入 `training.play_only=true`。
3. **建 env 前**：五个 play 入口（rsl_rl / appo / offpolicy / mlx_ppo / him_ppo）调用 `resolve_sim2sim_config`，把目标配置与源 run 的契约快照逐字段比对。
4. **加载权重时**：`policy_load_dim_guard` 包裹 checkpoint 加载，把底层 tensor 维度不匹配的晦涩报错重抛为清晰的 sim2sim 诊断。

## 守卫的字段

守卫按 dotted path 分三档（定义见 `src/unilab/training/sim2sim.py`）：

| 档位 | 行为 | 字段 |
|---|---|---|
| **DENYLIST** | 差异即 `CrossBackendIncompatibleError`，中断 | `algo.obs_groups`、`env.control_config.action_scale`、`algo.policy.actor_hidden_dims` / `critic_hidden_dims`、`algo.empirical_normalization` / `algo.obs_normalization`、`env.sampling_mode` |
| **WARNING_LIST** | 仅打印 warning，继续 | `reward.*`、`env.control_config.simulate_action_latency`、`env.ctrl_dt` |
| **ALLOWLIST** | 自由覆盖，不检查 | `training.sim_backend`、`env.scene`、`training.play_steps`、`env.domain_rand`、`env.noise_config`、`env.commands.vel_limit` |

## 当 DENYLIST 字段不一致时

若目标后端的 DENYLIST 字段与训练时不同（例如某 task 的两端 owner 在 `action_scale` 上取值不同），守卫会在建 env 前中断并列出差异字段。处理方式二选一：

- **对齐契约**（推荐）：让目标后端 owner 的 DENYLIST 字段与训练后端一致，再回放。
- **强制放行**（自担风险）：`uv run eval ... training.sim2sim_strict=false`，把 DENYLIST 差异降级为 warning。

> 兼容旧 run：若 `run_config.json` 没有 `contract_snapshot`（早期训练），守卫自动跳过并打印 warning，不会中断现有工作流。

## 另请参阅

- {doc}`1-backend_swap`
- {doc}`4-reward_parity`
- {doc}`../../4-developer_guide/9-sim2sim_contract_status`
