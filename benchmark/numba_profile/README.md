# numba_profile：面向 `g1_motion_tracking` 的 fused `update_state` profile

这个目录用于回答一个具体问题：把单进程训练里单线程执行的 Env
overhead（obs / reward / termination）改成 Numba `prange` 融合 kernel，收益
上限有多大；同时，这种改法是否还能保留 UniLab 现有的结构化、配置驱动
reward 写法。

这是 issues **#651**（collector active throughput 目标）、**#663**（Env
overhead 并行化）和 **#665**（Numba `prange` 融合上限）的后续验证。本目录验证的是
`g1_motion_tracking` 的 task-specific reward + termination 热路径切片，提供
结构化 Numba 实现，并把**速度**和**数值一致性**放在同一套脚本里验证。

不需要 MuJoCo / Motrix：这个 profile 从 backend 状态已经 materialize 成 typed
arrays 之后开始，只运行机器人状态和 motion reference 之间的纯数组计算。输入数组用
真实 shape / dtype 合成。

## 文件说明

| 文件 | 作用 |
|------|------|
| `spec.py` | 单一事实源：维度、`RewardConfig` scales/stds、termination 阈值、`TERM_ORDER`。与 `tracking.py` 对齐，行号写在注释里。 |
| `state.py` | 按真实 shape 构造确定性的 synthetic batch：`N envs x 14 bodies x 29 DoF`，约 2% termination。 |
| `numpy_reference.py` | golden oracle：把 `tracking.py` 中的 `_reward_*` / `_compute_terminations` 数学逻辑逐项 port 到 numpy。 |
| `numba_terms.py` | 每个 reward term 一个 `@njit(inline="always")` scalar device function，`.py_func` 可用于 Python 侧调试。 |
| `numba_fused.py` | `@njit(parallel=True)` 的 `prange` driver：静态 term superset、scale vector gate、per-thread log scratch、numpy fallback。 |
| `test_parity.py` | 一致性验证：numba vs numpy 的 reward、termination、per-term log，以及 device `.py_func` vs numpy。 |
| `bench.py` | 性能验证：按 `num_envs` 和线程数对比 numpy 与 numba。 |

## 运行

```bash
uv run --with numba python -m benchmark.numba_profile.test_parity
uv run --with numba python -m benchmark.numba_profile.bench
PROBE_NUM_ENVS=32768 uv run --with numba python -m benchmark.numba_profile.bench
```

`numba` 是唯一额外依赖。一次性运行可用 `uv run --with numba ...`；如果本地环境已经
固定，也可以先执行 `uv pip install numba`。

## 与真实 task 的对应关系

热路径数学逻辑来自 `src/unilab/envs/motion_tracking/g1/tracking.py`：

- **11 个默认 reward-scale terms**：来自 `RewardConfig.scales`，保留默认
  `scales` 和 `std_*`。默认有 8 个非零 scale；`motion_ee_body_pos_z` /
  `motion_joint_pos` / `motion_joint_vel` 为 `0.0`，和真实 reward loop 一样跳过。
  真实 `_init_reward_functions` registry 里还有 `undesired_contacts`，但默认 config
  没有给它 scale，因此不属于这个 default-scale profile。
- **Reward 数学逐项 port**：anchor pos / ori、`2 * acos(abs(dot))` 的 quat error、
  per-body mean xyz error、joint-limit violation、action-rate L2、`exp(-err/std^2)`、
  最终 `reward *= ctrl_dt`，其中 `ctrl_dt = 0.02`。
- **Termination 逻辑逐项 port**：anchor-Z、基于 body-frame gravity-z 的
  anchor-orientation、end-effector-Z。
- **维度对齐真实配置**：`NUM_ACTION = 29`，14 个 tracked bodies，anchor body 为
  `torso_link`（idx 7），end-effectors 为 `{ankles, wrists}`（idx 3、6、10、13）。

唯一有意偏离：这里喂 synthetic robot / motion arrays，而不是 stepping physics。
原因是本 profile 验证的是 Env overhead 里的数组计算切片，不是 physics。输入幅度调到
reward 非退化，并产生约 2% termination，用来匹配 #665 probe 中的 reset/copy 压力。

## 结构化方案

这个实现的目标不是把 reward 写死到一个难维护的大 kernel 里，而是在保留配置驱动
属性的前提下做机器码融合：

1. **单一 term 源码**：`numba_terms.py` 中每个 reward term 一个 scalar
   `@njit(inline="always")` 函数。kernel 编译时内联这些函数，因此没有中间
   `(N, ...)` 数组；源码层面仍然保持“一项 reward 一个函数”，且 `.py_func` 可在
   Python 侧调试。
2. **静态 superset + scale vector**：`numba_fused.py` 不做 codegen，也不在 nopython
   中传 runtime dict。kernel 固定覆盖 `TERM_ORDER` superset；冷路径从 config dict
   构造 dense `scale` vector。`scale == 0` 时贡献为 0，等价于 numpy 参考实现里的
   `continue`。权重仍由 config 拥有，不 baked into compiled code。
3. **统一 registry / 顺序**：`spec.TERM_ORDER` 负责 name 与 index 的唯一映射；
   numpy oracle、numba kernel 和 per-term log 都共享这个顺序。
4. **per-thread log scratch**：每个线程写自己的 `(nthreads, N_TERMS)` scratch，再在主
   线程聚合。直接共享 `log[k] += ...` 会在 cache line 上 false sharing，是 #665 中
   已经测到的 scaling ceiling。
5. **一致性作为 gate**：`numpy_reference.py` 保留为 oracle；`test_parity.py` 验证
   total reward、termination、每个 active per-term log，以及 device `.py_func` 与
   numpy term 的一致性。由于 `fastmath` / FMA 会重排浮点计算，验证采用容差而非
   bit-exact。
6. **小 batch fallback**：`update_state(force="auto")` 在 `num_envs < 256` 时回落到
   numpy，因为并行 kernel 的 launch / barrier 成本不划算。

## 结果

以下数字来自 Xeon 8568Y+、160T 上的一次 standalone profile run。它们是本目录
`update_state` micro-slice 的结果，不是 #665 原始 standalone probe 数字，也不是完整
训练路径吞吐。

`update_state` 切片只包含 reward + termination，dtype 为 float32：

| num_envs | numpy | numba 1T（fusion） | numba best | speedup |
|---------:|------:|-------------------:|-----------:|--------:|
| 8 192  | 8.63 ms | 2.67 ms（3.2x） | 0.33 ms @ 48T | 26.6x |
| 32 768 | 35.57 ms | 10.65 ms（3.3x） | 0.73 ms @ 64T | 49.0x |

结论与 #665 的判断一致：单核 fusion 主要消除 per-term intermediate arrays，带来约
3.2-3.3x；在这个 standalone profile 中，多线程并行又带来约 8-15x；batch 越大，
launch / barrier 成本越容易摊薄。

一致性方面，total reward 和每个 per-term log 都在容差内匹配，termination exact。

## 范围和 caveats

- 这不是完整 `NpEnv.update_state` 的通用 numba 化。真实热路径还包含 backend getter、
  sampler、obs dict/info、reset/RNG 等边界；落地时更合理的形态是 task-owned typed
  snapshot + task-specific fused kernel。
- 这里只覆盖 reward + termination 这一段。`reset_done`（backend `set_state`）和 obs
  noise 的多次 `standard_normal` 是另外的 ceiling，Numba `prange` 不会自动解决。
- Obs assembly / concat 没有在这个 profile 中优化；它更接近 concat-bound，真实集成时
  需要重新确定 kernel 边界。
- 端到端 collector throughput 会被 physics、reset、RNG、obs assembly 稀释。#665 中
  motion_tracking overhead -71%、throughput 约 +72% 是针对当时服务器/profile 的投影，
  不是本目录已经完成的集成训练结果。
- 首次调用会触发 JIT 编译；`cache=True` 会把编译产物缓存在
  `__pycache__/*.nbi` / `*.nbc`，跨进程复用。
