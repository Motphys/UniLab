# MuJoCo collector：UniSim 1.4.0 → 1.5.1

在相同 UniLab 代码和 Go2 任务配置下，1.5.1 的 collector 活跃窗口吞吐比 1.4.0 低：1024 环境下降 **36.47%**，4096 环境下降 **42.57%**。三对运行均出现下降，主要增量位于状态更新阶段。本次依赖更新仍按请求精确固定为 `unisim-core==1.5.1`，没有关闭新版正确性修复来提高分数。

## 结果

单位是 **环境 transition/s**，每格为三次独立进程运行的中位数及最小—最大范围，不是 physics substep/s 或训练吞吐。

| 环境数 | 1.4.0 | 1.5.1 | 中位数变化 |
| --- | ---: | ---: | ---: |
| 1024 | 138,780（131,699–141,773） | 88,167（82,537–89,607） | −36.47% |
| 4096 | 209,773（206,357–209,957） | 120,476（117,284–120,949） | −42.57% |

逐对变化：1024 环境为 −36.80%、−40.53%、−33.05%；4096 环境为 −42.57%、−44.14%、−41.39%。未丢弃慢样本。

下表为每次运行的 vector-step 阶段均值，再对三次运行取中位数；各列独立取中位数，不保证其和等于总时长的中位数。

| 环境数 | Physics，旧→新 | Update state，旧→新 | Replay，旧→新 |
| --- | ---: | ---: | ---: |
| 1024 | 5.743 → 6.306 ms | 1.012 → 4.443 ms | 0.524 → 0.581 ms |
| 4096 | 16.344 → 16.811 ms | 2.482 → 15.994 ms | 0.465 → 0.941 ms |

12 次运行的计量窗口中 **reset 数量全部为零**，所以本结果不覆盖 reset 吞吐，也不能用 M2 稀疏 reset 的消融结果解释本次变化。

## 测试条件与口径

- 测量时两版本使用同一 UniLab 提交 `84a00d049eeac9292bbd58c25fcd06a4b64cd712`；没有混入尚未合入的 M2 消费代码。该版本适用于历史 1.4.0。依赖更新附带的默认字段声明和旧单测 fixture 适配发生在测量之后，不改变 collector 热循环。
- 入口为既有 `scripts/benchmark/rl/benchmark_offpolicy_collector_active.py`，案例 `flashsac/go2_joystick_flat/mujoco`。Actor obs=49、critic obs=52、action=12。
- CPU：AMD Ryzen 9 9950X3D2，16 物理核；固定进程和 MuJoCo worker 到 CPU 0–7，共 8 个不同物理核，不使用它们的 SMT siblings 16–23。Torch、OpenMP、BLAS、Numba 线程均固定为 8。未运行其他引擎 benchmark，MuJoCo 和 replay 均在 CPU。
- Python 3.13.14、MuJoCo 3.11.0、mjbatch-uni 0.2.1、NumPy 2.4.4、Torch 2.8.0+cu128、unilab-rl 1.2.1。每次切换用 `uv pip install --no-deps`，逐次核对全部 133 个非 UniSim 包版本一致。
- `algo.seed=1` 与 `env.seed=1`，动作全零；每次新进程预热 30 个 vector steps，再计量 300 步；ReplayBuffer 容量为 64 个 vector steps。
- 每版本、每规模重复三次，串行执行。1024 环境顺序为 A/B、B/A、A/B；4096 为 B/A、A/B、B/A，A=1.4.0、B=1.5.1。没有把此前错误基线 1.5.0 的运行混入统计。
- 指标为 `N × 300 / Σ(env_step + replay + bookkeeping)`；包括环境步进、奖励、终止处理、ReplayBuffer 写入/发布/提交和 collector 记账。不含策略推理、learner 等待、跨进程 IPC、初始化和进程总时长。
- CPU affinity 不是独占 CPU；后台负载、频率和温度未隔离。只有三次进程重复，不据此推广所有任务、机器或训练配置。

## 分析与不能推断的内容

[1.4.0…1.5.1 源码差异](https://github.com/unilabsim/unisim/compare/v1.4.0...v1.5.1)包含 M0 状态正确性修复。新版 MuJoCo `_sync_tracked_body_state()` 在 step 后首次 body 查询时，逐 dirty environment 同步最终运动学/速度及 tracking sensor；旧版直接读取已有缓存。UniLab 的 body 状态查询会消费这条路径。因此，“physics 基本接近、update state 大幅增加”与新增刷新工作一致。

这是有源码支撑的候选解释，不是关闭该路径后的因果消融；本轮没有为跑分禁用刷新、修改旧包或调整任务物理配置。新版也修正了旧状态读数，因此版本间并非单纯执行完全等价语义的速度竞赛。后续性能优化应保留最终状态新鲜度，并针对该同步路径单独验证，不能直接恢复旧缓存行为。

G1 平地行走曾做兼容预检，但 1.4.0 在 sensor XML 物化时将 `pelvis_contour_link.STL` 解析到 `g1/` 而非已有的 `g1/assets/`，构造失败；其异常清理还以 `OSError: Bad file descriptor` 覆盖了原始错误。该任务没有吞吐结果，未纳入表格。

## 复现与证据

在隔离 checkout/venv 中使用上述 UniLab 提交与其余锁定依赖。历史 1.4.0 不满足当前项目 manifest，是通过 `--no-sync` 显式执行的兼容性实验；生产安装和最终 lock 均为 1.5.1。

```bash
uv sync --locked --extra mujoco
uv run unilab-pull-assets --robot go2
# 分别将 VERSION 设为 1.4.0、1.5.1；按上面的交替顺序重复三次。
VERSION=1.4.0
uv pip install --python .venv/bin/python --no-deps "unisim-core==$VERSION"
taskset -c 0-7 env \
  UNILAB_COLLECTOR_TORCH_THREADS=8 OMP_NUM_THREADS=8 \
  OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMBA_NUM_THREADS=8 \
  CUDA_VISIBLE_DEVICES= MUJOCO_GL=disable \
  uv run --no-sync python scripts/benchmark/rl/benchmark_offpolicy_collector_active.py \
  --cases flashsac/go2_joystick_flat/mujoco --num-envs 1024 \
  --warmup-steps 30 --measure-steps 300 --replay-capacity-steps 64 \
  --override algo.seed=1 --override '++env.seed=1' \
  --override '++env.cpu_ids=[0,1,2,3,4,5,6,7]' \
  --out-json /tmp/collector-140-n1024-r1.json
# 同样运行 --num-envs 4096；完成后恢复项目锁定依赖。
uv sync --locked --extra mujoco
```

[汇总 JSON](unisim_1_4_0_vs_1_5_1_collector.json)保存每次吞吐、阶段均值、精确版本、CPU/seed、asset/benchmark hash、原始文件 SHA256 和运行顺序。[完整原始结果 gzip](unisim_1_4_0_vs_1_5_1_collector.raw.json.gz)保留全部 12 次 benchmark 输出、逐步计时与实际命令。原始运行日志另位于忽略目录 `scripts/benchmark/outputs/unisim_1_4_0_vs_1_5_1_collector/`。
