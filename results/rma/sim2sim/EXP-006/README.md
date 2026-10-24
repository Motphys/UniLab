# EXP-006：精选实验数据

对应[实验报告](../../../../experiments/rma/EXP-006_sim2sim_nominal.md)。
本目录保存一次冻结策略、名义条件、单种子评估的证据，不是跨场景鲁棒性结论。

| 内容 | 用途 |
| --- | --- |
| `comparison.json` | 两端动作执行后 IMU 速度的统一统计口径 |
| `plots/` | 速度、动作接近边界比例、Genesis 姿态与高度三图 |
| `traces/mujoco.csv.gz`、`traces/genesis.csv.gz` | 两端各 1000 行逐步数据，可离线复查 |
| `mujoco/run_config.json`、`mujoco/summary.json` | 原始评估元数据与原报告 |
| `provenance.json` | 原始路径、模型/源文件 SHA256、数据行数与文件校验值 |
| `analysis_snapshot/` | 本次离线分析脚本与名义评估约定的存档 |

注意：原 MuJoCo summary 的速度口径是动作执行前；跨引擎比较以
`comparison.json` 的动作执行后口径为准。MuJoCo 原日志生成时模式名称尚未改为 v2，
但此次运行已使用 kd=0 和初始回合计数清零；保留原始元数据不回写。

## 离线复查

无需重新训练或启动仿真。将两份 gzip 分别解压为独立临时目录中的
`mujoco/transitions.csv` 与 `genesis.csv`，再用 Python 3.10+ 和 Matplotlib 执行
存档的 `compare_sim2sim_traces.py`，参数为：

```text
--mujoco-trace <临时目录>/mujoco
--genesis-csv <临时目录>/genesis.csv
--output-dir <新的输出目录>
--command 0.5 0 0
```

运行时副本与 GitHub fork 的源码版本不同；本次仅归档结果，不把运行目录整体覆盖到 fork。
运行时没有 Git 历史，不能用当前 fork 的 commit 冒充该实验的源码版本。
源文件与模型哈希记录于 provenance；该存档不含模型、全部运行源码、资产及完整依赖，
因此支持数据离线复算，不宣称仅克隆仓库即可重跑全部仿真。

`.pt`、`.onnx`、`.engine`、完整训练日志与资产留在本地；此目录只保存精选证据。
下一步先记录降速时的接触与控制信号，区分实际力矩与代数估计，再选择一个对照变量。
