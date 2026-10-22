# RMA 实验复现与验证范围

## 先区分两个代码版本

旧实测来自没有 Git 历史的本地 UniLab 1.3.3 工作副本。
本分支从用户 fork 的 `c19b8810059e3c495dc5394a5652d3c82c00f70f`（1.3.4）开始，导入 RMA 增量代码。
`results/rma/provenance.json` 保存了运行依赖、来源文件哈希和关键框架文件差异。
存在差异时，不应把旧模型成绩标成当前 Git commit 上重跑的结果。

## 复现表格和图表

在有 PyTorch 和 matplotlib 的 Python 环境中，从项目根目录执行：

```bash
uv run --no-sync python scripts/report_rma.py
```

输入是 `results/rma/baseline/seed*.json`、`results/rma/forward_v1/seed*.json`
和 `results/rma/forward_v1/training_metrics.csv`。
输出是 `results/rma/evaluation_summary.csv` 与 `results/rma/figures/*.png`。
训练图使用真实的 3000 行 CSV，平滑线为包含启动区间的尾随 100 次更新均值。
没有填补不存在的模型速度曲线，也没有生成虚构视频。

## 核心算法/接口测试与 C++ mock

```bash
uv run --no-sync python -m unittest discover -s tests/rma -p 'test_*.py'
cmake -S deploy_cpp -B build/rma -DCMAKE_BUILD_TYPE=Release
cmake --build build/rma -j2
ctest --test-dir build/rma --output-on-failure
```

这些测试验证算法接口、历史缓冲、概率/终止处理、导出接口和并发 mock。
它们不能证明 Go2 速度跟踪、真机安全、硬实时或 TensorRT 在目标设备上的精度与延迟。

## 真实评估协议

保留原 checkpoint，固定 `(0.5, 0, 0)`，每个种子运行 8 环境 × 1000 个控制步，
种子为 123、456、789，模型权重不变。完整参数见 EXP-001。
历史 JSON 仅记录总转移数 8000，未单独保存环境数和步数；8 × 1000 来自实验命令记录。
报告脚本核对教师策略、种子、目标命令及总转移数，不能仅凭总转移数证明环境数和步数。
后续评估应同时保存环境数、控制步数与完整命令，不能给历史 JSON 补造缺失字段。
加入 `--diagnostics` 才输出各轴速度、各轴 RMSE 与奖励分项。
统计包含启动和重置过程；下一阶段应提前约定稳态统计窗口，并同时保留全程指标。
日志若使用 `tee` 保存，会混有启动文字，不是纯 JSON 文件。

当前静态配置继承的上游基础奖励定义在 1.3.3 与 1.3.4 有变化。
历史配置快照应与原运行环境一起用于追溯，不能单独覆盖新版本的全部框架文件。
重新运行当前分支得到的结果，应另起实验编号，并保存当前 commit 和完整配置。

## 训练与 checkpoint 语义

3000 次更新 × 64 环境 × 24 rollout 长度 = 4,608,000 次环境转移。
策略/更新 CUDA 不等于物理仿真也在 GPU；本次物理环境为 CPU。
当前 `--checkpoint` 只恢复网络权重，未恢复 optimizer/RNG/环境状态，属于 warm-start。
当前保存行为每 20 次覆盖同一路径；旧运行没有保存全部历史 checkpoint，无法补算其历史速度曲线。

## 环境和资产

机器人资源通过上游资产流程获取，路径与 home keyframe 必须匹配。
不要将 `.venv`、mesh、完整运行目录或大模型文件直接加入 Git。
本分支没有发布训练权重；checkpoint SHA256 用于之后核对同一模型。
环境安装方式以 fork 上游说明为准；复现旧数据需匹配 provenance 中记录的版本及配置。
