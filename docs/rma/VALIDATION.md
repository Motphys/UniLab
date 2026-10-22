# RMA 分支验证记录

日期：2026-10-22。

## 已通过

- 导入本地 RMA 增量层及精选实验数据；数据汇总严格匹配 6 份 JSON 和 3000 行训练 CSV。
- `uv run --no-sync python scripts/report_rma.py`：生成汇总 CSV 与两张静态图；图表已人工查看。
- `uv run --no-sync python -m unittest discover -s tests/rma -p 'test_*.py'`：41 项通过，包含追踪开关不改变动作、终止状态对齐、控制映射、单变量配置及旧模型防覆盖检查。
- `cmake -S deploy_cpp -B build/rma -DCMAKE_BUILD_TYPE=Release`：配置成功。
- `cmake --build build/rma -j2`：构建成功。
- `ctest --test-dir build/rma --output-on-failure`：3/3 通过（并发一致性、mock 控制和 12 个非零 one-hot 动作的执行器映射）。
- `make check` 中的 ruff format/check、`mypy src/unilab`：通过。

核心测试与报告脚本使用现有 WSL Python 环境。C++ 为 GNU 11.4.0、CMake 生成的 Release 构建。
没有追加 Go2 长训，没有更改用户运行目录的原模型或基线仿真配置。
新增独立任务 `rma_forward_no_pose`，只关闭默认站姿偏离惩罚；Hydra 完整解析对比只差这一项。
在原 1.3.3 运行目录中完成 2 环境 × 24 步 × 1 次更新的 CPU 训练冒烟测试，checkpoint 与日志保存成功。
随后 2 环境 × 3 步追踪评估成功；惩罚分项为 0，新任务 YAML 已存入快照，原基线模型 SHA256 不变。
这仅验证实验入口；正式 3000 次更新的 EXP-003 尚未运行，没有性能改善结论。

原 1.3.3 运行目录中已执行一次真实追踪评估：`teacher_forward_v1.pt`、种子 123、8 环境 × 1000 步，
生成 8000 行 CSV、独立 JSON 与三张图。全程速度/RMSE 与此前该模型同种子的报告一致。
逐行验证分项 rate 之和乘 0.02 与实际 reward 的最大误差约 `7.98e-09`。
完整压缩 CSV、摘要、配置记录及图见 `results/rma/diagnostics/EXP-002/seed123/`。
这只验证 1.3.3 中的新增评估入口，不等于 1.3.4 真实训练完成兼容。

使用旧学生权重重新导出 ONNX，12 组输入数值检查通过，最大误差 `2.682209e-07`。
导出元数据已区分策略关节顺序与场景执行器存储顺序；C++ 输出边界已按映射散射。
ONNX 权重图本身未加排列；尚未用真实机器人或 TensorRT 机器人闭环验证本次修正。

## 完整门禁尚未通过

在本 fork 1.3.4 源码上，复用了已有 1.3.3 的 WSL 环境运行完整门禁：

```bash
UV_PROJECT_ENVIRONMENT=/home/abc/my_unilab/.venv UV_NO_SYNC=1 make check
UV_PROJECT_ENVIRONMENT=/home/abc/my_unilab/.venv UV_NO_SYNC=1 make test-all
```

Pyright 报告上游现有框架文件的 10 项错误和 3 项警告，包括缺失的 packed-reset/capability API、
未解析导入与可选值检查。报错位置没有被本次改动修改。
依赖/源码版本不一致是已观察到的限制，尚未证明这是所有错误的唯一原因。
`make test-all` 因前置类型检查未通过而未到达后续全仓测试阶段，不能报告成全仓测试通过。
未为通过门禁而升级正在使用的训练环境或更改上游框架代码。

若创建 PR，应保持 Draft；本地核心验证不替代完整门禁和当前 head 的远端 CI。
1.3.4 分支下的真实 Go2 训练、旧 checkpoint 行为兼容、TensorRT/真机与跨引擎测试未在本次执行。

## 历史数据边界

发布的运动指标是已有本地 1.3.3 运行副本产生的结果，不是当前 1.3.4 分支重跑的结果。
`results/rma/forward_v1/config_snapshot` 是训练完成后从仍在使用的工作副本采集的配置，
并非训练开始时自动保存的不可变快照；不能据此补造完整历史溯源。
发布副本的 `base.yaml` 仅清除了行尾空白，未修改 YAML 参数；原运行文件的哈希另有记录。
原运行配置、CSV 与来源 SHA256 记录见 `results/rma/provenance.json`；
checkpoint SHA256 见 `results/rma/forward_v1/checkpoint_metadata.json`。
