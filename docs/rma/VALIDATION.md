# RMA 分支验证记录

日期：2026-10-22。

## 已通过

- 导入本地 RMA 增量层及精选实验数据；数据汇总严格匹配 6 份 JSON 和 3000 行训练 CSV。
- `uv run --no-sync python scripts/report_rma.py`：生成汇总 CSV 与两张静态图；图表已人工查看。
- `uv run --no-sync python -m unittest discover -s tests/rma -p 'test_*.py'`：34 项通过，包含教师比较中拒绝学生报告的回归检查。
- `cmake -S deploy_cpp -B build/rma -DCMAKE_BUILD_TYPE=Release`：配置成功。
- `cmake --build build/rma -j2`：构建成功。
- `ctest --test-dir build/rma --output-on-failure`：2/2 通过（并发一致性和 mock 控制）。
- `make check` 中的 ruff format/check、`mypy src/unilab`：通过。

核心测试与报告脚本使用现有 WSL Python 环境。C++ 为 GNU 11.4.0、CMake 生成的 Release 构建。
没有追加 Go2 长训，没有更改用户运行目录的策略参数或仿真配置。

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
