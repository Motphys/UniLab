# 来源与贡献范围

- 上游框架：UniLab，仓库 `https://github.com/unilabsim/UniLab`。
- 用户 fork：`https://github.com/gongbaojiding-1123/UniLab`。
- 原有 UniLab 源码、许可证和归属保持不变；许可证见仓库根目录 `LICENSE`。
- RMA 增量：`rsl_rl/unilab_rma`、RMA 脚本与配置、`deploy_cpp`、相关测试与本组实验文档。
- `legged_gym/go2_config.py` 只声明接口常量，不是完整 legged_gym 的移植。
- 当前训练器是本项目独立 PPO；不能将其结果归属于官方 RSL-RL 的默认实现。
- RMA 的教师隐变量/历史适应思路属于已有研究，本项目不宣称发明该方法。
- 机器人资源、TensorRT/CUDA、PyTorch 等依赖仍遵循其各自许可，未随实验结果重新分发。
- 代码和文档使用了 AI 辅助，结果数据来自已保存的训练 CSV 与用户提供的评估输出。

原运行目录未携带 Git 历史，无法证明它与某个上游 commit 完全一致。
本分支保留关键文件差异记录，不通过猜测补填上游 commit。
