# UniLab Documentation

本页是 UniLab 文档的入口索引。根据你的角色选择对应的阅读路径。

---

## 普通用户

想安装、运行、训练，不修改代码。

- [01 快速开始](users/zh_CN/01-getting-started.md) — 安装依赖、第一次运行、验证环境
- [02 仿真后端](users/zh_CN/02-simulation-backends.md) — MuJoCo / Motrix 支持范围与选择
- [03 训练指南](users/zh_CN/03-training.md) — 训练、回放、续训、Hydra 覆盖、W&B
- [04 算法说明](users/zh_CN/04-algorithms.md) — APPO、FastSAC、FastTD3 用法与区别
- [05 G1 全身运动跟踪](users/zh_CN/05-motion-tracking.md) — G1 运动跟踪任务
- [06 域随机化](users/zh_CN/06-domain-randomization.md) — 域随机化配置与最佳实践

---

## 合作开发者

要提交 PR、扩展功能、修改架构。

1. [CONTRIBUTING.md](../CONTRIBUTING.md) — 环境设置、常用命令、提交规范、PR 流程
2. [RL Infrastructure 开发标准](developers/zh_CN/development-standard.md) — Runtime Model、分层架构、Design Principles、Contract、验证边界
3. [协作流程](developers/zh_CN/collaboration.md) — Issue / Milestone / PR 协作规则、ADR 治理
4. [ADR 索引](developers/adr/README.md) — 已落地的架构决策记录

---

## AI Agents

1. [AGENTS.md](../AGENTS.md) — 核心原则、高风险区域、关键文件指针
2. 需要深层上下文时，参考 [开发标准](developers/zh_CN/development-standard.md)。
