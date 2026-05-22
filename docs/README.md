# UniLab Documentation

本页是 UniLab 文档总入口。文档按角色、信息层次和速查场景分层组织。

## 设计原则

- 单个文件只讲一个主题，不把安装、训练、算法、任务、设计讨论混在一起
- 用户文档和 developer 文档严格分层，默认先服务用户运行与速查
- 算法、机器人任务、专题场景分别建独立文件，避免大而全页面
- 目录结构预留双语扩展：后续英文文档按同一层级镜像
- agent / 速查场景需要最短路由，优先给出角色化入口

## 用户入口

先跑起来，再按主题深入。

1. [01 快速开始](users/zh_CN/01-getting-started.md)
2. [02 仿真后端](users/zh_CN/02-simulation-backends.md)
3. [03 训练指南](users/zh_CN/03-training.md)
4. [04 算法说明](users/zh_CN/04-algorithms.md)
5. [05 域随机化](users/zh_CN/05-domain-randomization.md)

专题和速查入口：

- [A 安装与环境](users/zh_CN/A-getting-started/01-install.md)
- [B 训练分专题](users/zh_CN/B-training/01-unified-cli.md)
- [C 算法分专题](users/zh_CN/C-algorithms/01-ppo-torch.md)
- [D 任务索引](users/zh_CN/D-tasks/01-task-index.md)
- [E 后端支持矩阵](users/zh_CN/E-reference/01-backend-support-matrix.md)
- [术语表](glossary.md)

## Developer 入口

默认入口只放正式规范和协作基线；设计草案与调查笔记归档在补充目录。

1. [CONTRIBUTING.md](../CONTRIBUTING.md)
2. [RL Infrastructure 开发标准](developers/zh_CN/development-standard.md)
3. [协作流程](developers/zh_CN/collaboration.md)
4. [Domain Randomization Contract](developers/zh_CN/domain-randomization-contract.md)
5. [ADR 索引](developers/adr/ADR-0000-index.md)

补充资料：

- [Motrix Contact Sensor 适配笔记](developers/zh_CN/motrix-contact-sensor-notes.md)
- [SceneCfg 与场景组合设计](developers/zh_CN/scene-composition-design.md)

## Agent / 速查入口

面向需要快速定位命令、任务入口和规范边界的 agent 或维护者。

1. [Agent 速查](agents/zh_CN/01-agent-quick-reference.md)
2. [AGENTS.md](../AGENTS.md)
