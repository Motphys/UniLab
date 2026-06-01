# 从 RSL-RL 迁移

你已经在独立使用 RSL-RL 了？好消息：UniLab 把 RSL-RL PPO 作为其受支持算法之一
（`unilab.algos.torch.rsl_rl_ppo`）提供，而且几乎是即插即用的。

## 迁移进 UniLab 后你能获得什么

1. **Env contract。** RSL-RL 把 env 结构留给你自己处理。UniLab 的
   `unilab.base.np_env.NpEnv` 标准化了 obs/info/reset 的签名，让并行与复位
   更不容易出错。
2. **任务 owner。** 基于 Hydra 的配置组合，外加 registry 驱动的
   backend / task / algo 选择。不再需要为每种机器人编写定制的训练脚本。
3. **异步 runner。** 把 RSL-RL PPO 包进 `unilab.algos.torch.appo`，在拥有许多
   CPU 核心的机器上获得更高吞吐量。
4. **部署流程。** 配合正确 wrapper 的 ONNX 导出、安全层文档，以及
   {doc}`../1-sim_to_real/1-overview` 流水线。

## 你不会失去什么

- 相同的 PPO 算法与超参数 —— RSL-RL 是被包装，而非被重新实现。
- Checkpoint 兼容性：在策略架构匹配的前提下，来自 UniLab 的
  `runs/<run>/model_*.pt` 文件可以被原生 RSL-RL 加载。

## 迁移步骤

1. 把你的 env 移到 `unilab.envs.<family>.<task>/`。
2. 把你的训练配置转换为 `conf/ppo/<task>/` 下的一个 Hydra 组。
3. 运行 `uv run train --algo ppo --task <task> --sim <backend>`。
4. 把 reward 曲线与你独立运行的 RSL-RL 基线对比。

## 何时不该迁移

如果你已经有一条能用的流水线、没有真实硬件目标，并且你主要只是需要 RSL-RL 提供
算法 —— 那么 UniLab 就有点杀鸡用牛刀了。当你需要以下任意一项时再用它：多后端、
异步采集、ONNX 部署，或任务注册。
