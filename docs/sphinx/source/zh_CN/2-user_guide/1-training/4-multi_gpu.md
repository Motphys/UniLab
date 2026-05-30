# 多 GPU

语言: 简体中文

当前的多 GPU 旋钮位于共享的 off-policy 训练配置中，即 `training.num_gpus`。该字段
由 off-policy 与 FlashSAC 路径消费；PPO、MLX PPO 和 APPO 并不暴露相同的多 GPU
contract。

```bash
uv run train --algo sac --task g1_walk_flat --sim mujoco \
  training.num_gpus=2 \
  training.no_play=true
```

将 task 与 backend 选择保留在 `--task` 和 `--sim` 中：

```bash
uv run train --algo td3 --task g1_walk_flat --sim mujoco \
  training.num_gpus=2
```

修改多 GPU 行为时，请在最接近 off-policy runner 与 IPC 边界处进行验证，而不是仅检
查顶层命令。

## Navigation

- Index: [文档](0-index.md)
