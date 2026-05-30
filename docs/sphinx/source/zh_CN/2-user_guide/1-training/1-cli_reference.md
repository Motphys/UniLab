# CLI 参考

语言: 简体中文

UniLab 为常见的训练与回放路由提供软件包命令，同时保留底层脚本以便调试
Hydra 组合。

## 统一命令

| 目标 | 命令形式 | 路由到的脚本 |
| --- | --- | --- |
| PPO | `uv run train --algo ppo --task <task> --sim <backend>` | `scripts/train_rsl_rl.py` |
| MLX PPO | `uv run train --algo mlx_ppo --task <task> --sim <backend>` | `scripts/train_mlx_ppo.py` |
| APPO | `uv run train --algo appo --task <task> --sim <backend>` | `scripts/train_appo.py` |
| SAC | `uv run train --algo sac --task <task> --sim <backend>` | `scripts/train_offpolicy.py` |
| TD3 | `uv run train --algo td3 --task <task> --sim <backend>` | `scripts/train_offpolicy.py` |
| FlashSAC | `uv run train --algo flashsac --task <task> --sim <backend>` | `scripts/train_offpolicy.py` |

示例：

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco
uv run train --algo appo --task g1_motion_tracking --sim motrix training.no_play=true
uv run train --algo sac --task g1_walk_flat --sim mujoco training.no_play=true
uv run train --algo flashsac --task go2_joystick_flat --sim mujoco
```

CLI 会根据 `--algo`、`--task`、`--sim` 以及可选的 `--profile` 构造出 owner YAML
路径。定义路由的取值必须使用 CLI flag；命令之后的 Hydra override 用于设置诸如
`algo.max_iterations`、`algo.num_envs` 和 `training.no_play` 等字段。

## 评估

`uv run eval` 会设置 `training.play_only=true`，并可选地将 `--load-run` 映射到
`algo.load_run`。

```bash
uv run eval --algo ppo --task go2_joystick_flat --sim mujoco --load-run -1
uv run eval --algo ppo --task go2_joystick_flat --sim motrix --load-run -1 \
  --render-mode record
```

支持的渲染模式为 `auto`、`interactive`、`record` 和 `none`。

## 演示

```bash
uv run demo dance
uv run demo wallflip
uv run demo boxtracking
uv run demo locomani
uv run demo inhandgrasp
uv run demo dance --refresh --device cpu
```

可用的 demo：`dance`、`wallflip`、`boxtracking`、`locomani`、`inhandgrasp`。
每个 demo 在首次运行时会从 `unilabsim/unilab-checkpoints` 这个 Hugging Face
数据集拉取预训练检查点，并缓存到 `src/unilab/assets/checkpoints/<demo>/model_0.pt`。
传入 `--refresh` 可重新下载。

demo 入口由 `src/unilab/demo.py` 实现，并从 `src/unilab/cli.py` 路由。

## 底层脚本

当你需要检查 Hydra 配置组或复现脚本层面的问题时，底层脚本仍然可用。在正常使用
中，请将定义路由的取值保留在上面的统一 CLI flag 中。

对于 off-policy 路由，请保持 `--algo` 与 `conf/offpolicy/task/<algo>/` 下的
owner 树对齐；不要在 `--task` 中包含算法名称。

## 常用 Override

```bash
training.no_play=true
algo.max_iterations=10
algo.num_envs=128
algo.load_run=-1
training.logger=wandb
```

回放时使用 `uv run eval ... --load-run ...`，渲染行为则使用
`--render-mode record` 或 `--render-mode none`。

后端选择属于 CLI 的 `--sim` 选项以及由此得到的 owner YAML。不要将
`training.sim_backend=<backend>` 当作独立的切换开关使用。

## Navigation

- Index: [文档](0-index.md)
