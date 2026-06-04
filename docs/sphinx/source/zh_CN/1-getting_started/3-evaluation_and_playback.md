# 评估与回放

```bash
# 最近一次运行
uv run eval --algo ppo --task go2_joystick_flat --sim motrix --load-run -1

# 无头视频导出
uv run eval --algo ppo --task go2_joystick_flat --sim motrix \
    --load-run -1 --render-mode record

# Off-policy 回放可以跳过 ONNX 导出，但仍然录制 MP4
uv run eval --algo sac --task g1_walk_flat --sim mujoco --load-run -1 \
    --render-mode record training.export_onnx=false

# 演示（首次运行会从 HF 下载检查点）
uv run demo dance
```

渲染模式：

- `interactive` — 打开查看器窗口（macOS Motrix 上的默认值）。
- `record` — 将 MP4 写入 `runs/<run>/playback/`。
- `none` — 跳过渲染，仅计算指标。

`training.export_onnx=false` 目前仅适用于 off-policy 回放路径
（`scripts/train_offpolicy.py` 以及使用 `--algo sac|td3|flashsac` 的 CLI 运行）。它会跳过
`policy.onnx` 的导出与校验，但仍会执行回放和视频录制。

## MuJoCo viewer 可视化脚本

常规评估和视频导出优先使用 `uv run eval`。需要直接打开 `mujoco.viewer`
调试策略时，可以使用低层脚本 `scripts/play_interactive.py`。

`scripts/play_interactive.py` 是通用 MuJoCo viewer 入口，适合 PPO、APPO、
SAC、FlashSAC 和 HORA distill 的策略可视化。它使用 `--algo / --task / --sim`
选择算法和 owner config；无论 `--sim` 选择 MuJoCo 还是 Motrix，窗口都使用
`mujoco.viewer` 可视化，`--sim` 只决定读取哪份配置。

```bash
# 使用 owner config 中的 interactive.action_mode；全局默认是 zero action
uv run scripts/play_interactive.py --algo ppo --task go2_joystick_flat --sim mujoco

# 随机动作
uv run scripts/play_interactive.py --algo ppo --task go2_joystick_flat --sim mujoco \
    interactive.action_mode=random

# 策略动作
uv run scripts/play_interactive.py --algo ppo --task go2_joystick_flat --sim mujoco \
    algo.load_run=-1 interactive.action_mode=policy

uv run scripts/play_interactive.py --algo flashsac --task g1_walk_flat --sim motrix \
    algo.load_run=-1 interactive.action_mode=policy

uv run scripts/play_interactive.py --algo ppo --task go2_joystick_flat --sim mujoco \
    interactive.action_mode=policy interactive.keyboard=true
```

动作模式通过 `interactive.action_mode=zero|random|policy` 选择；不传时使用
owner config 中的设置（全局默认是 `zero`，部分任务 YAML 会覆盖为 `policy`）。
键盘控制通过 Hydra override 开启：`interactive.action_mode=policy
interactive.keyboard=true`。开启键盘控制时会检查 policy obs 是否包含 velocity
command，不满足时退出。处于 `policy` 模式、且 locomotion 速度指令任务的 policy
obs 中包含 velocity command 时，窗口会自动显示绿色目标速度和蓝色当前速度箭头；
`zero` 和 `random` 模式不显示速度箭头。

底层 API 请参阅 `unilab.visualization.playback`。
