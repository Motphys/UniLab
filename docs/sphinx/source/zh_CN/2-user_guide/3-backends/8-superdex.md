# SuperDex 后端

SuperDex 是由 `unisim.backend.superdex` 拥有的可选 CPU 物理后端。UniLab 首个
owner 为固定基 `FR3JointTarget`，配置位于
`src/unilab/conf/ppo/task/fr3_joint_target/superdex.yaml`。任务使用 7 维力矩动作、
21 维观测、关节状态 reset 和标准 NumPy manager。当前支持等级为 **Configured**；
短 rollout 或少量训练迭代不能证明完整训练效果、性能或跨平台支持。
实施见 [#1534](https://github.com/Motphys/UniLab/issues/1534)，所属
roadmap 为 [#1533](https://github.com/Motphys/UniLab/issues/1533)。

## 本地开发安装

该开发配置使用本地链接的 UniSim 与 UniLab，不要求发布新版本。SuperDex
Physics/Robotics 1.0.0 要求 Python 3.12；CPU 物理不需要 CUDA。FR3 owner 暂不包含
原生渲染和视频，默认 `no_play=true`。

在 UniLab checkout 中使用已有的 Python 3.12 环境，或创建环境后安装本地包：

```bash
uv venv --python 3.12
export UNILAB_LOCAL_UNISIM=/absolute/path/to/unisim
uv pip install -e "${UNILAB_LOCAL_UNISIM}[superdex,mujoco]" -e . --group pyproject.toml:dev
export UV_NO_SYNC=1
export SUPERDEX_ASSETS_PATH=/absolute/path/to/project_superdex/assets
```

`UNILAB_LOCAL_UNISIM` 启用严格的本地依赖验证：测试同时检查 editable 安装元数据
和实际 import 路径确实指向指定 checkout。不设置时保留正常的索引发布包检查。
`UV_NO_SYNC=1` 让现有 Make 目标保留本地链接；`uv sync` 会重新解析锁定的发布依赖。

FR3 原生资产保留在上游 checkout。asset hub 注册
`bots/arms/fr3_v2/fr3_v2.superdex_bot`，在物理构造前验证 collision SDF、render、
`LICENSE` 和 `NOTICE`。UniLab 不打包或下载这些二进制。单次运行可通过
`env.superdex_assets_root=/absolute/path/to/project_superdex/assets` 覆盖环境变量。

## 运行 FR3 任务

```bash
uv run --no-sync train --algo ppo --task fr3_joint_target --sim superdex \
  algo.max_iterations=2 algo.num_steps_per_env=16 \
  algo.algorithm.num_learning_epochs=1
```

目标关节角、reward、reset 范围和动作缩放由任务 `base.yaml` 声明。力矩上限
`[20,20,20,20,5,5,5]` Nm 是显式研究配置，不是硬件额定值；
`superdex_effort_limits` 在 native backend 边界声明同样的上限。
`superdex_num_threads=0` 使用串行物理；线程数是进程级设置，同进程多个 backend
实例必须一致。进程并行仍由现有 spawn collector 拥有。

固定根具有名称和可读的 body state，但 reset 只写 joint state，不要求 free-root
layout。该任务不需要接触 sensor、相机、site Jacobian 或材料 DR。
`play_render_mode=none` 会完全跳过回放，不能作为 checkpoint rollout 已执行的证据。

`superdex_allow_contact_approximation` 默认 `false`，只供经过审核的 MJCF 转换配置
显式启用。启用后会警告 contact/material 近似，包括 torsional/rolling friction
不等价；这不代表任意 MJCF 任务已兼容。
接触查询返回最近一次完成求解的结果。reset 清除该结果，需要一次正时间步才会
产生新接触，不能把 reset 后的 contact 当成即时几何重叠测试。运动学 body/joint
getter 则在 reset 后立即刷新。

## 验证与归属

`go2_joystick_flat/superdex` PPO owner 是研究性质的 sim2sim 配置。它继承 MuJoCo
owner，保留 49 维 actor 观测、52 维 critic 观测、12 维位置目标动作、归一化、网络
维度和控制时序；关闭 runtime PD gain DR，并显式接受接触近似。adapter 的冷路径
MJCF 转换有明确限制，该 owner 不代表任意场景或行走效果等价。

先用 MuJoCo owner 创建一个小型来源 checkpoint，再把路径传给可选 checkpoint 测试：

```bash
uv run --no-sync train --algo ppo --task go2_joystick_flat --sim mujoco \
  algo.num_envs=2 algo.max_iterations=2 algo.num_steps_per_env=16 \
  algo.algorithm.num_mini_batches=1 algo.algorithm.num_learning_epochs=1 \
  training.device=cpu training.no_play=true
export UNILAB_SUPERDEX_GO2_CHECKPOINT=/absolute/path/to/source/run/model_1.pt
uv run --no-sync pytest tests/envs/test_go2_superdex.py -q
```

测试在构造 env 前验证来源 `run_config.json`，检查修改动作语义时确实拒绝，随后
通过 production playback session 加载真实策略，无渲染执行 64 个 SuperDex 控制步。
它验证有限数值和接口兼容性；只训练两轮的 checkpoint 不以可靠行走为验收标准。

```bash
uv run --no-sync pytest tests/assets/test_superdex_assets.py \
  tests/envs/test_fr3_superdex.py tests/test_cli_runtime_requirements.py -q
```

原生测试要求 SDK 和 `SUPERDEX_ASSETS_PATH`，覆盖有限数值 rollout、局部 reset
隔离、即时观测刷新及 spawn `EnvFactory`。缺失 SDK/资产会明确 skip，不能将 skip
记为原生验证通过。基础资产和配置测试不依赖原生资产 checkout。

引擎转换与物理由 UniSim 拥有；资产注册、Hydra 与任务 term 由 UniLab 拥有。
相关约束见 {doc}`/adr/ADR-0007-unisim-extraction-boundary`、
{doc}`/adr/ADR-0006-community-manager-api-on-numpy-runtime` 和
{doc}`/adr/ADR-0002-backend-capability-boundary-for-play-and-snapshot`。
