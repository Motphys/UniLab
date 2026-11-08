# 常见问题

按主题分组的快问快答，覆盖"对不上、找不到、行为变了"三类问题。每条给出直接
答案和一个拥有细节的页面链接。

训练过程中的排障（NaN 定位、reward 分解、后端不一致调试、trace 分析）请从
{doc}`../6-how_to/3-training_diagnostics` 开始——本页只做指引，不重复其内容。

## 安装与平台

### 启动时报 `sim=<backend> requires the <backend> extra`

```
SystemExit: sim=mujoco requires the MuJoCo extra. Install it with
`pip install unilab[mujoco]` (or `uv sync --extra mujoco` in a source checkout).
```

CLI 在路由前检查各后端的 Python 依赖（`src/unilab/cli.py`）。用匹配的 extra
重新同步——例如 `uv sync --extra mujoco --extra mjwarp`——再重试。`isaacsim`
和 `isaacgym` 还需要独立的外部 worker 环境，用
`scripts/tools/setup_isaacsim_env.sh` / `scripts/tools/setup_isaacgym_env.sh`
安装。后端 extra 与前置条件见 {doc}`2-installation`。

### `uv sync --extra mujoco` 在编译 `mjbatch` 时失败

`fatal error: Python.h: No such file or directory` 或
`c++: No such file or directory` 表示钉定的 `mjbatch` 引擎在从源码编译，但缺少
C++17 工具链或 Python 开发头文件。uv 管理的 Python（`uv python install`）自带
头文件；系统 Python 需要 `python3-dev`（Debian/Ubuntu）或
`python3-devel`（Fedora/RHEL）。完整的报错对照表（包括
`mjbatch was built against MuJoCo <version>` 看门狗）见
{doc}`2-installation` 的 "Install Error Signatures" 一节。

### 支持哪些平台和 Python 版本？

Python `>=3.10,<3.14`（`pyproject.toml`），依赖解析覆盖 Linux x86_64、Linux
aarch64 与 Apple Silicon macOS；推荐解释器为 `3.13`。Windows 上 `mjbatch`
引擎没有 wheel，因此 MuJoCo 物理后端仅限 Linux/macOS。ROCm 与 Intel XPU 使用
专用的 `make sync-rocm` / `make sync-xpu` profile。详见 {doc}`2-installation`。

### 资产或 checkpoint 下载失败 / Hugging Face 不可达

机器人 mesh、动作片段、场景与 demo checkpoint 在首次使用时从 Hugging Face
数据集仓库惰性下载。`huggingface.co` 不可达时设置
`HF_ENDPOINT=https://hf-mirror.com`（镜像失败时下载器会自动回退重试官方端点
——`src/unilab/assets/hub.py`）。CI 或离线准备可用
`uv run unilab-pull-assets` 预取机器人资产（`--robot all` 拉取全部已注册机器人）。
资产布局见 {doc}`2-installation`；demo 缓存见 {doc}`1-quick_demo`。

### httpx SOCKS 代理缺少 socksio

```
ImportError: Using SOCKS proxy, but the 'socksio' package is not installed.
```

`huggingface_hub` 使用 `httpx`，检测到 `ALL_PROXY=socks5://...` 后需要
`socksio`。安装到项目 `.venv/`（不是 conda）：

```bash
uv pip install httpx[socks] --python .venv/bin/python
```

或改用 HTTP 代理避开 SOCKS：

```bash
unset all_proxy ALL_PROXY
export http_proxy=http://proxy.example.com:8080
export https_proxy=http://proxy.example.com:8080
```

请将 `proxy.example.com:8080` 替换为你的代理入口。

### conda install 连接失败

```
CondaHTTPError: HTTP 000 CONNECTION FAILED for url
```

部分 conda 环境不会使用 shell proxy 变量。位于代理后方且解析包失败时，单独配置
conda 代理：

```bash
conda config --set proxy_servers.http http://proxy.example.com:8080
conda config --set proxy_servers.https http://proxy.example.com:8080
```

### native H2D 扩展不可用

```
Native H2D extension unavailable: CUDA_HOME not set and nvcc not found.
```

off-policy CUDA replay pipeline 可能会 JIT 编译可选的 `unilab_native_h2d`
扩展。该扩展不可用时，UniLab 会回退到纯 PyTorch async copy 路径，因此这条信息通
常是诊断信息而不是致命错误。如果你希望启用 native extension 路径，请检查这些工
具链前置条件：

**C++ 编译器**：

```bash
sudo apt-get install build-essential
```

**CUDA Toolkit**：

```bash
conda install -c nvidia cuda-toolkit=12.8 -y
```

**CUDA 路径**：

```bash
export CUDA_HOME=$CONDA_PREFIX
```

UniLab 会同时检查 `$CUDA_HOME/include` 和
`$CUDA_HOME/targets/x86_64-linux/include` 中的 `cuda_runtime_api.h`。

## 配置与 CLI

### Hydra 拒绝 override：`Key '...' is not in struct`

```
Key 'not_a_key' is not in struct
    full_key: algo.not_a_key
```

该 override key 在组合后的配置中不存在——几乎总是拼写错误、旧版本遗留的 key，
或字段位于别的 group 下。打印完整组合配置并比对 key 路径：

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco --cfg job
```

override 机制与 owner YAML 布局见
{doc}`../2-user_guide/1-training/2-hydra_config`；逐字段表格见
{doc}`../2-user_guide/2-algorithms/1-ppo`（及 APPO/SAC/FlashSAC 兄弟页面）。

### `No owner config exists for algo=..., task=..., sim=...`

该 任务/算法/后端 组合没有 owner YAML
（`src/unilab/conf/<algo>/task/<task>/<backend>.yaml`），训练会拒绝启动。请从
支持矩阵中选择受支持的组合。`eval` 更宽松：它会回退到同任务的兄弟后端 owner，
并通过 sim2sim 契约检查重新套用请求的后端（`src/unilab/cli.py`）。受支持组合见
{doc}`../5-reference/5-support_matrix`；eval 回退及其守门见
{doc}`../3-deployment/2-sim_to_sim/7-config_guard`。

### 设置了 `training.sim_backend=<backend>` 但后端没有切换

它不是后端开关：`training.sim_backend` 是由选中的 owner YAML 设置的身份字段，
CLI 将其视为保留 override。请用 `--sim`（配合 `--algo`/`--task` 选择 owner）切换
后端，例如 `uv run train --algo ppo --task go2_joystick_flat --sim mjwarp`。
背景见 {doc}`../2-user_guide/1-training/2-hydra_config`。

### `--load-run` 拒绝 checkpoint 路径

`--load-run` 只接受 `-1`（最新 run）或任务日志根目录下的 run 目录名，不接受任意
路径（`src/unilab/cli.py`）。run 目录命名为 `YYYY-MM-DD_HH-MM-SS_<sim_backend>`，
位于 `logs/<algo.algo_log_name>/<task>/` 下。日志布局与 resume 语义见
{doc}`../2-user_guide/1-training/3-logging`。

## 训练与调试

### 训练出现 NaN——先看什么？

NaN guard 会写出状态 dump（默认 `/tmp/unilab/nan_dumps` 下的
`nan_dump_<timestamp>_step<N>.npz`），其中记录了检测步与出现 NaN 的 env 行；
用 `uv run unilab-viz-nan <dump> --env-index 0` 回放，并对照检测步之前的
`reward/<term>` 曲线找出发散项。完整诊断链见
{doc}`../6-how_to/3-training_diagnostics`。

### 策略不学习 / reward 曲线异常

先对照 `Train/mean_reward`、`Train/mean_episode_length` 与逐项
`reward/<term>` 曲线，再对照算法参考表检查超参。症状到检查项的映射（reward
爆炸、曲线持平、逐项归因）在 {doc}`../6-how_to/3-training_diagnostics`；指标
key 语义在 {doc}`../2-user_guide/1-training/3-logging`。

### 后端 A 训练的 checkpoint 在后端 B 上拒绝回放

这是 sim2sim 配置守门在按设计工作：训练时会把定义 policy 的字段快照到
`run_config.json`，播放在构造环境之前用目标 owner 配置与该快照逐字段比对。
在默认的 `training.sim2sim_strict=true` 下，DENYLIST 字段（observations、
actions、网络形状、`ctrl_dt` 等）不匹配会抛出
`CrossBackendIncompatibleError`；`training.sim2sim_strict=false` 可将其降级为
警告（风险自负）。字段分级与解决途径见
{doc}`../3-deployment/2-sim_to_sim/7-config_guard`；调试路径见
{doc}`../6-how_to/3-training_diagnostics`。

### 如何用多 GPU 训练？

`CUDA_VISIBLE_DEVICES` 是唯一的 GPU 拓扑来源：不设置则为单 rank；列出每张 GPU
一个条目（物理序号、UUID 或 MIG token 均可）则每条目对应一个 rank。每个 rank
是完整的 learner+collector 对，绑定各自的本地 `cuda:0`；`algo.num_envs` 是
**每 rank** 数量，不是全局预算。拓扑、seed 与聚合语义见
{doc}`../2-user_guide/2-algorithms/1-ppo`。

### NVLink 主机上多卡梯度同步很慢

UniLab 的数据并行启动默认设置 `NCCL_P2P_DISABLE=1` 与 `NCCL_SHM_DISABLE=1`
（`os.environ.setdefault`，你的环境变量优先），因为部分主机的 peer 传输会挂起。
在 NVLink 健康的主机上，启动前导出 `NCCL_P2P_DISABLE=0` 和
`NCCL_SHM_DISABLE=0` 可恢复 peer/SHM 带宽
（`src/unilab/rl/runtime/distributed/sync.py`）。

## 后端与兼容性

### 哪些 任务/算法/后端 组合受支持？

支持矩阵由 registry 条目、owner YAML 与验证清单生成——请以它为支持声明，而不只
是 `--sim` 的可选列表。见 {doc}`../5-reference/5-support_matrix`。

### IsaacSim record/interactive 回放在 `createHydraEngine` 中崩溃

已知运行时阻塞：在已验证的 IsaacSim 5.1.0 / Kit 107.3.3 主机上，两种渲染模式都
在 `AppLauncher` 初始化阶段崩溃
（[issue #2579](https://github.com/Motphys/UniLab/issues/2579)）。headless SAC
`g1_walk_flat` 训练是受支持的能力单元；评估请用 `--render-mode none`。当前证据
与契约矩阵见 {doc}`../2-user_guide/3-backends/4-isaacsim`。

### 同一任务在两个后端上数值不一致

不同仿真器的轨迹不保证逐位一致；UniLab 的契约是 policy I/O 字段一致（由 sim2sim
配置守门强制执行），求解器层面的动力学可以不同。已知后端能力差异见
{doc}`../3-deployment/2-sim_to_sim/6-capability_gaps`；跨后端对齐方法论见
{doc}`../3-deployment/2-sim_to_sim/4-reward_parity`。

## 版本迁移

### `NpEnv` / `np_env.py` 不存在了

1.3.4 移除：Manager-Based 运行时改为 tensor-only，`TorchEnv` 是唯一环境生命周期
基类（{doc}`/adr/ADR-0011-torch-only-manager-based-runtime`）。基于 NumPy 的
代码请迁移到 Torch 契约。版本说明见 {doc}`/changelog`。

### HIM-PPO / HORA / Go2 机械臂与 Sharpa 任务不见了

1.2.0 移出 UniLab：Go2 机械臂操作与 HIM-PPO 迁至
[legged-manipulation_unilab](https://github.com/unilabsim/legged-manipulation_unilab)，
Sharpa 灵巧手与 HORA 迁至 `sharpa_rl_unilab`；相关配置、资产与脚本已从本仓库
移除。版本说明见 {doc}`/changelog`。

### `import uni_rl` 失败 / `unirl` extra 没有了

unilab-rl 包已合并回本仓库的 `unilab.rl`
（{doc}`/adr/ADR-0016-rl-reintegration-into-unilab-rl`）；RL 栈随基础安装提供，
旧 `uni_rl` extra 已移除——不要再传给 `uv sync` 或 `pip install`。导入映射见
{doc}`../3-deployment/3-framework_migration/7-from_unilab_rl`。

### v1.4 的 reset 代码在 `reset_mask` 上抛 `TypeError`

1.5 恢复了 `env_ids` 行索引 reset 契约，并无兼容层地移除了 v1.4 的定长 bool
mask 形式。迁移指南与签名对照表见 {doc}`6-migration_1_4_to_1_5`。

### 新的 TensorBoard run 里没有 `Episode_Reward/*`

有意移除：它与 `reward/<term>` 重复但语义不同。逐项诊断用 `reward/<term>`，整体
进度用 `Train/mean_reward`；历史 event 文件仍会显示旧 key。当前指标契约见
{doc}`../2-user_guide/1-training/3-logging`。

## 速查表

| 现象 | 解决 |
|------|------|
| `sim=<backend> requires the <backend> extra` | `uv sync --extra <backend>`；Isaac 系需要外部 worker 安装脚本 |
| 编译 mjbatch 时 `Python.h: No such file or directory` | 用 uv 管理的 Python，或安装 `python3-dev` / `python3-devel` |
| Hugging Face 不可达 | `export HF_ENDPOINT=https://hf-mirror.com`；用 `uv run unilab-pull-assets` 预取 |
| socksio not installed | `uv pip install httpx[socks] --python .venv/bin/python` |
| conda HTTP 000 | 按你的代理配置 `conda config --set proxy_servers.*` |
| native H2D 扩展不可用 | 安装 C++ / CUDA Toolkit 依赖，或使用 fallback |
| `Key '...' is not in struct` | override 拼写错误或已移除；用 `--cfg job` 检查 |
| `No owner config exists for ...` | 从支持矩阵选择受支持组合 |
| 后端没有切换 | 用 `--sim`，不要用 `training.sim_backend=` |
| 训练中 NaN | 用 `uv run unilab-viz-nan` 读 guard dump，再看诊断页 |
| 回放时 `CrossBackendIncompatibleError` | 对齐 DENYLIST 字段，或自担风险设 `training.sim2sim_strict=false` |
| IsaacSim 回放崩溃 | 已知阻塞；用 `--render-mode none` |
| `NpEnv` / HIM-PPO / HORA / `uni_rl` 找不到 | 已移除或迁出；见上方迁移条目与 {doc}`/changelog` |
