# 从 unilab-rl（`uni_rl`）迁移

本页面向曾安装 `uni_rl` extra（`pip install unilab[uni_rl]`）或直接
`import uni_rl` 的用户。

原 `unilab-rl` distribution（import namespace `uni_rl`，仓库
[unilabsim/unilab_rl](https://github.com/unilabsim/unilab_rl)）已并入本
仓库的 `unilab.rl`（issue #2372，决策见
{doc}`/adr/ADR-0016-rl-reintegration-into-unilab-rl`）。发行面收敛为单一
`unilab` distribution：APPO / SAC / FlashSAC runner、异步 IPC runtime
与 RL logging 栈均随基础安装提供。

## 安装变化

- 从 `uv sync` 命令中去掉 `--extra uni_rl`，pip 安装中去掉
  `unilab[uni_rl]`；RL 栈不需要任何 extra。
- 空的过渡期 `uni_rl` extra 已移除，旧命令会因 unknown extra 报错。请从
  脚本和 CI 中删除该 extra。
- 从环境中卸载独立的 `unilab-rl`（`pip uninstall unilab-rl`），避免旧
  实现残留并与 `unilab.rl` 并存。
- CLI 算法名、日志路径、checkpoint 字段与 metric schema 均不变。

## Import 路径映射

| 旧 import（`uni_rl.*`） | 新 import |
|---|---|
| `uni_rl.env_contract` | `unilab.base.contracts.env` |
| `uni_rl.training_state` | 无替代（checkpoint 保存/加载由 `unilab.training.experiment` 负责） |
| `uni_rl.utils.observations` | `unilab.base.contracts.observations` |
| `uni_rl.algos.common.device`（probe） | `unilab.base.contracts.probe` |
| `uni_rl.algos.appo` / `.sac` / `.flash_sac` / `.common` | `unilab.rl.algos.*`（子包名不变） |
| `uni_rl.algos.common.collector_timing` | `unilab.rl.runtime.collector_timing` |
| `uni_rl.ipc.async_runner` | `unilab.rl.runtime.async_runner` |
| `uni_rl.ipc.dp_launcher` | `unilab.rl.runtime.distributed.context` / `.launcher` / `.affinity`；Hydra 侧的 `apply_dp_rank_config` 移到 `unilab.training.dp_config` |
| `uni_rl.ipc.dp_sync` | `unilab.rl.runtime.distributed.sync` |
| `uni_rl.ipc.*`（其余模块） | `unilab.rl.runtime.ipc.*` |
| `uni_rl.offpolicy.*` | `unilab.rl.runtime.offpolicy.*` |
| `uni_rl.logging.*` | `unilab.rl.logging.*` |
| `uni_rl.utils.seed` | `unilab.rl.runtime.seed`（`apply_worker_seed`、`derive_worker_seed`；`TrainingSeedInfo`、`resolve_training_seed` 与父进程 `apply_training_seed` 仍在 `unilab.training.seed`） |
| `uni_rl.utils.nan_guard` | `unilab.training.tensor_diagnostics`（`NanGuard`）；runner/collector 现在接收注入的 factory |
| `uni_rl.utils.tensor` | 无替代（tensor-native collector 不再使用 host 转换辅助函数） |
| `uni_rl.utils.device` | `unilab.rl.runtime.device` |
| `uni_rl.utils.tensor_runtime` | `unilab.rl.runtime.settings` / `unilab.rl.runtime.placement`（纯设置与放置）；Hydra resolver 在 `unilab.training.tensor_runtime` |

`unilab.rl` 的包级公开面（`from unilab.rl import RslRlVecEnvAdapter`、
`UNILAB_DP_LOG_DIR`、`launch_torchrun_workers` 等
`src/unilab/rl/__init__.py` 中列出的名称）不变，经惰性加载解析。

不承诺永久支持 `import uni_rl`，请迁移到 `unilab.rl.*` 路径。

## 行为变化

- SAC / FlashSAC runner builder（`unilab.rl.algos.sac.double_buffer` 与
  `unilab.rl.algos.flash_sac.double_buffer` 中的 `build_double_buffer_runner`）
  改为注入式：训练装配层传入
  `runtime_settings`、`inference_placement`、
  `backend_device_resolver`、`backend_device_configurer`；旧
  `backend_device_binder` 参数已移除。缺省时使用 runner 侧默认值。
- 扩展机制不变：`runtime_resolver`、`class_name`、`actor_adapter_modules`
  与 actor registry 继续有效，指向外部包的 dotted path 保持原值；只有内置
  dotted path 移到了 `unilab.rl.*` namespace。
- "uni_rl 不得 import UniLab" 的跨包边界不再适用；RL 栈改为遵守
  `unilab.rl` 的内部分层规则（RL 核心不依赖 env/training 层），分层 DAG 见
  {doc}`/adr/ADR-0016-rl-reintegration-into-unilab-rl`。
