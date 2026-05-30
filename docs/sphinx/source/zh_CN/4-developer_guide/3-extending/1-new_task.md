# 扩展 UniLab：新任务

语言: 简体中文

从契约出发：{doc}`../2-contracts/1-env_contract`、
{doc}`../2-contracts/3-task_owner` 与
{doc}`/adr/ADR-0005-unified-obs-critic-env-and-ipc-contract`。

## 实现清单

1. 在 `src/unilab/envs/` 下选取最接近的 owner 包。
2. 定义或扩展一个 env config dataclass，并用
   `@registry.envcfg("EnvName")` 注册它。
3. 用 `@registry.env("EnvName", sim_backend="mujoco")` 或
   `@registry.env("EnvName", sim_backend="motrix")` 注册每个受支持的
   后端实现。
4. 如果任务位于一个新模块中，请把该模块加入包的
   `__unilab_registry_modules__` 元组，以便 `ensure_registries()` 导入它。
5. 保持 `obs_groups_spec` 准确。它必须包含 `obs`，并且可以包含
   `critic`；wrapper 和 learner 都信任这些维度。
6. 把 reset 与 step 语义保留在 env owner 层：
   `reset(env_indices)` 返回 `(obs_dict, info_dict)`，而 `step(actions)`
   返回 `NpEnvState`。
7. 在相关 config 根目录下添加 owner YAML，例如
   `conf/ppo/task/<task>/<backend>.yaml` 或
   `conf/offpolicy/task/<algo>/<task>/<backend>.yaml`。
8. 把任务或场景的 keyframe 放进通过 `SceneCfg.fragment_files` 引用的
   任务/场景 XML fragment；不要把 task-level keyframe 放进 `robot.xml`。

## 在风险点附近验证

- Registry 与 config 形状：`tests/base/test_registry.py`、
  `tests/config/test_config_system.py`
- Env observation/reset 行为：`tests/base/test_np_env.py` 以及
  `tests/envs/` 下最接近的特定任务测试
- 脚本组合：`tests/scripts/test_train_script_configs.py`

## 仓库内证据

- Registry API：`src/unilab/base/registry.py`
- Env 状态契约：`src/unilab/base/np_env.py`
- 场景配置：`src/unilab/base/scene.py`
- 现有任务示例：`src/unilab/envs/locomotion/go2/joystick.py`、
  `src/unilab/envs/manipulation/allegro_inhand/rotation.py`

## Navigation

- Index: [文档](0-index.md)
