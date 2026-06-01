# 为已有任务添加后端 YAML

假设你有一个跑在 Motrix 上的任务，现在想让它也能跑在 MuJoCo 上。本页
就是操作配方。

## 不应该做的事

- 在 Python 中添加后端分支。后端特有的行为属于后端适配层和 owner YAML，
  绝不能写进 env 代码。这是
  {doc}`../../4-developer_guide/2-contracts/2-backend_contract` 的铁律。
- 把 `training.sim_backend` 当作 override 来设置。该字段是 owner YAML 的
  **身份回显（identity echo）**，不是切换开关。

## 应该做的事

1. 复制已有的 owner YAML：
   ```bash
   cp conf/ppo/task/go2_joystick_flat/motrix.yaml \
      conf/ppo/task/go2_joystick_flat/mujoco.yaml
   ```
2. 在新文件里，设置 `training.sim_backend: mujoco`。
3. 调整新后端所需的**物理参数**：
   - 接触摩擦 / 阻尼 —— MuJoCo 对每个接触对使用 solver 参数；Motrix 使用材质
     属性（material props）。确保你位于
     `src/unilab/assets/robots/<robot>/` 下的任务资源文件声明了该后端所需的值。
   - Solver 设置与 timestep —— 把它们放在拥有该行为的 owner YAML 或后端适配层中。
4. 重新核定任何依赖后端的 DR 范围。某些随机化在一个后端上有意义，但在另一个后端上
   是空操作（no-op）（例如摩擦阻尼系数）。
5. 验证该 env/backend 组合已经注册，并且可以通过 registry bootstrap 导入（参见
   {doc}`../../4-developer_guide/1-architecture/5-registry`）。

## 验证闸门

在声称该后端受支持之前，你（至少）需要：

- 一次达到与参考后端相同成功阈值的训练运行，*或者*一份说明它为何达不到的文档
  （例如能力缺口）。
- Reward 一致性检查：参见 {doc}`4-reward_parity`。
- 在 `tests/` 下新增一个测试：用新后端导入该 env，并无错误地运行
  `reset` + `step(10)`。

::::{admonition} 证据等级
:class: note
验证之后，如果支持状态发生了变化，刷新生成的支持数据：

```bash
uv run scripts/generate_support_matrix.py --write
```

证据等级的定义参见 {doc}`/glossary`。
::::
