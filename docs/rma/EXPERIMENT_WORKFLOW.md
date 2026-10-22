# 从一次评估到一份可复查的实验记录

## 先分清三种“完成”

组件验证：RMA 接口、历史缓冲、导出接口、C++ mock 等已有测试。
框架兼容：本地运行副本是 1.3.3，GitHub fork 基线是 1.3.4，整仓类型检查尚未通过。
任务效果：已有教师未达到 0.5 m/s 前进指令。配置能运行，不意味着策略已经学会任务。

后续完整接入应先固定一套源码版本，在独立环境里匹配依赖和资产，完成整仓门禁，
再做真实环境的训练、评估、ONNX/TensorRT 数值与任务效果验证。
不要把新 fork 整个覆盖到现有可运行目录，也不要在现有环境盲目升级所有依赖。
目前仍以原可运行目录的教师作为诊断对象；本次仅增量更新评估追踪与部署映射代码。

## 第一件事：给现有模型记录数据

这是评估，不更新网络权重。训练入口和奖励配置不需要为记录数据而修改。
`--trace-dir` 是可选开关；不开启时不导入追踪器、不创建文件、不读取逐环境奖励。
开启后仅评估阶段多出文件写入和 CPU 拷贝，不能承诺记录本身零开销。

在已经匹配好依赖和资产的原运行目录中执行：

```bash
cd /home/abc/my_unilab
uv run --no-sync python scripts/evaluate_rma.py \
  --policy teacher \
  --checkpoint runs/rma/teacher_forward_v1.pt \
  --task go2_joystick_flat/rma_forward \
  --num-envs 8 --steps 1000 --seed 123 \
  --command 0.5 0 0 --diagnostics \
  --trace-dir runs/rma/evaluations/EXP-002/seed123 \
  --trace-warmup-steps 100
```

每次都会在指定父目录下创建独立的 `eval_<unique>` 子目录，重复运行不覆盖旧记录。
终端 JSON 的 `trace.directory` 给出本次的准确绝对路径。

| 文件 | 保存什么 |
|---|---|
| `transitions.csv` | 每一步、每个环境一行：目标/实际速度、12 维动作、关节相对角度/角速度、奖励分项、回合编号及终止标记 |
| `run_config.json` | 命令参数、checkpoint SHA256、可获取的 Git 信息、关键源码 SHA256、实际控制周期、关节顺序和 scale/offset |
| `task_source_snapshot/*.yaml` | 当次运行目录的任务源配置，包含可用的消融任务；不是完整物化 Hydra 配置，也不包含资产或依赖 |
| `summary.json` | 原评估结果，以及动作范围、动作接近边界比例、全程和启动窗口外的速度误差 |

1000 步 × 8 环境应得到 8000 行数据，另加一行 CSV 表头。
`before_*` 对应动作执行前的状态，`after_*` 对应执行后的终止状态；结束回合时不会误记重置后的姿态。
`episode_id` 与 `episode_step` 将各环境的不同回合分开。

`reward_rate/<name>` 是**已经乘奖励权重、还没有乘控制 dt** 的分项。
`reward` 才是该环境这一步实际返回的总奖励。不要直接把分项 rate 与 reward 比大小。
当前任务使用 dt 缩放时，分项 rate 之和乘 dt 才能与该步 reward 核对；不能跨不同权重直接比较总奖励。

`normalized_action_near_bound_fraction` 统计 `|action| >= 0.95` 的比例。
它只是 tanh 动作接近输出边界的指标，不能证明电机扭矩饱和。

启动窗口外的指标逐回合排除前 100 步；0.02 s 控制周期时对应前 2 s。
这只是约定的分析窗口，不保证系统已经进入稳态；全程指标同时保留。
不足该窗口的回合没有有效样本时，指标为 `null`，不伪造为零误差。

## 第二件事：画图，再对照看数据

将下面的路径换成上一条命令输出的实际 `trace.directory`：

```bash
uv run --no-sync python scripts/plot_evaluation_trace.py \
  /home/abc/my_unilab/runs/rma/evaluations/EXP-002/seed123/eval_<unique>
```

生成 `plots/velocity_tracking.png`、`action_distribution.png`、`reward_components.png`。
速度图显示各环境均值及跨环境标准差；包含重置，不把标准差称作置信区间。
CSV 保留各个环境和回合，可进一步检查平均值是否掩盖个别异常。

本次已经生成的第一份真实记录位于
`/home/abc/my_unilab/runs/rma/evaluations/EXP-002/seed123/eval_hhlzyanz/`。
仓库中另发布了 [压缩数据与图](../../results/rma/diagnostics/EXP-002/seed123/)；
无需仿真环境即可对发布副本运行下面的命令重新画图：

```bash
uv run --no-sync python scripts/plot_evaluation_trace.py \
  results/rma/diagnostics/EXP-002/seed123
```

| 先看什么 | 怎么判断 | 还不能推出什么 |
|---|---|---|
| 指令列 | `command_vx_m_s` 是否一直为 0.5，vy/wz 是否为 0 | 期望值正确不代表动力学/控制正确 |
| 速度列与误差 | `before_vx_m_s` 是否接近 0.5，侧向和转向是否偏离 | 只看总奖励不能代替速度指标 |
| 动作分布 | 是否几乎不变化、偏置很大、频繁接近 ±1 | 任一现象都不是根因的充分证明 |
| 奖励分项 | 哪些项贡献大，跟踪项是否与速度一致 | 分项大不自动说明权重必须减小 |
| 控制映射 | `action_target_joint_order` 与 `joint_observation_order` 是否正确对应 | 两种顺序不同不一定是错误，框架可能已正确映射 |

当前源配置的观测关节和运行时动作目标均按 FL/FR/RL/RR，场景执行器存储按 FR/FL/RR/RL。
追踪文件保存**运行时解析出来的**关节名、索引、scale 与 offset，而不是拿观测顺序猜动作顺序。
在没有 clip/encoder bias 修正的简化情形，关节目标是 `offset + scale * action`，当前配置 scale 为 0.25。
这里记录的是配置映射和观测，不是实际驱动器写入目标/扭矩；进一步确认执行效果需要专门记录那些信号。
不能因为腿编号顺序不一样就直接交换策略输出。

本次追踪发现旧 ONNX 元数据将动作顺序误写成执行器存储顺序，旧 C++ mock 也直接按相同索引相加。
现已修正输出边界的 12 维映射并新增非零 one-hot 测试；`last_action` 仍保持策略原顺序。
旧 JSON 中的 `action_actuator_order` 不能再作为物理存储顺序使用；重新导出后同时记录
`action_joint_order`、`scene_actuator_order` 和 `action_to_actuator_indices`。
此错误影响独立 C++ 路径，不是 Python/UniLab 教师原地踏步的已确认原因。

## 第三件事：只选一个变量做新实验

先用同一模型完成种子 123、456、789 的诊断，再选择最有证据支持的假设。
例如怀疑摆腿辅助奖励干扰前进：复制当前前进任务为新的实验配置，只改
`reward.swing_feet_z.weight` 一项，记录旧值和新值。
新数值应在实验前选定并记录，不能在多个候选中挑最好结果后只公开那一次。

这是下一次重新训练时的配置变化；仅给旧 checkpoint 换奖励做评估不会让它学会新动作。
训练初始化、指令范围、PPO 参数、控制器、训练预算和评估协议保持一致。
新模型使用新文件名，保留旧模型；同样协议下比较实际速度与 RMSE。
不要同一轮又加训练量、又改奖励、又换控制器，那样无法判断是谁产生作用。

首个独立消融现已准备为 [EXP-003](../../experiments/rma/EXP-003_no_pose_penalty.md)：
仅将 `similar_to_default.weight` 从 -0.1 设为 0.0，从头训练并使用新输出文件。
完整命令、日志位置和评估协议见该记录；这是待执行的实验，不是已验证的性能提升。

## GitHub：数据、Issue、PR 各自放什么

| 位置 | 职责 | 更新频率 |
|---|---|---|
| 本地 `runs/rma/...` | 自动保存完整 CSV、checkpoint、运行配置 | 每次运行自动产生；不把所有大文件上传 Git |
| `experiments/rma/EXP-xxx.md` | 假设、唯一变化、命令、数据位置、结果、结论 | 每个有明确问题的实验一份，使用 TEMPLATE.md |
| `results/rma/` | 精选小数据、摘要和曲线，支持复查 | 整理有价值的结果后提交 |
| Issue #1 | “原地踏步”这个问题的调查过程，链接数据/实验/提交 | 有新证据或新结论时追加评论 |
| PR #2 | 当前实验分支相对 main 的代码变化、测试和审查 | 提交并 push 后代码差异自动更新；变更范围/验证结论变化时更新说明 |

在 Issue #1 的 Add a comment 中写进展，再点 Comment 即可。
不需要每次实验都在 Issue 和 PR 粘贴两份相同日志。
同一问题的后续证据继续放 Issue #1；独立新问题才单独开 Issue。
PR 是代码变更的审查入口，不是每次实验都新建或反复粘贴日志的地方。
现在 PR #2 的内容在实验分支，尚未合入 main，查看文件时要选对分支。

一条适合 Issue 的评论模板：

```text
实验：EXP-002，检查原地踏步
假设：待诊断；本次不改算法和奖励
模型：teacher_forward_v1.pt，SHA256 见 run_config.json
协议：0.5/0/0，8 环境 × 1000 步，种子 123/456/789
唯一变化：仅开启评估追踪
数据/配置/代码：填写已提交的链接；本地绝对路径只是自己定位用
结果：填写每个种子的平均 vx、vx RMSE、动作边界比例及关键分项
结论：证据支持什么，仍不能确定什么
下一步：只测试哪个变量，以及为什么
```

“期望它学会走路”是目标；“测量到多少、改了什么、能排除什么”才是实验记录。
