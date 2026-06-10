# 操作

操作任务位于 `src/unilab/envs/manipulation/` 中，Go2 机械臂 manip-loco
env 位于 `src/unilab/envs/locomotion/go2_arm/` 中。

## 手内操作

- `allegro_inhand` 和 `allegro_inhand_grasp` 拥有 MuJoCo 和 Motrix PPO owner。
- `sharpa_inhand`、`sharpa_inhand_grasp` 以及 `sharpa_inhand` 的 `hora`
  配置在当前 config 中都是 MuJoCo owner 路径。

```bash
uv run train --algo ppo --task allegro_inhand --sim mujoco
uv run train --algo ppo --task allegro_inhand --sim motrix training.no_play=true
uv run train --algo ppo --task sharpa_inhand --sim mujoco --profile hora training.no_play=true
```

HORA student 蒸馏由
`conf/hora_distill/task/sharpa_inhand/mujoco.yaml` 配置；它当前未作为
单独的顶层 CLI 路线暴露。

## 平台平衡

`stewart_balance` 是一个 6 自由度并联（Stewart）平台，用于把一个自由小球
平衡在顶部托盘上。策略输出 2 维托盘倾角（roll、pitch）；逆运动学步骤再把该
托盘位姿换算成 6 条移动关节腿的长度，由位置执行器跟踪。奖励由居中项、零速度
推进项与静止奖励组成，并对掉落施加惩罚；回合在小球掉落或持续静止成功时结束。

底座焊接固定在世界系。Motrix 是已验证的训练后端；mujoco owner 可构造并步进，
但其刚性闭环求解器在负载下尚不稳定，暂不能稳定训练。

```bash
uv run train --algo ppo --task stewart_balance --sim motrix training.no_play=true
```

## 移动操作

`go2_arm_manip_loco` 是已提交的 Go2 + Airbot owner 路径：

```bash
uv run train --algo ppo --task go2_arm_manip_loco --sim mujoco training.no_play=true
```

有关任务专属的说明，请参阅 {doc}`../8-manipulation/1-dexterous_inhand` 和
{doc}`../8-manipulation/2-manip_loco`。
