# 算法

算法页面描述每个内置入口运行的内容、其配置所在位置，以及用哪种命令形式来选择它。
关于通用 flag，请参见 {doc}`../1-training/1-cli_reference`。

| 算法 | 类型 | 入口 | 配置证据 |
| --- | --- | --- | --- |
| PPO | 同步 on-policy | `scripts/train_rsl_rl.py` | `conf/ppo/config.yaml` |
| APPO | 异步 on-policy | `scripts/train_appo.py` | `conf/appo/config.yaml` |
| SAC | off-policy | `scripts/train_offpolicy.py` | `conf/offpolicy/algo/sac.yaml` |
| TD3 | off-policy | `scripts/train_offpolicy.py` | `conf/offpolicy/algo/td3.yaml` |
| FlashSAC | off-policy | `scripts/train_offpolicy.py` | `conf/offpolicy/algo/flashsac.yaml` |
| HIM-PPO | 高度估计器 PPO 路径 | `scripts/train_him_ppo.py` | `conf/ppo_him/config.yaml` |
| HORA | teacher/student 蒸馏路径 | `scripts/train_hora_distill.py` | `conf/hora_distill/config.yaml` |
| MLX PPO | 面向 Apple Silicon 的同步 on-policy | `scripts/train_mlx_ppo.py` | `conf/ppo/config_mlx.yaml` |

```{toctree}
:hidden:

1-ppo
2-appo
3-sac
4-td3
5-flash_sac
6-him_ppo
7-hora
8-mlx_ppo
```
