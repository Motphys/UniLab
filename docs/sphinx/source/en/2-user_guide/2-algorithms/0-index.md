# Algorithms

Algorithm pages describe what each checked-in entrypoint runs, where its config
lives, and which command shape selects it. For general flags, see
{doc}`../1-training/1-cli_reference`.

| Algorithm | Style | Entrypoint | Config Evidence |
| --- | --- | --- | --- |
| PPO | synchronous on-policy | `scripts/train_rsl_rl.py` | `conf/ppo/config.yaml` |
| APPO | async on-policy | `scripts/train_appo.py` | `conf/appo/config.yaml` |
| SAC | off-policy | `scripts/train_offpolicy.py` | `conf/offpolicy/algo/sac.yaml` |
| TD3 | off-policy | `scripts/train_offpolicy.py` | `conf/offpolicy/algo/td3.yaml` |
| FlashSAC | off-policy | `scripts/train_offpolicy.py` | `conf/offpolicy/algo/flashsac.yaml` |
| HIM-PPO | height-estimator PPO path | `scripts/train_him_ppo.py` | `conf/ppo_him/config.yaml` |
| HORA | teacher/student distillation path | `scripts/train_hora_distill.py` | `conf/hora_distill/config.yaml` |
| MLX PPO | synchronous on-policy for Apple Silicon | `scripts/train_mlx_ppo.py` | `conf/ppo/config_mlx.yaml` |

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
