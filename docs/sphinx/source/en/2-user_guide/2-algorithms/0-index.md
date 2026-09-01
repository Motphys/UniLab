# Algorithms

Algorithm pages describe what each checked-in entrypoint runs, where its config
lives, and which command shape selects it. For general flags, see
{doc}`../1-training/1-cli_reference`.

| Algorithm | Style | Entrypoint | Config Evidence |
| --- | --- | --- | --- |
| PPO | synchronous on-policy | `src/unilab/scripts/train_rsl_rl.py` | `src/unilab/conf/ppo/config.yaml` |
| APPO | async on-policy | `src/unilab/scripts/train_appo.py` | `src/unilab/conf/appo/config.yaml` |
| SAC | off-policy | `src/unilab/scripts/train_sac.py` | `src/unilab/conf/sac/config.yaml` |
| TD3 | off-policy | `src/unilab/scripts/train_td3.py` | `src/unilab/conf/td3/config.yaml` |
| FlashSAC | off-policy | `src/unilab/scripts/train_flashsac.py` | `src/unilab/conf/flashsac/config.yaml` |
| HIM-PPO | height-estimator PPO path | `scripts/train_him_ppo.py` | `src/unilab/conf/ppo_him/config.yaml` |
| HORA | teacher/student distillation path | `scripts/train_hora_distill.py` | `src/unilab/conf/hora_distill/config.yaml` |

```{toctree}
:hidden:

1-ppo
2-appo
3-sac
4-td3
5-flash_sac
6-him_ppo
7-hora
```
