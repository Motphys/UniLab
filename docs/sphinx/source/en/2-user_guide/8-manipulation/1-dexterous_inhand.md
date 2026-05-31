# Dexterous In-Hand Manipulation

This page covers the checked-in Allegro and Sharpa in-hand manipulation paths.
Select backends with `--task` and `--sim`; do not override
`training.sim_backend` alone. The owner YAMLs remain the internal evidence for
which combinations are configured.

## Allegro

Allegro rotation uses the registered env `AllegroInhandRotation`. The rotation
owner is `allegro_inhand`, and grasp-cache generation uses
`allegro_inhand_grasp`.

Owner evidence:

- `conf/ppo/task/allegro_inhand/mujoco.yaml`
- `conf/ppo/task/allegro_inhand/motrix.yaml`
- `conf/ppo/task/allegro_inhand_grasp/mujoco.yaml`
- `conf/ppo/task/allegro_inhand_grasp/motrix.yaml`
- `conf/appo/task/allegro_inhand/mujoco.yaml`
- `conf/appo/task/allegro_inhand/motrix.yaml`

The typical flow is two stages: first generate a grasp cache, then train the
rotation policy.

```bash
uv run train --algo ppo --task allegro_inhand_grasp --sim mujoco training.no_play=true
uv run train --algo ppo --task allegro_inhand --sim mujoco training.no_play=true
```

Motrix owner YAMLs also exist for the PPO Allegro paths:

```bash
uv run train --algo ppo --task allegro_inhand_grasp --sim motrix training.no_play=true
uv run train --algo ppo --task allegro_inhand --sim motrix training.no_play=true
```

The rotation owner defaults to the grasp cache at `cache/allegro_grasp_50k.npy`.
To use a custom cache, override `env.grasp_cache_path`:

```bash
uv run train --algo ppo --task allegro_inhand --sim mujoco \
  env.grasp_cache_path=cache/my_allegro_grasp.npy
```

Replay a trained checkpoint with `eval` (`--load-run -1` picks the latest run):

```bash
uv run eval --algo ppo --task allegro_inhand --sim mujoco --load-run -1
uv run eval --algo appo --task allegro_inhand --sim mujoco --load-run -1
```

## Sharpa

Sharpa rotation uses the registered env `SharpaInhandRotation`. Current checked
in training paths are MuJoCo owner paths.

Owner evidence:

- `conf/ppo/task/sharpa_inhand/mujoco.yaml`
- `conf/ppo/task/sharpa_inhand/mujoco_hora.yaml`
- `conf/ppo/task/sharpa_inhand_grasp/mujoco.yaml`
- `conf/appo/task/sharpa_inhand/mujoco.yaml`
- `conf/appo/task/sharpa_inhand/mujoco_hora.yaml`
- `conf/hora_distill/task/sharpa_inhand/mujoco.yaml`

The full HORA path is three stages:

1. Generate the grasp cache.
2. Train the teacher policy.
3. Distill a student policy when needed.

The full HORA teacher/student path is MuJoCo-owner-primary. The Motrix path
currently covers only phase-1 PPO rotation and grasp-cache collection; it is not
a full HORA capability-equivalent path.

### Grasp cache and scale

The default caches are hosted on Hugging Face (`unilabsim/unilab-caches`) and are
downloaded automatically into `src/unilab/assets/caches/` on first training, so
no manual step is needed.

Sharpa rotation samples from per-scale grasp caches, so multi-scale caches are
collected by running the grasp task once per scale (cache files are named
`<prefix>_<scale>.npy`):

```bash
uv run train --algo ppo --task sharpa_inhand_grasp --sim mujoco 'env.domain_rand.scale_list=[0.8]' training.no_play=true
uv run train --algo ppo --task sharpa_inhand_grasp --sim mujoco 'env.domain_rand.scale_list=[1.0]' training.no_play=true
uv run train --algo ppo --task sharpa_inhand_grasp --sim mujoco 'env.domain_rand.scale_list=[1.2]' training.no_play=true
```

Or use the helper script, which collects each scale sequentially:

```bash
bash scripts/sharpa_collect_grasps.sh 0.8 1.0 1.2
```

Motrix can also collect a grasp cache (phase-1 scope only):

```bash
uv run train --algo ppo --task sharpa_inhand_grasp --sim motrix \
  'env.domain_rand.scale_list=[1.0]' \
  env.grasp_collection_target=1000 \
  training.no_play=true
```

To use a custom cache prefix, override `env.grasp_cache_path`:

```bash
uv run train --algo ppo --task sharpa_inhand --sim mujoco \
  env.grasp_cache_path=cache/my_sharpa_grasp_cache
```

### Teacher and student

Train the HORA teacher with the `hora` profile (PPO or APPO):

```bash
uv run train --algo ppo --task sharpa_inhand --sim mujoco --profile hora training.no_play=true
uv run train --algo appo --task sharpa_inhand --sim mujoco --profile hora training.no_play=true
```

Replay a teacher run with `eval --profile hora --load-run -1`:

```bash
uv run eval --algo ppo --task sharpa_inhand --sim mujoco --profile hora --load-run -1
uv run eval --algo appo --task sharpa_inhand --sim mujoco --profile hora --load-run -1
```

Student distillation is configured by
`conf/hora_distill/task/sharpa_inhand/mujoco.yaml` and implemented by
`scripts/train_hora_distill.py`; the top-level CLI does not currently expose a
separate HORA distillation route (it is not in the CLI `SUPPORTED_ALGOS`). To
distill from an APPO teacher, set `teacher.algo_family=appo` in that low-level
config.

Common log directories:

- `logs/hora_ppo/SharpaInhandRotation/`
- `logs/hora_appo/SharpaInhandRotation/`
- `logs/hora_distill/SharpaInhandRotation/`

The scale / grasp-cache / DR boundary is sensitive here; see
{doc}`../5-domain_randomization/0-index` for the lifecycle rules.

For the category-level task page, see {doc}`../4-tasks/3-manipulation`.
