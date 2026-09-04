# Extending UniLab: New Algorithm

Algorithm work must preserve the env, config, and runner contracts. Start with
{doc}`../2-contracts/1-env_contract`, {doc}`../2-contracts/3-task_owner`, and
{doc}`../2-contracts/5-runner_lifecycle`.

## Three Extension Tiers

Ordered from shallowest to deepest; prefer the shallowest tier that meets your
needs.

### 1. Pure Config: Reuse An Existing Algorithm

Change no code — only adjust the Hydra config under `src/unilab/conf/<algo>/`:

- Algorithm hyperparameters are inlined in `src/unilab/conf/<algo>/config.yaml`;
- Each task×backend combination maps to one owner YAML:
  `src/unilab/conf/<algo>/task/<task>/<backend>.yaml`;
- Swapping policy / algorithm implementation classes works through the
  `class_name` dotted path in the owner YAML (in-repo example:
  `uni_rl.algos.hora:HoraActorModel`), with no new code path required.

### 2. `runtime_resolver`: Algorithm Code In Your Own Repository

A researcher imports `uni_rl` (distribution name `unilab-rl`) as a library in
their own repository and implements the runner / learner / play logic there —
**no fork of unilab_rl needed**. The owner YAML declares a dotted path
(`module:attr`) via `algo.runtime_resolver`:

```yaml
algo:
  runtime_resolver: my_pkg.my_module:resolve_my_runtime
```

Contract:

- Signature is `(rl_cfg: dict) -> Runtime | None`; returning `None` falls back
  to the algorithm's default runtime;
- The returned object must carry `runner_cls`; depending on the algorithm
  family it may optionally carry `play_fn` (APPO style) or `wrapper_cls`
  (PPO style);
- Resolution happens on the uni_rl side (`uni_rl.algos.appo.runtime` /
  `uni_rl.algos.rsl_rl_runtime`); the dotted path may point at any importable
  module.

In-repo example: `src/unilab/conf/appo/task/sharpa_inhand/mujoco_hora.yaml`
points at `unilab.scripts.play_hora_appo:resolve_hora_appo_runtime`; the HORA
SAC / PPO variants point at `uni_rl.algos.hora.sac:resolve_hora_sac_runtime`
and `uni_rl.algos.hora.rsl_rl:resolve_hora_ppo_runtime` respectively.

### 3. Fork unilab_rl: Modify `uni_rl/algos/`

Only when you need to change the shared runner / learner / collector
implementation (for example a new IPC lifecycle) should you fork
[unilabsim/unilab_rl](https://github.com/unilabsim/unilab_rl) and modify
`uni_rl/algos/`. Async algorithms should reuse `AsyncRunner`, `ReplayBuffer` /
`RolloutRingBuffer`, and `SharedWeightSync` instead of creating a new IPC
lifecycle.

## Footprint Of A CLI-Routable Algorithm

`uv run train --algo <algo>` / `uv run eval --algo <algo>` use
convention-based routing (`available_algos` and `build_route` in
`src/unilab/cli.py`): an algo name `<algo>` is routable if and only if both of
the following exist — **no cli.py change required**:

1. `src/unilab/conf/<algo>/config.yaml` — the Hydra config root with the
   algorithm hyperparameters inlined;
2. `src/unilab/scripts/train_<algo>.py` — the entrypoint script, kept as a
   thin assembly shell: compose Hydra, call `ensure_registries()`, construct
   the env through the registry path, then hand control to the runner or
   trainer. Thin-shell precedent: `train_sac.py` / `train_td3.py` /
   `train_flashsac.py` reuse the shared implementation in `train_offpolicy.py`.

On top of that, each task×backend combination needs an owner YAML at
`src/unilab/conf/<algo>/task/<task>/<backend>.yaml`. Unknown algos fail
closed, and the error message lists every available algo (built-in plus
convention-discovered).

Notes:

- Config trees that have a conf directory but no entrypoint script (such as
  `hora_distill` and `ppo_him`) are not routable — they are not standalone
  CLI algos.
- The special script-name mappings for built-in algorithms are preserved:
  `ppo` → `train_rsl_rl.py`, `appo` → `train_appo.py`.
- The dataclasses in `src/unilab/structured_configs.py` are an **optional**
  conventional mirror; there is no ConfigStore enforcement, so a new algorithm
  is not required to add one.

## Implementation Checklist

1. Pick the integration path from the three tiers above: prefer pure config
   over code, and `runtime_resolver` over a fork.
2. Keep third-party adapter naming at adapter boundaries. Do not change the
   internal `obs` plus optional `critic` env contract to match a library.
3. Keep new entrypoint scripts as assembly; no long-term business rules in
   `scripts/`.

## Validation Near Risk

- CLI routing and convention discovery: `tests/test_cli.py`
- Script/config tests: `tests/scripts/test_train_script_configs.py`,
  `tests/scripts/test_train_scripts.py`

## Evidence In Repo

- Structured config dataclasses: `src/unilab/structured_configs.py`
- Training helpers: `src/unilab/training/common.py`,
  `src/unilab/training/run.py`
- Existing algorithm packages: `uni_rl` in
  [unilabsim/unilab_rl](https://github.com/unilabsim/unilab_rl)
