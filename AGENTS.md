# UniLab Agent Guide

This file contains repository facts and durable constraints. Codex loads it from
the repository root; a more specific `AGENTS.md` or `AGENTS.override.md` may add
or override rules for its subtree. Keep this file short and update it when the
architecture changes.

## Working agreement

- Start by inspecting the relevant owner layer, callers, tests, and config. State
  assumptions when repository evidence is incomplete.
- Infer the requested scope and carry it through to a reviewable result. Ask a
  focused question only when the answer materially changes the outcome; continue
  all independent, authorized work while waiting.
- For a complex task, use parallel subagents for independent read-heavy
  exploration, test/log analysis, or review when that improves the result.
  Keep one writer for an overlapping file set and consolidate evidence before
  editing or reporting.
- Make the smallest complete change that satisfies the requested outcome. Keep
  scripts as orchestration; put durable behavior in its owner module, registry,
  backend adapter, or config.
- Use `rg` for search and `apply_patch` for manual edits. Run Python through
  `uv run`; do not invoke `python` directly.
- Preserve unrelated work in a dirty tree. Do not reset, checkout, or rewrite
  other people's changes.
- Before finishing, report changed files, commands run, results, and any
  unresolved limitation. A failed check is evidence to investigate, not a reason
  to claim success.

## Architecture contracts

- FlashSAC owner YAML enables DP whole-cycle CUDA graphs by default; they require
  synchronizer lifecycle hooks. Explicit
  `algo.algo_params.dp_whole_cycle_cuda_graph=false` disables DP capture;
  `algo.algo_params.use_whole_cycle_cuda_graph=false` disables it for all ranks.
  Capture failure must raise on DP; rank-local eager fallback can hang
  collectives. Release graphs before closing the process group. On the compiled
  CUDA path without GradScaler, loss validity
  travels with gradients so nonfinite losses with finite derivatives skip the
  same optimizer and target update on every rank.
- `training.dp_gradient_wire_dtype` defaults to `float32`. Optional `bfloat16`
  compresses FP32 groups with at least 1024 elements, restores FP32 before
  averaging, and changes numerical precision independently of AMP. Do not claim
  convergence equivalence from short throughput runs. Distributed communication
  remains in `rl/runtime/distributed`, never downstream task packages.
- Training without `CUDA_VISIBLE_DEVICES` defaults to one rank;
  an explicit list selects one rank per GPU. Preserve empty/-1
  masks as CUDA-disabled requests. Put generic minimal launch examples under
  README Train and evaluate, and leave Ecosystem focused on downstream projects.
- CUDA MPS remains manually managed through `uni-cumps`. Start services before
  training and stop only the services you own after clients exit. The default
  `training.cuda_process_sharing=null` uses independent contexts; explicit `mps`
  validates existing services. Do not add automatic MPS lifecycle here.

- UniLab provides environments and adapters, and now also owns the RL
  algorithm and training-execution stack in `unilab.rl` (merged from the
  unilab_rl repo @ 3654e4b). Its layers: `contracts` (env injection and
  observation semantics), `adapters/rsl_rl`, `algos` (APPO/SAC/FlashSAC),
  `runtime` (distributed/ipc/offpolicy), and `logging`. Internal DAG:
  `contracts` depends only on type/tensor libraries; `runtime` must not
  depend on `algos`/`training`/`base`; `algos` must not construct concrete
  envs; the RL core must not import `base`/`envs`/`tasks`/`managers`/
  `training`/`scripts`/`unisim`. These boundaries are guarded by
  `tests/test_library_import_boundary.py`. The PPO path is a direct rsl-rl
  integration (`unilab.rl.adapters.rsl_rl`: VecEnv adapter and distributed
  helpers; runner and algorithm are upstream rsl-rl). The env factory must
  be a pickleable `EnvFactory` for spawn collectors. The `unilab.rl`,
  `rl.runtime`, and `rl.runtime.distributed` `__init__` modules stay
  lightweight (PEP 562 lazy loading): importing them must not pull in torch
  or algorithm code.
- Environment reset returns `(obs_dict, info_dict)` and `TorchEnvState.obs` is a
  dict of tensors. Keep `obs_groups_spec` and policy dimensions consistent with
  wrappers and learners.
- The Manager-Based runtime is the sole task runtime; do not restore legacy
  class-config factory seams or compatibility layers.
- Manager-Based event terms negotiated through `SimBackend` capabilities are
  the sole DR lifecycle; do not reintroduce a DR manager/provider protocol, and
  unsupported capabilities fail closed.
- Backend-specific behavior belongs behind the declared `unisim.backend.base.SimBackend`
  interface. Extend that interface before consuming a capability in an env;
  never probe or call backend-private methods from env or training code.
- Task, reward, and backend selection belongs in Hydra owner YAML and registries.
  Select a backend with the task owner config/CLI; do not use
  `training.sim_backend` as a standalone backend switch.
- Asset/XML metadata is cold-path work only: init, materialization, or cache.
  Step/reset/randomization must not parse assets or branch on asset metadata.
  Robot meshes and textures come from the registered asset hub and stay out of
  git. Task keyframes belong in task/scene XML fragments, never `robot.xml`.
- Do not add new owner logic to `src/unilab/utils/`; those modules are transition
  shims. Keep cross-cutting changes in the owning package.

## API naming rules

Approved in roadmap #2434; enforce in review.

- Type/device qualifiers (`_tensor`, `_numpy`, `cpu_`, `torch_`) only when two
  real implementations coexist in one scope; when needed, use `_tensor`.
- Module names carry the qualifier; class/function names do not repeat it.
- Package-internal functions do not carry an engine/package-name prefix.
- Names consumed across modules never use a leading underscore; promote or
  document them as protected protocols instead.
- One public name per meaning; no aliases or deprecation fallbacks.
- No time-based names (`legacy`, `old`, `new`, `v2`, `tmp`) on live APIs.

## Sim2Sim contract

`src/unilab/utils/sim2sim.py` is the runtime source of truth. DENYLIST fields
must match across backends (strict by default), WARNING_LIST differences are
reported, and ALLOWLIST fields may vary. Training snapshots the contract in
`run_config.json`; play entrypoints validate before environment construction and
guard checkpoint dimensions. Update the contract and its tests together when a
policy-I/O or network-shape field changes.

## Validation

Use the smallest relevant check while iterating, then run the complete gate for
the final change:

```text
make check                 # formatting and type checks
make test                  # non-slow tests
make test-all              # required before creating/updating a PR
```

Dependency pin changes must keep the ROCm track in sync: apply the same bump
to `pyproject.rocm.toml` and regenerate `uv.rocm.lock` in the same PR.

Use focused tests for contract, IPC/runner, config, backend, asset, or docs
changes. Docs-only validation is defined in `docs/sphinx/AGENTS.md`. Record the
exact commands and outcomes in the PR. Do not run expensive benchmarks unless
the requested result or acceptance criteria needs them.

## Collaboration and review

The single source for issue, roadmap, branch, PR, ADR, and CI policy is
`docs/sphinx/source/en/4-developer_guide/5-contributing_workflow.md`; the Chinese
page is its maintained translation. Read that page before changing workflow
policy. In brief: define a reviewable outcome, choose the PR base before coding,
link the driving issue, validate the final head, and wait for current-head remote
CI only when the PR base is `main`.

When a change crosses runtime, backend, config, registry, or other public
contracts, link the relevant ADR. Add an ADR only for a new structural decision.
Treat new public contracts, lifecycle/protocol changes, routine CI, support
claims, durable benchmarks, or productionization as scope decisions and record
them in the issue/ADR before expanding the work.

## Useful pointers

- Env contract: `src/unilab/base/torch_env.py`
- Env factory/device binding: `src/unilab/base/env_factory.py`,
  `src/unilab/base/process_device.py`
- Backend owner: `src/unilab/base/backend_factory.py`
- Training/run helpers: `src/unilab/training/run.py`
- PPO (direct rsl-rl integration): `src/unilab/rl/adapters/rsl_rl/`
- RL algorithms (APPO/SAC/FlashSAC): `src/unilab/rl/algos/`
- FlashSAC custom actor contract (actor_config injection, aux losses,
  checkpoint semantics): `src/unilab/rl/algos/flash_sac/actor_spec.py`
- RL runtime (distributed/ipc/offpolicy): `src/unilab/rl/runtime/`
- Sim2Sim: `src/unilab/utils/sim2sim.py`
- Architecture and contracts: `docs/sphinx/source/zh_CN/4-developer_guide/0-index.md`
- Workflow policy: `docs/sphinx/source/en/4-developer_guide/5-contributing_workflow.md`
