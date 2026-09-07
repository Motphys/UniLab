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

- UniLab provides environments and adapters. `uni_rl` owns algorithm runners,
  learners, collectors, IPC, and training logs; `uni_rl` must not import UniLab.
  The env factory must be a pickleable `EnvFactory` for spawn collectors.
- Environment reset returns `(obs_dict, info_dict)` and `NpEnvState.obs` is a
  dict. Keep `obs_groups_spec` and policy dimensions consistent with wrappers
  and learners.
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

- Env contract: `src/unilab/base/np_env.py`
- Env factory/device binding: `src/unilab/base/env_factory.py`,
  `src/unilab/base/process_device.py`
- Backend owner: `src/unilab/base/backend_factory.py`
- Training/run helpers: `src/unilab/training/run.py`
- Sim2Sim: `src/unilab/utils/sim2sim.py`
- Architecture and contracts: `docs/sphinx/source/zh_CN/4-developer_guide/0-index.md`
- Workflow policy: `docs/sphinx/source/en/4-developer_guide/5-contributing_workflow.md`
