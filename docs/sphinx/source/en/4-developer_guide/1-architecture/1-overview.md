# Architecture Overview

UniLab is a contract-driven robot learning infrastructure repository. The core
rule is to fix behavior at the owner layer that owns the contract.

## Runtime Model

The async paths use a CPU simulation to accelerator learner pipeline:

```text
CPU physics backend -> collector / IPC -> learner
MuJoCo or Motrix      shared memory       torch or mlx
```

PPO and MLX PPO are synchronous single-process paths. APPO and off-policy
algorithms use the async runner, shared buffers, and weight synchronization
primitives under `src/unilab/ipc/` and `src/unilab/algos/`.

## Layer Boundaries

| Layer | Paths | Owns |
| --- | --- | --- |
| Backend | `src/unilab/base/backend/` | `SimBackend`, physics state, optional capabilities |
| Env | `src/unilab/envs/`, `src/unilab/base/np_env.py` | MDP semantics, observation, reward, reset |
| Config and registry | `conf/`, `src/unilab/base/registry.py`, `src/unilab/structured_configs.py` | Schema, owner YAMLs, env/backend registration |
| Algorithms and IPC | `src/unilab/algos/`, `src/unilab/ipc/` | Learners, runners, buffers, weight sync |
| Scripts | `scripts/`, `src/unilab/cli.py` | Thin assembly and CLI routing |

## Design Rules

- Keep backend differences in backend implementations, env adapters, and owner
  YAMLs.
- Use `uv run train --algo <algo> --task <task> --sim <backend>` or
  `uv run eval ...` to select the public algorithm/task/backend route. These
  flags compose the matching owner YAML; `training.sim_backend` is an identity
  field.
- Prefer config over branching. The escalation order for any extension is config
  schema -> registry -> env/backend adapter layer -> and only as a last resort a
  script branch.
- Do not parse XML or assets in hot paths such as `step`, `reset`, or interval
  domain randomization.
- If shared env code needs a backend operation, add it to `SimBackend` before
  using it.
- Make evidence-graded claims. Use grades such as `Registered`, `Configured`,
  `Tested`, `Benchmarked`, or `Recommended`; do not claim stable support without
  evidence in the repo.
- Lift reusable primitives. Shared logic belongs in `src/unilab/base/` or
  `src/unilab/utils/`, not copy-pasted across workflows.
- Validate at the closest boundary to the risk: config tests for Hydra changes,
  env tests for observation/reset changes, IPC tests for runner changes.

## Validation

Validate near the risk. A top-level smoke run supplements, but does not replace,
validation at the boundary a change actually touched.

| Change type | Minimum validation |
| --- | --- |
| Docs only | `uv run pytest tests/scripts/test_check_docs.py -q`, plus manually verify every support claim against the repo |
| Hydra / task / reward config | `make test` (`tests/config/`, `tests/scripts/`) |
| Env contract / observation | `make test` (`tests/base/test_np_env.py` and env tests) plus a 1-iteration smoke run |
| Runner / IPC | `make test`; add `make test-slow` when needed |
| Backend path | the matching backend smoke run, plus a slow test when needed |
| Training entrypoint | the relevant tests plus a 1-iteration smoke run |

Use `make test` for the fast path and `make test-all` (`make check` plus
`make test-cov`) before opening a PR.

## Review Checklist

1. Which contract did this change touch?
2. Should this problem be solved at a lower layer?
3. Is backend or task behavior expressed through config, or hidden by a script
   special-case?
4. Is the support claim backed by registry, config, test, or benchmark evidence?
5. Is validation done at the closest boundary to the risk?

## High-Signal Files

- `scripts/train_rsl_rl.py`
- `scripts/train_mlx_ppo.py`
- `scripts/train_appo.py`
- `scripts/train_offpolicy.py`
- `src/unilab/base/np_env.py`
- `src/unilab/base/backend/base.py`
- `src/unilab/base/registry.py`
- `src/unilab/ipc/async_runner.py`
- `src/unilab/training/run.py`

## Related ADRs

- {doc}`ADR index </adr/ADR-0000-index>`
- {doc}`Runtime model and layer boundaries </adr/ADR-0001-runtime-model-and-layer-boundaries>`
- {doc}`Backend capability boundary </adr/ADR-0002-backend-capability-boundary-for-play-and-snapshot>`
- {doc}`Task owner and config compose contract </adr/ADR-0003-task-owner-and-config-compose-contract>`
- {doc}`Registry bootstrap contract </adr/ADR-0004-registry-bootstrap-contract>`
- {doc}`Unified obs critic env and IPC contract </adr/ADR-0005-unified-obs-critic-env-and-ipc-contract>`
