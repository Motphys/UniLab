# Collaboration Workflow

This page is the repository's source of truth for issue scope, roadmap branches,
pull requests, architecture decisions, and release gates. Stable policy belongs
here; live status, owners, and delivery tracking belong in GitHub issues and PRs.

## Working With Agents

An agent should establish the requested outcome, inspect the owner layer and its
tests/configuration, then make the smallest complete change. It may make routine
implementation choices and continue all authorized, reversible work without
waiting. It asks a focused question when an answer changes product direction,
public contracts, irreversible external actions, or durable maintenance cost.

For a task with independent research, test, log-analysis, or review work, agents
may delegate those read-heavy parts in parallel and return concise evidence to
the main task. Do not use concurrent writers for the same files. The implementing
agent remains responsible for reconciling evidence, validating the final tree,
and reporting the result.

## Work Items

Use one issue for one observable, reviewable result. Include its required code,
configuration, tests, and docs in the same issue when they only have value as a
complete vertical slice. Split a child issue when it has its own acceptance
criteria, owner decision, rollback path, or independently useful result.

Every implementation issue records:

1. The problem and repository evidence.
2. The delivered behavior and explicit scope boundary.
3. Affected owner layers and contracts.
4. Acceptance criteria and focused validation.
5. Dependencies, intended PR base, and material platform/backend impact.

Use a roadmap only for one integrated result that needs multiple independently
reviewable child PRs. A roadmap starts with a short maintainer-facing summary:
goal, recommended path, delivery boundary, estimated review scale, durable
responsibility, and decisions required. Research or benchmark work produces
reproducible evidence and a decision; it does not imply production support.

## Scope Decisions

Development authorization covers the confirmed issue or roadmap boundary. Pause
for a maintainer decision before adding a public contract, execution path,
runner/env lifecycle, synchronization protocol, routine CI, support commitment,
durable benchmark facility, history rewrite, or production subsystem. Also
revisit scope when evidence changes the estimated scale, dependencies, owner
boundary, or makes a smaller/upstream solution viable. Record the decision in
the issue, roadmap, or ADR before extending the work.

## Roadmap Branches

After a roadmap is approved for development, record its declared base and create
`dev/issue-<roadmap-number>-<slug>` from that base's latest head. The base can
be `main` or a parent roadmap integration branch. Create each child branch from
the latest integration branch using a conventional type prefix, such as
`feat/issue-<number>-<slug>` or `fix/issue-<number>-<slug>`, and target its PR
at the integration branch.

Before review, update a child with the current integration branch and validate
the resulting head. Once child PRs are merged, validate the integration head
again and open its final PR to the declared base. Synchronize a moving declared
base at planned integration points, then rerun the relevant gate.

## Pull Requests And CI

Choose and record the intended base before development. Every PR links its
driving issue, explains the delivered behavior and impact, and lists exact
commands run on its final local head. The required local and remote evidence is
defined below; the PR template records it.

| PR base | Required evidence before merge |
| --- | --- |
| `main` | Focused checks and `make test-all` on the final head; all applicable current-head remote CI passes. |
| Any other branch | Focused checks and `make test-all` on the final head; local results and review are the gate. The later PR to `main` runs remote CI. |

Use Conventional Commit titles. State whether the change affects MuJoCo, Motrix,
another backend, macOS, Linux, or training behavior. A documentation-only change
uses its focused docs checks as well as the repository gate required for its PR.

## Architecture Decisions

When work changes a runtime, backend, config, registry, or other public
contract, link the governing ADR in the issue or PR. Add an ADR in the same PR
only for a new structural decision that existing ADRs do not cover. Use the
{doc}`ADR template </adr/ADR-TEMPLATE>` and include status, owners, supersession,
alternatives, repository evidence, and related documents.

## Releases

Release from an approved final commit: update the single version in
`pyproject.toml`, run `make test-all` and `uv build`, and ensure `ci.yml` has a
successful run for that exact commit. Push an annotated `v<version>` tag matching
the project version. The release workflow builds and verifies distributions, then
publishes tagged builds through PyPI trusted publishing. A code change after a
failed release requires a new version and tag; never replace an already published
version.
