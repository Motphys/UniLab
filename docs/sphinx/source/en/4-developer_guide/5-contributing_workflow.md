# Collaboration Workflow


Repository documentation only records stable standards. Execution status, owners, and stage progression belong in GitHub collaboration objects.

If you just want to install or train UniLab, start with
{doc}`/en/1-getting_started/2-installation` and
{doc}`/en/1-getting_started/1-quick_demo`.

## Work Item Granularity

Each issue should at minimum answer these questions:

1. What problem are we solving?
2. What is the expected deliverable?
3. What is the completion criterion?
4. Who is responsible for execution?
5. What upstream blockers exist?

Recommended issue types:

- `bug`
- `work item`: feature / infra / benchmark / test / sim / docs work

## AI Roadmap, Issue, And Branch Governance

This section carries forward the focus of [discussion
#883](https://github.com/unilabsim/UniLab/discussions/883) on project direction,
informed decisions, and durable maintenance ownership. It organizes multi-PR
delivery around clear decisions, useful issue boundaries, an integration
branch, and base-aware CI routing. It applies to roadmaps, architecture
issues, and multi-PR plans proposed by AI agents.

### Communication Baseline

- UniLab maintainers own product direction and long-term maintenance choices,
  using concrete code, configuration, data-flow, and performance evidence.
- The roadmap author translates the proposal into existing repository concepts.
  Describe what changes in the repository before introducing an abstraction.
- Introduce each new concept with the problem it solves, the existing modules it
  affects, its durable responsibilities, and one repository example.
- AI review, tests, benchmarks, and gates provide implementation evidence. The
  owner summary and explicit maintainer choices record product judgment.
- When more context is needed, restate the proposal with shorter language,
  concrete paths, and real choices until the delivery boundary is shared.

### Value And Minimum Complete Design

A roadmap first answers these questions directly:

1. Which UniLab core goal does this work serve?
2. Which current code, configuration, tests, bugs, or benchmarks establish the
   opportunity?
3. What is the smallest complete design, and how much can reuse the current
   owner layer or upstream capability?
4. What are the expected benefit, opportunity cost, and priority basis?
5. Which contracts, execution paths, configurations, tests, CI, or support
   responsibilities remain after merge?

While evidence is still developing, frame the work as research, a benchmark,
or an adapter case study and record the conditions for a later
production/support decision. Each support level then stays aligned with
repository evidence.

### Roadmap Content

Organize a roadmap in two layers:

1. **Owner summary**: a brief plain-language summary, roughly 150 words,
   covering the goal, recommended design, delivery boundary, estimated scale,
   durable maintenance, and maintainer decisions.
2. **Technical detail**: after direction is confirmed, describe owner
   boundaries, data flow, dependencies, risk, validation, child issues, and
   integration order. Types, method names, state machines, and performance plans
   map to confirmed delivery needs.

The owner summary should let a maintainer restate:

> What this roadmap delivers, where its boundary lies, and what the repository
> will maintain afterward.

Use these writing principles:

- Put the recommended design before background, terminology, and architecture
  detail.
- When a choice is needed, provide two or three real options and state user
  value, implementation scale, and long-term cost for each.
- Pair each new abstraction with a repository example and the tradeoff of using
  the current structure.
- Specify near-term child issues as executable, verifiable outcomes. Record
  farther work as direction, dependencies, and start conditions that evolve
  with evidence.
- Record one integrated outcome, the declared base branch, child-issue list,
  dependency order, and final acceptance. Add the integration branch after
  development authorization.

### Let Delivery Boundaries Set Issue Size

Issue size serves understanding, review, and delivery efficiency:

| Type | Primary purpose | Delivery path |
| --- | --- | --- |
| Roadmap | Define an integrated result, key decisions, and acceptance boundary that require multiple PRs | Collect child PRs on an integration branch, then merge it back to the roadmap's declared base |
| Implementation | Deliver one observable, reviewable result with the code, config, tests, and docs that complete it | Usually one focused PR |
| Research / Benchmark | Produce reproducible evidence and a clear decision result | Independently reviewable artifacts that inform later implementation |

An implementation issue is a complete vertical slice. Keep these together:

- the contract, owner implementation, configuration, tests, and user docs for
  one behavior;
- adjacent owner-layer changes required to preserve an end-to-end runnable path;
- migration steps whose acceptance value appears only when combined;
- benchmarks, compatibility work, and cleanup required to accept the primary
  outcome.

Create a child issue when a component has one of these properties:

- it independently produces user or repository value with its own acceptance;
- it has a separate architecture choice, risk decision, review owner, or
  delivery cadence;
- it can be reverted independently and collaborates through a stable interface;
- parallel delivery materially shortens the cycle and has clear dependencies.

File count, handwritten LOC, directory count, owner-layer count, and PR count
are planning signals for review effort and scheduling. Define issue boundaries
by independent delivery value. When an estimate changes materially, update the
issue's scale, dependencies, and review plan, then choose the most coherent
delivery boundary. A helper, one config, a test group, or supporting docs
normally stay with the primary outcome they serve.

### Writing An Implementation Issue

Keep the issue body concise, concrete, and directly executable. Longer research
notes, interface drafts, and benchmark data can live in an ADR, document, or
attachment. Include as applicable:

1. **Problem and evidence**: the opportunity or gap and its repository facts.
2. **Primary deliverable**: the complete capability available after merge.
3. **Delivery boundary**: work included here and related independently delivered
   work.
4. **Owner and contract impact**: primary owner layer, adjacent boundaries, and
   durable responsibilities.
5. **Roadmap relationship and target branch**: parent roadmap, roadmap declared
   base, dependencies, and PR base.
6. **Scale and review plan**: estimated files, handwritten LOC, PR organization,
   and appropriate reviewers.
7. **Acceptance and validation**: observable results, focused tests, and required
   benchmarks.
8. **Scope-review points**: discoveries that call for an updated design or a
   maintainer choice.

Present product choices explicitly in the owner summary and make technical
detail serve the confirmed boundary. Durable CI, evidence, and support
facilities correspond to reusable long-term needs. Record future ideas with
their start conditions. AI review conclusions follow the maintainer decision as
supporting evidence.

### Roadmap Integration Branch Workflow

After explicit roadmap development authorization:

1. Record the declared base branch in the roadmap issue. It may be `main` or a
   parent roadmap's integration branch. Create and push
   `dev/issue-<roadmap-number>-<slug>` from that base's latest head, following
   the repository's existing `dev/issue-*` convention.
2. Create each child-issue branch from the latest integration branch. Use a
   change-type prefix such as `feat/issue-<number>-<slug>`,
   `fix/issue-<number>-<slug>`, `refactor/issue-<number>-<slug>`,
   `perf/issue-<number>-<slug>`, `test/issue-<number>-<slug>`, or
   `docs/issue-<number>-<slug>`.
3. Align the child branch's final review head with the current integration
   branch, run focused tests plus local `make test-all`, and record the commands
   and results in the PR.
4. Set the child PR base to this roadmap's integration branch. Merge after the
   local gate and review pass. These PRs use local validation; remote execution
   belongs to a PR whose actual base is `main`.
5. Continue through approved child issues in dependency order, honoring product
   checkpoints recorded by the roadmap.
6. After all child issues are integrated, run `make test-all` again on the
   integration branch's latest head and open the final PR from
   `dev/issue-...` back to the declared base.
7. Route the final PR by its actual base: a `main` base waits for current-head
   remote CI; another base uses the local gate and review, with remote validation
   provided by the later PR that reaches `main`. After merge, clean up this
   roadmap's integration and child branches according to normal repository
   maintenance practice.

When the declared base advances during roadmap development, synchronize it at
planned integration points and rerun the local gate on the updated head.
Parallel child issues synchronize the current integration branch before review
so that local results cover the actual merge candidate.

### Authorization And Scope Review

Authorization has two explicit stages:

- **Planning authorization** covers drafting the roadmap, creating issues,
  researching evidence, and organizing child issues while repository
  implementation stays unchanged.
- **Development authorization** covers one ordinary implementation issue, or
  continuous delivery of a roadmap's confirmed boundary and child-issue set
  through its integration branch.

Treat public contracts, execution paths, runner/env lifecycles, training paths,
synchronization protocols, routine CI, support levels, durable
benchmark/evidence facilities, history rewrites, and promotion of an adapter to
a production subsystem as explicit roadmap decisions. Development authorization
covers decisions already confirmed in the owner summary. When implementation
introduces another such decision, update the owner summary, impact, and durable
cost for maintainer confirmation.

Review the scope when:

- actual scale, dependencies, or durable responsibilities change materially
  from the issue estimate;
- backend work reaches env, manager, runner, or learner contracts outside the
  confirmed boundary;
- test results identify a need for a new abstraction or durable facility;
- upstream reuse or a smaller design now satisfies the primary result;
- the maintainer needs a more concrete path, call chain, or tradeoff description.

A scope review presents current facts, a recommendation, and impacts so the
maintainer can continue, adjust, split, or conclude the work. Record that choice
in the roadmap or implementation issue and continue from the updated boundary.

## Milestone Structure

Each milestone should:

- Exist as a milestone object in GitHub
- Have a tracking issue that aggregates sub-issues
- Keep execution details in the sub-issues, not in the milestone description
- Define completion by delivered artifacts, not just "code merged"

Typical completion artifacts:

- Validation for the PR base: local `make test-all`, plus green remote CI when
  the actual base is `main`
- benchmark results or W&B run link
- demo video / ONNX export / checkpoint path
- if user-visible behavior changes, accompanying docs updates

## PR Evidence Standard

Every PR should:

- Link the driving issue
- Record its base branch; a child PR also links its parent roadmap and integration
  branch
- Describe user-visible changes and training impact
- List the validation commands actually executed and the final local head's
  `make test-all` result
- Record current-head remote CI for a `main` base, or the local gate for any
  other base
- State whether behavior changes between `mujoco`, `motrix`, macOS, or Linux

## Ownership Model

Execution owners are expressed via GitHub assignees, and review owners are expressed via `CODEOWNERS`. If a stable GitHub handle is not yet available, leave the issue unassigned and note the intended owner temporarily in the issue body.

## ADR Governance

When a change touches runtime / backend / config / registry contracts, the issue or PR must explicitly link the corresponding ADR:

- Architecture standards entry: {doc}`Architecture Overview </en/4-developer_guide/1-architecture/1-overview>`
- ADR index: {doc}`ADR Index </adr/ADR-0000-index>`
- Backend capability boundary: {doc}`ADR-0002 </adr/ADR-0002-backend-capability-boundary-for-play-and-snapshot>`
- task owner / compose: {doc}`ADR-0003 </adr/ADR-0003-task-owner-and-config-compose-contract>`
- Registry bootstrap: {doc}`ADR-0004 </adr/ADR-0004-registry-bootstrap-contract>`

If existing ADRs cannot cover a new structural decision, add a new ADR in the same PR and link it back into the documents above.
New ADRs use the {doc}`ADR Template </adr/ADR-TEMPLATE>` and must explicitly state `Supersedes`, `Superseded by`, `Alternatives Considered`, and `Evidence In Repo`.
