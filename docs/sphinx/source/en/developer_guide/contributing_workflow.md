# Collaboration Workflow


Repository documentation only records stable standards. Execution status, owners, and stage progression belong in GitHub collaboration objects.

If you just want to install or train UniLab, start with
{doc}`/en/getting_started/installation` and
{doc}`/en/getting_started/first_training`.

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

## Milestone Structure

Each milestone should:

- Exist as a milestone object in GitHub
- Have a tracking issue that aggregates sub-issues
- Keep execution details in the sub-issues, not in the milestone description
- Define completion by delivered artifacts, not just "code merged"

Typical completion artifacts:

- green CI
- benchmark results or W&B run link
- demo video / ONNX export / checkpoint path
- if user-visible behavior changes, accompanying docs updates

## PR Evidence Standard

Every PR should:

- Link the driving issue
- Describe user-visible changes and training impact
- List the validation commands actually executed
- State whether behavior changes between `mujoco`, `motrix`, macOS, or Linux

## Ownership Model

Execution owners are expressed via GitHub assignees, and review owners are expressed via `CODEOWNERS`. If a stable GitHub handle is not yet available, leave the issue unassigned and note the intended owner temporarily in the issue body.

## ADR Governance

When a change touches runtime / backend / config / registry contracts, the issue or PR must explicitly link the corresponding ADR:

- Architecture standards entry: {doc}`Architecture Overview </en/developer_guide/architecture/overview>`
- ADR index: {doc}`ADR Index </adr/ADR-0000-index>`
- Backend capability boundary: {doc}`ADR-0002 </adr/ADR-0002-backend-capability-boundary-for-play-and-snapshot>`
- task owner / compose: {doc}`ADR-0003 </adr/ADR-0003-task-owner-and-config-compose-contract>`
- Registry bootstrap: {doc}`ADR-0004 </adr/ADR-0004-registry-bootstrap-contract>`

If existing ADRs cannot cover a new structural decision, add a new ADR in the same PR and link it back into the documents above.
New ADRs use the {doc}`ADR Template </adr/ADR-TEMPLATE>` and must explicitly state `Supersedes`, `Superseded by`, `Alternatives Considered`, and `Evidence In Repo`.
