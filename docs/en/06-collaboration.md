# Collaboration Workflow

Languages: English | [简体中文](../zh_CN/06-collaboration.md) | [日本語](../ja/06-collaboration.md) | [한국어](../ko/06-collaboration.md)

Repository docs should capture stable standards. Execution status, owners, and phase tracking should live in GitHub collaboration objects.

If you only want to install or train UniLab, start with `README.md`, `docs/en/01-getting-started.md`, and `docs/en/03-training.md`.

## Work Item Granularity

Every issue should answer at least these questions:

1. What problem are we solving?
2. What is the expected deliverable?
3. What counts as done?
4. Who is responsible for execution?
5. What upstream blockers exist?

Recommended issue types:

- `bug`
- `work item`: feature / infra / benchmark / test / sim / docs work

## Milestone Structure

Each milestone should:

- exist as a milestone object in GitHub
- have a tracking issue that summarizes child issues
- keep execution detail in child issues, not the milestone description
- define completion in terms of delivered artifacts, not only merged code

Typical completion artifacts:

- green CI
- benchmark results or a W&B run link
- demo video / ONNX export / checkpoint path
- a docs update when user-visible behavior changes

## PR Evidence Standard

Every PR should:

- link the driving issue
- describe the user-facing change and training impact
- list the validation commands that actually ran
- state whether behavior changes across `mujoco`, `motrix`, macOS, or Linux

## Ownership Model

Use GitHub assignees for execution ownership and `CODEOWNERS` for review ownership. If a stable GitHub handle is not available yet, keep the issue unassigned and note the expected owner temporarily in the issue body.

## Navigation

- Previous: [G1 Motion Tracking](05-g1-motion-tracking.md)
- Next: [Contributing](../../CONTRIBUTING.md)
