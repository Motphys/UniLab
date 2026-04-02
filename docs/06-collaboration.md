# Collaboration Workflow

This document defines where different kinds of information should live in UniLab.

## Source of Truth

- `README.md`
  Use for stable project-facing information: architecture, installation, training entrypoints, and supported workflows.
- `CONTRIBUTING.md`
  Use for developer workflow: branching, validation, review expectations, and CI policy.
- GitHub Issues
  Use for actionable work items, bugs, benchmarks, docs tasks, and infra work.
- GitHub Milestones
  Use for phase-level planning such as `M1`.
- GitHub Projects
  Use for status tracking across issues and PRs when project scopes are available.

## What Not To Track In Docs

Do not keep the following in `README.md` or temporary markdown notes:

- ownership assignments
- sprint / milestone task lists
- execution status like "in progress", "blocked", or "done"
- review checklists for individual deliverables
- benchmark rollout plans

These belong in GitHub Issues, Milestones, and Projects.

## Issue Structure

Each issue should answer the following:

1. What problem are we solving?
2. What is the expected deliverable?
3. How do we know it is done?
4. Who owns it?
5. What other work blocks it?

Recommended issue types:

- `bug`
- `work item` for feature / infra / benchmark / test / sim work

## Milestone Structure

For each milestone:

- create one milestone object on GitHub
- create one tracking issue that links all child issues
- put execution details in child issues, not in the milestone description
- define completion by artifacts, not just merged code

Typical required artifacts:

- green CI
- benchmark result or W&B run link when applicable
- demo video / ONNX export / checkpoint path when applicable
- docs update if user-facing behavior changed

## PR Expectations

Every PR should:

- link the driving issue
- describe user-facing and training-impact changes
- list validation commands actually run
- state whether behavior changes across `mujoco`, `motrix`, macOS, or Linux

## Ownership

Use GitHub assignees for execution ownership and `CODEOWNERS` for review ownership.
If the responsible engineer is not yet mapped to a GitHub handle, keep the issue unassigned and
record the expected owner in the issue body until the handle mapping is confirmed.

## Navigation

- Previous: [G1 Motion Tracking](05-g1-motion-tracking.md)
- Next: [Contributing](../CONTRIBUTING.md)
