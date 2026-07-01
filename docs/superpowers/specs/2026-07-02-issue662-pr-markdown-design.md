# Issue662 PR Markdown Design

## Goal

Create a Chinese GitHub PR-description markdown document at:

```text
issues/pr/issue662_fast_sac_gpu_utilization.md
```

The document should explain how the current `issue662` work addresses the Issue 1 FastSAC GPU-utilization problem. It should read like an engineering PR, not like an LLM conversation or a chat transcript.

## Source Material

The PR document must be grounded in repository artifacts only:

- `issues/issue1.md`
- `issues/why_issue1/README.md`
- recursively referenced markdown from `issues/why_issue1/README.md`
- `issues/6_main_vs_issue662_ab_profile/reports/report.md`
- `issues/8_issue_node_ablation/reports/report.md`
- `memory/tools.md`
- `memory/hardware.md`

SVG figures must be read as source and rendered to PNG for visual inspection before being used. Report figures and selected root-cause figures should be copied into `issues/pr/figures/` so the PR document is self-contained.

## Recommended Narrative

Use a problem-attribution-optimization-validation narrative:

1. State the issue correspondence: FastSAC on `sac/g1_walk_flat/mujoco` exposes low GPU duty cycle caused by learner tiny-kernel / CUDA launch fragmentation.
2. Establish the environment boundary from local hardware/tooling records.
3. Summarize root-cause evidence from `why_issue1`: trace excludes steady replay/H2D starvation, Nsight Systems shows launch API time dominating kernel time, NCU shows underfilled kernels, `torch.compile` helps but is incomplete, and NVTX ranges localize tiny kernels to learner subranges.
4. Present `main` vs `issue662` evidence from report 6 across `g1_walk_flat_mujoco` and `g1_walk_rough_mujoco`.
5. Present issue-node ablation evidence from report 8 to show that the major improvement comes from CUDA graph and graph-boundary launch reduction, while earlier local cleanup stages are not the main E2E driver.
6. Close with limits and next steps: reward observations are smoke-level evidence, stream-sync increases must be acknowledged, and the next work is to evaluate whether other algorithms show similar launch-bound fragmentation.

## Document Structure

The PR markdown should contain these sections:

- `## 背景与问题对应`
- `## 当前实验环境`
- `## 根因证据`
- `## 优化效果：main vs issue662`
- `## 优化来源：issue node ablation`
- `## 结论与边界`
- `## 后续工作`
- `## Validation`

## Figure Policy

Copy all figures from the two final reports into `issues/pr/figures/`:

- 9 figures from `issues/6_main_vs_issue662_ab_profile/figures/`
- 4 figures from `issues/8_issue_node_ablation/figures/`

Also copy selected root-cause figures from `issues/why_issue1/` when they are necessary to make the issue correspondence self-contained. Each copied image must be referenced in the PR body.

Every figure reference must include a paragraph-length caption in Chinese that explains:

- what experiment or measurement the figure represents
- what each important numeric axis or metric means
- what conclusion can and cannot be drawn from the figure

Captions should be academic-paper-like but concise. They should not exaggerate beyond the measured artifact.

## Claim Boundaries

The PR must not claim:

- that replay/H2D was the main bottleneck for this issue
- that `torch.compile` alone solved the problem
- that reward curves prove full long-run learning equivalence
- that `make test-all` passed unless there is explicit evidence
- that improvements generalize to all algorithms or all hardware

The PR should explicitly state:

- the measured environment is RTX 5090 D / 170 SM, EPYC 7K62, Ubuntu 22.04.5, PyTorch 2.7.0+cu128, CUDA 12.8 runtime, Nsight Systems 2024.6.2, Nsight Compute 2025.1.1
- report 6 used 3 repeats for each task/variant
- report 8 used a stage-level ablation with repeat count 1
- Nsight metrics and E2E wall time come from separate measurement contexts where appropriate
- stream synchronization time increased in `issue662`, so it should be treated as a graph-boundary/runtime-shape tradeoff to keep watching

## Validation For This Documentation Task

Before completion:

- Verify the PR file exists.
- Verify every image reference in the PR points to an existing file.
- Verify all report 6 and report 8 figures are copied into `issues/pr/figures/`.
- Verify the PR includes environment information from `memory/tools.md` and `memory/hardware.md`.
- Verify the PR states the next step: evaluate other algorithms for similar launch-bound fragmentation.
