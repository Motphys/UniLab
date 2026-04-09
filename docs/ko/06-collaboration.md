# 협업 워크플로

언어: [English](../en/06-collaboration.md) | [简体中文](../zh_CN/06-collaboration.md) | [日本語](../ja/06-collaboration.md) | 한국어

저장소 문서는 안정적인 표준만 담아야 합니다. 실행 상태, owner, 단계 진행 상황은 GitHub 협업 객체에 두어야 합니다.

UniLab을 설치하거나 학습하려는 목적이라면 `docs/ko/README.md`, `docs/ko/01-getting-started.md`, `docs/ko/03-training.md`부터 읽으세요.

## Work Item Granularity

각 issue는 최소한 다음 질문에 답해야 합니다:

1. 어떤 문제를 해결하는가?
2. 기대하는 산출물은 무엇인가?
3. 완료 기준은 무엇인가?
4. 실행 책임자는 누구인가?
5. 어떤 상위 blocker가 있는가?

권장 issue 유형:

- `bug`
- `work item`: feature / infra / benchmark / test / sim / docs work

## Milestone Structure

각 milestone은 다음을 만족해야 합니다:

- GitHub에 milestone object로 존재해야 합니다
- 하위 issue를 요약하는 tracking issue가 있어야 합니다
- 실행 세부사항은 milestone 설명이 아니라 하위 issue에 둡니다
- 완료 정의는 "코드가 merge되었다"가 아니라 산출물 기반이어야 합니다

전형적인 완료 산출물:

- green CI
- benchmark 결과 또는 W&B run link
- demo video / ONNX export / checkpoint path
- 사용자에게 보이는 동작이 바뀌는 경우 docs update

## PR Evidence Standard

각 PR은 다음을 만족해야 합니다:

- driving issue를 link합니다
- user-facing change와 training impact를 설명합니다
- 실제로 실행한 validation command를 나열합니다
- `mujoco`, `motrix`, macOS, Linux 사이에서 동작이 바뀌는지 명시합니다

## Ownership Model

실행 owner는 GitHub assignees로 표현하고 review owner는 `CODEOWNERS`로 표현합니다. 안정적인 GitHub handle이 아직 없다면 issue를 unassigned 상태로 두고 예상 owner를 issue body에 임시로 적으세요.

## Navigation

- Previous: [G1 Motion Tracking](05-g1-motion-tracking.md)
- Next: [Contributing](CONTRIBUTING.md)
