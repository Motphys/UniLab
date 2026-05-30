---
orphan: true
---

# Changelog

UniLab versioning follows [SemVer](https://semver.org/). Notable changes are recorded here; for the day-to-day commit log see the [UniLab repository](https://github.com/unilabsim/UniLab).

## Unreleased

- Documentation source uses the bilingual Sphinx tree described in `docs/sphinx/README.md`: `source/en/`, `source/zh_CN/`, and shared `adr/`, `api_reference/`, `glossary.md`, and `changelog.md`.
- ADR, glossary, and changelog pages are shared content rather than per-language pages.

## 0.1.0 (current package metadata)

- `pyproject.toml` declares package version `0.1.0` and the first-level console entrypoints `train`, `eval`, `demo`, `unilab-complete`, `unilab-viz-nan`, and `unilab-export-scene`.
- The repository README documents the CPU simulation, shared-memory runtime, and GPU learning architecture, with MuJoCo and Motrix named as physics backends.
- Accepted ADRs in `docs/sphinx/source/adr/README.md` cover runtime layer boundaries, backend capability boundaries, task owner config compose, registry bootstrap, and observation / IPC contracts.
