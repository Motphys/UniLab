AI SLOP CLEANUP REPORT
======================

Scope: DrakeUni Stage 3 changed files for Go1-only native backend integration.

Behavior Lock: Focused Drake backend tests and replay/training smoke paths:
- `uv run pytest tests/base/backend/test_drake_go1_pool.py tests/base/backend/test_drake_native_pool.py -q`
- `uv run python scripts/train_rsl_rl.py task=go1_joystick_flat/drake training.play_only=true training.play_render_mode=record training.play_steps=20 training.play_env_num=1 algo.load_run=2026-06-12_18-11-08_drake`

Cleanup Plan: Bound the pass to changed Drake backend/pool/test files. Inventory fallback-like code first, then fix the safest contract mismatch, then rerun focused checks.

Fallback Findings:
- Optional native extension import in `src/unilab/base/backend/drake/native/__init__.py`: grounded compatibility/fail-safe fallback. It preserves import error evidence and is covered by native availability tests.
- `pkg-config` fallback in `scripts/build_drake_native.py`: grounded toolchain fallback for local Drake SDK discovery. No masking behavior observed.
- Optional pydrake import in `src/unilab/base/backend/drake/backend.py`: existing grounded backend availability guard.
- Native-vs-pydrake routing in `src/unilab/base/backend/__init__.py`: explicit selected runtime boundary, covered by subprocess test that asserts native mode does not import pydrake.
- Duplicate Go1 MJCF metadata parser in `backend.py` and `backend_native.py`: duplication noted and intentionally deferred. Extracting it now would broaden the final cleanup pass; keeping it local preserves the pydrake-free native import boundary.

UI/Design Findings: N/A.

Passes Completed:
- Fallback-like code resolution gate - preserved grounded compatibility paths; no masking fallback slop found.
1. Pass 1: Dead code deletion - no dead code found in the scoped pass.
2. Pass 2: Duplicate removal - deferred metadata parser extraction as a later bounded cleanup.
3. Pass 3: Naming/error handling cleanup - corrected `NativeDrakeBackend.get_play_capabilities()` to avoid advertising unsupported frame-by-frame native capture.
4. Pass 4: Test reinforcement - native subprocess integration test covers selected native backend mode and verifies pydrake is not imported.

Quality Gates:
- Regression tests: PASS
- Lint: PASS
- Typecheck: N/A
- Tests: PASS
- Static/security scan: N/A

Changed Files:
- `src/unilab/base/backend/drake/backend_native.py` - corrected playback capability reporting.
- `tests/base/backend/test_drake_native_pool.py` - added subprocess coverage for native backend routing and pydrake isolation.
- `src/unilab/base/backend/drake/native/drake_env_pool.cc` - refreshed stale pydrake-mixing comment to match the native-only process boundary.

Fallback Review:
- Findings: optional native import, toolchain discovery fallback, optional pydrake backend guard, explicit native/pydrake route split, duplicate parser.
- Classification: grounded compatibility/fail-safe fallback, except parser duplication which is a deferred cleanup item.
- Escalation Status: no escalation; duplicate parser noted as a follow-up because current behavior and import isolation are verified.

Remaining Risks:
- Go1-only hardcoded Drake body indices remain intentional for this milestone until a native metadata query layer exists.
- Contact-force output remains zero/stubbed, consistent with the current Go1 contract and plan risk section.
