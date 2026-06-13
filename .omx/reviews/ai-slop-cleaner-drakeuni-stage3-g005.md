AI SLOP CLEANUP REPORT
======================

Scope: G005 blocker-resolution changes for DrakeUni Stage 3 final review.

Behavior Lock: Post-fix targeted checks passed:
- `uv run python scripts/build_drake_native.py --drake-home /Users/huanghaochen/solver/drake/install`
- `uv run ruff check src/unilab/base/backend/__init__.py src/unilab/base/base.py src/unilab/envs/locomotion/go1/joystick.py src/unilab/base/backend/drake/backend.py src/unilab/base/backend/drake/backend_native.py src/unilab/base/backend/drake/native/__init__.py src/unilab/base/backend/drake/pool.py scripts/build_drake_native.py tests/base/backend/test_drake_go1_pool.py tests/base/backend/test_drake_native_pool.py tests/scripts/test_train_scripts.py`
- `uv run pytest tests/base/backend/test_drake_go1_pool.py tests/base/backend/test_drake_native_pool.py tests/scripts/test_train_scripts.py::test_ppo_go1_drake_native_config_matches_current_contact_support -q`
- `git diff --check`
- Focused pytest result: 15 passed.
- Fresh native Drake training smoke: `logs/rsl_rl_ppo/Go1JoystickFlat/2026-06-12_18-41-11_drake`
- Fresh replay MP4: `logs/rsl_rl_ppo/Go1JoystickFlat/2026-06-12_18-41-11_drake/play_video.mp4`

Cleanup Plan: Fix final review blockers in order: runtime isolation diagnostics, unsupported reward config, native-test availability diagnostics, build helper portability, and local `.omx` artifact hygiene.

Fallback Findings:
- Optional native extension import in `src/unilab/base/backend/drake/native/__init__.py`: grounded compatibility/fail-safe fallback. It now preserves the underlying `ImportError` through `native_import_error()`.
- Native backend selection in `src/unilab/base/backend/__init__.py`: no masking fallback; it now explicitly rejects native mode when `pydrake` is already present in `sys.modules`.
- Native package import in `src/unilab/base/backend/drake/native/__init__.py`: no bypass path remains; it now also fails closed before loading the native extension if `pydrake` is already present.
- `pkg-config` helper in `scripts/build_drake_native.py`: grounded toolchain discovery fallback. It does not hide build failure; missing Drake SDK now fails with explicit `DRAKE_HOME`/`--drake-home` guidance.
- Python fallback backend in plan/code: intentionally preserved until native packaging is portable across supported platforms.
- Duplicate Go1 metadata parser: still deferred and bounded to Go1-only milestone; no new generic abstraction was added.

UI/Design Findings: N/A.

Passes Completed:
- Fallback-like code resolution gate - repaired native import diagnostics and process-isolation guard.
1. Pass 1: Dead code deletion - removed unsupported Drake `contact` reward scale from the config and refreshed stale top-level replay-plan wording.
2. Pass 2: Duplicate removal - no broad extraction; preserving pydrake-free native import path is safer for this final gate.
3. Pass 3: Naming/error handling cleanup - build helper now requires explicit Drake SDK path and uses platform-specific Python link flags.
4. Pass 4: Test reinforcement - added native import diagnostic test, direct native-import guard test, pydrake-loaded factory guard test, native self-collision filter test, and Drake config contact-support regression test.

Quality Gates:
- Regression tests: PASS
- Lint: PASS
- Typecheck: N/A
- Tests: PASS
- Static/security scan: N/A

Changed Files:
- `src/unilab/base/backend/drake/native/__init__.py` - preserves native import error diagnostics and enforces direct-import pydrake isolation.
- `src/unilab/base/backend/__init__.py` - enforces native-vs-pydrake process isolation before native load.
- `src/unilab/base/backend/drake/backend_native.py` - exposes preserved native import errors in backend failures.
- `src/unilab/base/backend/drake/native/drake_env_pool.cc` - mirrors pydrake Go1 self-collision filtering before plant finalization and exposes a filtered-geometry diagnostic.
- `conf/ppo/task/go1_joystick_flat/drake.yaml` - removes unsupported contact reward scale.
- `tests/base/backend/test_drake_native_pool.py` - adds non-skipped diagnostic tests, direct native-import guard coverage, and native self-collision filter coverage.
- `tests/scripts/test_train_scripts.py` - adds Drake config contact-support regression.
- `scripts/build_drake_native.py` - removes workstation default and macOS-only unconditional link flags.
- `.gitignore` - excludes local `.omx` workflow state while leaving plan/review artifacts explicit.

Fallback Review:
- Findings: optional native import, toolchain discovery, Python fallback backend, duplicate parser.
- Classification: grounded compatibility/fail-safe fallback; duplicate parser is deferred cleanup.
- Escalation Status: no escalation; final review blockers have been fixed and will be sent back through independent review.

Remaining Risks:
- Native Drake backend remains intentionally Go1-only with hardcoded body indices for this milestone.
- Contact-force output remains zero/stubbed; the Drake training config no longer claims a contact reward until real contact support lands.
