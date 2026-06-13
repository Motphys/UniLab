CODE REVIEW REPORT
==================

Scope: DrakeUni Stage 3 native backend integration for Go1-only train/replay.

Files Reviewed: 23
Total Issues: 5
Architectural Status: WATCH

CRITICAL (0)
------------
(none)

HIGH (1)
--------
1. `src/unilab/base/backend/drake/native/__init__.py:3`
   Issue: Native extension import failure is collapsed into `native_available() == False`; native mode then reports only that the backend is unavailable. This can hide the real pydrake/native libdrake conflict.
   Fix: Preserve the native import exception, expose it from native-mode errors, add an explicit guard when `pydrake` is already loaded, and add a subprocess regression test.

MEDIUM (2)
----------
1. `conf/ppo/task/go1_joystick_flat/drake.yaml:41`
   Issue: The Drake config enables `contact: 0.24`, but the task reward registry does not include a `contact` reward; the reward dispatch skips unknown names.
   Fix: Remove the contact reward from the Drake config until real contact-force support is wired, or implement/register the reward with valid Drake contact forces.

2. `tests/base/backend/test_drake_native_pool.py:18`
   Issue: Native tests can skip the whole native path when the extension is absent, while the Drake task now selects native mode.
   Fix: Keep unavailable-native diagnostics tested, and make Drake-capable runs build the extension before the native integration tests.

LOW (2)
-------
1. `scripts/build_drake_native.py:11`
   Issue: The build helper defaults to this workstation's Drake path and macOS linker flags.
   Fix: Prefer explicit `DRAKE_HOME` or `--drake-home`; leave broader packaging/CMake work for the next portability milestone.

2. `.omx/tmux-hook.json:1`
   Issue: Local `.omx` workflow state is untracked and should not be committed with source artifacts.
   Fix: Commit only intentional `.omx/plans` and `.omx/reviews` artifacts; ignore local `.omx/state`, `.omx/context`, `.omx/handoffs`, `.omx/tmux-hook.json`, and `.omx/notepad.md`.

ARCHITECTURE WATCHLIST
----------------------
- Native/pydrake process isolation is implied by routing and lazy imports, but not enforced as a first-class runtime guard.
- The Go1-only native backend scope is acceptable for this milestone, but hardcoded Go1 metadata must stay clearly bounded until a native metadata query layer exists.

SYNTHESIS
---------
- code-reviewer recommendation: REQUEST CHANGES
- architect status: WATCH
- final recommendation: REQUEST CHANGES

RECOMMENDATION: REQUEST CHANGES
