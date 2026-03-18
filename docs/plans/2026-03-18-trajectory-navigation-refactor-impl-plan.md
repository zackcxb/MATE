# Trajectory Navigation Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reorganize `mate/trajectory` for easier source navigation while preserving behavior and current runtime contracts.

**Architecture:** Keep runtime entrypoints at `mate/trajectory/` and move helper modules into `mate/trajectory/_support/`. Preserve package-level re-exports and update imports in one contained pass so the refactor stays behavior-neutral.

**Tech Stack:** Python, pytest

---

### Task 1: Create Internal Support Package

**Files:**
- Create: `mate/trajectory/_support/__init__.py`
- Move: `mate/trajectory/collector.py`
- Move: `mate/trajectory/diagnostics.py`
- Move: `mate/trajectory/exporters.py`
- Move: `mate/trajectory/launcher.py`
- Move: `mate/trajectory/renderer.py`
- Move: `mate/trajectory/replay_cache.py`
- Move: `mate/trajectory/validator.py`

**Steps:**
1. Create `mate/trajectory/_support/__init__.py`.
2. Move the seven helper modules into `mate/trajectory/_support/`.
3. Do not change behavior inside moved modules except import path adjustments required by the move.
4. Commit once the support package exists and imports resolve.

### Task 2: Update Runtime Imports

**Files:**
- Modify: `mate/trajectory/__init__.py`
- Modify: `mate/trajectory/backend.py`
- Modify: `mate/trajectory/monitor.py`
- Modify: `mate/trajectory/pipe.py`
- Modify: `mate/trajectory/tree.py`
- Modify: any other repo file importing the moved modules

**Steps:**
1. Update package-internal imports to reference `mate.trajectory._support`.
2. Keep top-level `mate.trajectory` re-exports stable for helper APIs currently exposed.
3. Update tests to import from the same public locations unless a test intentionally checks internal structure.
4. Commit once the repo imports are consistent.

### Task 3: Verify Behavior Neutrality

**Files:**
- Test: `tests/trajectory`

**Steps:**
1. Run `pytest tests/trajectory -q`.
2. If import errors appear, fix only import-path issues; do not mix in behavior edits.
3. Confirm no runtime-contract assertions changed as part of the refactor.
4. Commit the final import-only refactor.

### Task 4: Optional Cleanup Guard

**Files:**
- Review: `mate/trajectory/__init__.py`

**Steps:**
1. Check whether package-level re-exports still match the current repo usage.
2. If an export is no longer used, leave it in place for this refactor unless it is clearly wrong.
3. Defer public API cleanup to a later dedicated change.

## Verification Commands

Run:

```bash
pytest tests/trajectory -q
```

Optional broader regression check:

```bash
pytest -q
```

## Constraints

1. No behavior changes.
2. No runtime contract edits.
3. No opportunistic renaming.
4. No deeper package redesign.

Plan complete and saved to `docs/plans/2026-03-18-trajectory-navigation-refactor-impl-plan.md`. Two execution options:

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

Which approach?
