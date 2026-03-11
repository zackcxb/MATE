# Doc Cleanup And Script Slimming Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Align project docs with the actual V0.2 status and reduce script coupling by extracting shared trajectory-analysis helpers.

**Architecture:** Keep the production library untouched unless script refactoring proves it necessary. Fix factual drift in repository docs, then move duplicated pure helper logic out of the validation scripts into a shared `scripts` module so `run_tree_validation.py` and `visualize_trajectories.py` stop depending on `run_real_validation.py` for unrelated utilities.

**Tech Stack:** Python, pytest, markdown docs

---

### Task 1: Correct Document Status

**Files:**
- Modify: `docs/project-context.md`
- Modify: `docs/plans/2026-03-05-training-integration-spec.md`

**Step 1: Update current-state facts**

- Replace outdated V0-era test/commit bullets in `docs/project-context.md` with the current `main` state and V0.2 verification status.
- Add one factual note that OrchRL currently integrates through adapter + `VLLMBackend`, not `AsyncLLMServerManager`.

**Step 2: Reframe training integration spec**

- Mark `docs/plans/2026-03-05-training-integration-spec.md` as a historical direct-integration spec.
- Add a short note near the top clarifying that the current OrchRL path uses adapter + server addresses and that `VerlBackend` remains a future direct-integration option.

**Step 3: Sanity review**

- Re-read both docs and ensure they contain no implementation claims that conflict with current code.

### Task 2: Extract Shared Script Helpers

**Files:**
- Create: `scripts/_trajectory_utils.py`
- Modify: `scripts/run_real_validation.py`
- Modify: `scripts/run_tree_validation.py`
- Modify: `scripts/visualize_trajectories.py`
- Create: `tests/scripts/test_trajectory_utils.py`

**Step 1: Write failing tests**

- Add focused tests for pure helper behavior:
  - flattening/sorting turns
  - integrity report construction
  - tree reward collection
  - prefix sharing computation

**Step 2: Run tests to verify red**

Run: `python -m pytest tests/scripts/test_trajectory_utils.py -q`

Expected: fail because the new shared helper module does not exist yet.

**Step 3: Implement minimal shared helper module**

- Create `scripts/_trajectory_utils.py` with pure helpers now duplicated across scripts.
- Keep it free of CLI/runtime concerns.

**Step 4: Rewire script imports**

- Update `run_real_validation.py`, `run_tree_validation.py`, and `visualize_trajectories.py` to import the extracted helpers.
- Reduce direct dependency from `run_tree_validation.py` to `run_real_validation.py` to only runtime-specific pieces that truly belong there.

**Step 5: Verify green**

Run:
- `python -m pytest tests/scripts/test_trajectory_utils.py -q`
- `python -m pytest tests/scripts/test_run_real_validation.py -q`
- `python -m pytest tests/trajectory tests/scripts -q`

Expected: all pass.

### Task 3: Recommend OrchRL Adaptation Workflow

**Files:**
- No code changes required unless a handoff doc becomes necessary

**Step 1: Assess execution mode**

- Decide whether OrchRL tree adaptation should happen in this session with subagents or in a separate execution window.

**Step 2: Produce handoff**

- If separate-window execution is better, provide a ready-to-use prompt and explain why.
- If current-session execution is better, explain the subagent ownership split and review gates.
