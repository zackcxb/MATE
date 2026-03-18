# Trajectory Navigation Refactor Design

**Goal:** Improve source navigation in `mate/trajectory` before OrchRL/VERL integration by reducing top-level clutter without changing runtime behavior or public contract semantics.

## Problem

`mate/trajectory` currently mixes runtime entrypoints and auxiliary helpers in one flat package.

This is not yet an architecture failure, but it does increase navigation cost:

1. engineers must scan many peer files to locate the runtime path
2. helper modules look equally important as orchestration modules
3. future additions will make the top-level package noisier

## Design Decision

Do a **navigation-only package reorganization**.

Keep the main runtime components at the package top level:

1. `backend.py`
2. `datatypes.py`
3. `monitor.py`
4. `parallel.py`
5. `pipe.py`
6. `reward.py`
7. `tree.py`

Move auxiliary modules into an internal support subpackage:

```text
mate/trajectory/_support/
  collector.py
  diagnostics.py
  exporters.py
  launcher.py
  renderer.py
  replay_cache.py
  validator.py
```

## Rationale

This keeps the runtime reading path short:

1. request/render/generate/record flow still starts from top-level files
2. helpers remain separate by concern, instead of being collapsed into a `utils.py` catch-all
3. the change is structurally useful while staying low risk

## Compatibility Strategy

This refactor should preserve external usage as much as possible:

1. keep `mate.trajectory` package-level re-exports stable where practical
2. update internal imports to point at `_support`
3. avoid public API cleanup in the same change
4. avoid renaming runtime concepts in the same change

## Non-Goals

This change must not:

1. modify runtime behavior
2. change V0.3 runtime contracts
3. alter test semantics
4. introduce deeper architectural layering such as `runtime/`, `contracts/`, or `orchestration/`
5. bundle unrelated helpers into a generic `utils.py`

## Acceptance Criteria

The refactor is acceptable if:

1. the top-level `mate/trajectory` package contains only the main runtime modules
2. auxiliary modules are grouped under `_support/`
3. tests pass without behavior changes
4. external imports used by the current repo remain valid or are updated in one contained pass

## Recommendation

Proceed with the small reorganization before integration work.

It is justified because it improves navigation and keeps the next integration phase cleaner, but it is intentionally limited to package organization only.
