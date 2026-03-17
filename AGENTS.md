# AGENTS.md

This file defines agent execution practices for the `MATE` repository.

## 1. Purpose

Use this file for stable agent workflow rules only.

Do not put milestone progress, temporary conclusions, or one-off debugging notes here.

## 2. Document Boundaries

1. `AGENTS.md`: agent behavior rules, orchestration rules, review gates, escalation rules.
2. `docs/project-context.md`: project status, frozen/open decisions, milestone progress.
3. `docs/plans/*.md`: design documents and implementation plans.
4. `skills/*.md`: workflow skill specs (trigger, input, process, output, done criteria).

The following directories may be created when needed:

- `docs/retros/*.md`: evidence-based retrospectives and decision traces.
- `docs/prompts/*.md`: reusable task prompts/templates.
- `docs/templates/*.md`: shared structured templates/schema registries.
- `docs/references/*.md`: external/local references with necessity notes, not policy source.

If content does not fit this boundary, move it to the correct document.

## 3. Subagent Orchestration Rules

1. Use focused subagents with one clear responsibility per task.
2. Do not nest subagent spawning.
3. Do not run multiple implementer subagents on overlapping file ownership.
4. Use explicit ownership in prompts (files, scope, constraints).
5. Run two-stage review for meaningful changes:
   - spec-compliance review
   - code-quality/risk review
6. Trust no success claim without verification evidence (tests/log output + commit/diff).
7. Add necessary explanatory comments for non-obvious logic (contracts, invariants, failure paths), and keep comments concise and non-redundant.

### Meaningful Changes Threshold

Treat a change as meaningful when it impacts behavior or contract, not only file count.

Meaningful if any condition is true:

1. Crosses module/package boundaries and changes runtime behavior.
2. Changes public interface, config contract, or reason-code vocabulary.
3. Alters failure handling, retry, validation, or security-relevant paths.
4. Changes multiple production files with user-visible or training-visible effect.
5. Introduces non-trivial design tradeoffs that need explicit review evidence.

Not meaningful by default:

1. Docs-only or comments-only updates.
2. Formatting or mechanical refactor with no behavioral change.
3. Pure renaming/move where tests and behavior remain unchanged.

## 4. Code Explanation Rule

Trigger: PR preparation, milestone closeout, or explicit request.

Mode selection:
- Default Mode A (module overview): active development, new module additions.
- Mode B (detailed change log): PR preparation, milestone closeout, or code review ("代码审查").

Frequency guard:

1. At most one explanation per task/PR scope unless the scope changes materially or user requested.

Skill: skills/skill-quick-code-change-explanation.md

## 5. Blackbox Retrospective Rule

Goal: make hidden decision logic fully explicit so another engineer can replay the full execution flow.

Skill: skills/skill-agent-self-audit-asset-precipitation.md

Frequency guard:

1. At most one retrospective per milestone (or explicitly declared incident scope).
2. Extra retrospective for a complex bugfix is allowed only when the incident scope is explicitly declared.

## 6. Rule Promotion Policy (Retro -> AGENTS)

A retrospective insight can be promoted to `AGENTS.md` only if all are true:

1. Repeatable across tasks (not one-off).
2. Actionable and testable.
3. Reduces defects, rework, or context pollution.
4. Expressible as a short rule without embedding project timeline details.

Governance frequency:

1. Run promotion sweep once per milestone by default (not per single retrospective doc).

## 7. Project Context Update Policy

1. Update `docs/project-context.md` at milestone checkpoints.
2. Keep updates factual and state-oriented (no narrative postmortems there).
3. Add links to newly canonical docs (skills/prompts/retros) in the reference map.

## 8. Entry Document Hygiene Rule

Applies to entry documents that bootstrap work or phase understanding, including:

1. `docs/project-context.md`
2. handoff prompts in `docs/prompts/`
3. phase design reset docs / pre-plans
4. other documents whose primary role is to orient the next agent quickly

When such a document is updated after a phase shift, decision reversal, or handoff rewrite, review every retained section against three checks:

1. `实时性`：does it still describe the current phase rather than the previous one
2. `有效性`：does it still help the next decision or execution step
3. `间接性`：should this document summarize/link the information instead of embedding full detail

Required cleanup rule:

1. Delete or move stale conclusions, obsolete TODOs, historical validation detail, and duplicated indexes when they no longer serve the document's entry role.
2. Do not preserve superseded context "just in case" inside entry documents; move detail to `docs/retros/`, `docs/plans/`, or `docs/ref/`.
3. If an entry document was materially simplified or retargeted, re-check neighboring entry docs for now-invalid residue.

Skill: `skills/document-entry-hygiene.md`

## 9. Git Workflow Rules

1. Use short-lived feature branches; avoid direct feature stacking on `main`.
2. Keep commits atomic (one logical change per commit).
3. Sync with latest remote branch before push/PR.
4. Do not force-push `main`; avoid destructive history rewrites on shared branches.
5. Never commit secrets or large local artifacts.

## 10. Execution Handoff Rule

Before starting a meaningful task, decide which execution venue best fits the work:

1. complete in the current window directly
2. complete in the current window with subagents
3. complete in a separate window/worktree

Decision factors:

1. current context coverage and whether loading more context here would create confusion
2. repository scope and whether the task crosses repo/worktree boundaries
3. need for isolation, long-running execution, or a distinct review/verification loop
4. whether the task is implementation-heavy enough to benefit from a fresh execution session

If a separate window/worktree is the better fit, the current agent must prepare the handoff before asking the user to switch:

1. summarize the goal, current state, constraints, and success criteria
2. identify exact repos/workdirs, files, and reference docs to read first
3. state the preferred workflow (same-window subagents vs new window) and why
4. provide a ready-to-use handoff prompt when the task is non-trivial or likely to be reused

Do not offload work with only a vague suggestion to "open a new window". The originating agent is responsible for making the transition concrete.

## 11. Objectivity Rule

When answering questions, evaluating trade-offs, or proposing designs, provide objective assessments based on technical merit. Do not align with the user's apparent preference or prior decisions unless the evidence supports them. If the user's inclination has technical downsides, state them clearly. Agreeing for the sake of agreement wastes time and produces worse designs.

## 12. Conflict Resolution Order

When instructions conflict, resolve by scope and specificity:

1. Current task direct instructions from user/developer/system.
2. Canonical frozen design docs for this phase.
3. `AGENTS.md` workflow rules.
4. Prompt templates and retrospective notes.
