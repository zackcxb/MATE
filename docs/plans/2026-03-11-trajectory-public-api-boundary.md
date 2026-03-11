# MATE Trajectory Public API Boundary

**Status:** active guidance for V0.2 to OrchRL integration

**Goal:** define which `mate.trajectory` exports external callers may rely on now, before any hard API contraction.

## Why This Exists

`mate/trajectory/__init__.py` currently exports more symbols than OrchRL needs. That is acceptable during V0.2 integration, but it creates an avoidable risk: once downstream code starts importing low-level helpers, later cleanup turns into compatibility work.

The near-term objective is therefore:

1. keep the current exports unchanged
2. define a stable subset for external consumers
3. treat all other exports as internal or provisional until a later cleanup pass

This is a soft-governance document, not a breaking change announcement.

## Stable External API

External integrations should limit themselves to the following symbols:

### Rollout Entry Points

- `parallel_rollout`
- `tree_rollout`

These are the intended collection entry points for training-side adapters.

### Configuration Types

- `AgentPipeConfig`
- `ModelMappingEntry`

These define the config contract passed into rollout entry points.

### Default Backend Path

- `VLLMBackend`

This is the only backend path that is considered integration-stable for the current OrchRL workflow.

### Reward Wiring

- `FunctionRewardProvider`

This is the supported adapter point for training-side reward computation.

### Read-Only Result Consumption

External callers may read, serialize, or adapt the following result datatypes:

- `EpisodeResult`
- `TreeEpisodeResult`
- `BranchResult`
- `TurnData`
- `EpisodeTrajectory`

They should treat these as output payload contracts, not subclassing or extension points.

## Internal Or Provisional API

The following exports should be treated as non-stable even if they remain importable:

- `AgentPipe`
- `ModelMonitor`
- `MASLauncher`
- `ReplayCache`
- `TrajectoryCollector`
- `InferenceBackend`
- `RewardProvider`
- `RewardWorker`
- `InteractionRecord`
- `ModelRequest`
- `ModelResponse`

Rationale:

1. they expose lower-level orchestration details
2. some exist to support internal composition or testing
3. some may still change as OrchRL tree integration and future backend strategy settle

External code should not build new dependencies directly on these symbols unless the contract is promoted explicitly in a later design doc.

## OrchRL Guidance

For the current OrchRL adapter work:

1. keep imports at the rollout-adapter layer
2. prefer `parallel_rollout` and `tree_rollout` as the only collection entry points
3. construct config via `AgentPipeConfig` + `ModelMappingEntry`
4. use `VLLMBackend` for the real serving path
5. consume `EpisodeResult` / `TreeEpisodeResult` as returned data, not by reaching into lower-level monitor or pipe internals

If OrchRL finds it needs a lower-level MATE symbol, treat that as an API review trigger rather than importing it casually.

## Deferred Cleanup

No export removals are planned in this document.

The expected sequence is:

1. finish OrchRL `tree_rollout` adapter integration
2. observe which symbols are actually used externally
3. promote or demote contracts explicitly
4. only then consider shrinking `mate/trajectory/__init__.py`

This keeps V0.2 integration friction low while still preventing accidental public-surface creep.
