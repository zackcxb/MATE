# [RFC] Standardizing an External Agent Gateway for veRL

> **Status:** Draft for veRL GitHub Discussion  
> **Date:** 2026-03-19  
> **Authors:** @cxb (MATE team)  
> **Related:** [RFC: verl with Agent/Env #1172](https://github.com/volcengine/verl/issues/1172), [Agent Loop PR #2124](https://github.com/volcengine/verl/pull/2124), [FSM Refactor PR #3171](https://github.com/volcengine/verl/pull/3171)

---

## 1. Problem

veRL has made strong progress on internal agentic rollout through `AgentLoopBase`, `ToolAgentLoop`, and `SWEAgentLoop`. These implementations assume veRL owns the agent loop: veRL controls chat template application, model generation, tool execution, and token-state accumulation within a single process.

However, a growing class of agent systems operate **outside** veRL's process boundary:

- Multi-agent orchestration frameworks (e.g., custom search/debate/verification pipelines).
- Single-agent systems with their own environment interaction logic (e.g., external SWE-agent deployments, browser-based agents).
- Research prototypes where the agent loop evolves independently of the training framework.

For these external systems, there is currently no standardized way to:

1. **Intercept and route model requests** from the external agent to veRL-managed inference.
2. **Collect trajectory data** during the episode without modifying the agent system's code.
3. **Produce training-ready output** (`DataProto`) that veRL's trainer can consume directly.

The result is that each external agent integration requires a custom adapter, and trajectory collection logic is repeatedly reinvented.

## 2. Proposal

We propose that veRL adopt a standardized **External Agent Gateway** abstraction — a well-defined interface layer between external agent systems and veRL's inference/training infrastructure.

### 2.1 Core Responsibilities

An External Agent Gateway should handle:

| Responsibility | Description |
|---|---|
| **Request interception** | Accept OpenAI-compatible `/v1/chat/completions` requests from the external agent system, transparently route them to veRL-managed inference backends. |
| **Agent identity resolution** | Map incoming requests to agent roles/policies (e.g., via the `model` field), enabling per-agent routing and per-policy training batches. |
| **Trajectory recording** | Capture the full episode trajectory — including messages, model responses, token IDs, log-probabilities, and runtime metadata — without requiring modifications to the external agent's code. |
| **Episode lifecycle management** | Manage the start, execution, and completion of a single episode: launch the external agent process, wait for completion, collect the trajectory, compute rewards. |
| **Training-ready output** | Export the collected trajectory as a `DataProto`-compatible structure that veRL's existing trainer can consume. |

### 2.2 Scope Boundaries

This RFC is specifically about the **external agent gateway abstraction**. It does **not** propose:

- Replacing or unifying with veRL's internal `AgentLoopBase` / `ToolAgentLoop`. These serve a different use case (veRL-owned loops) and should coexist.
- Mandating multi-agent support as a first-class requirement. The gateway should work for single-agent external systems first; multi-agent is an enhancement.
- Defining a new training algorithm or modifying the trainer. The gateway's output is `DataProto`; what happens after that is unchanged.

### 2.3 Relationship to Existing Work

| veRL component | Role | Relationship to this proposal |
|---|---|---|
| `AgentLoopBase` | Abstract base for veRL-internal agent loops | Parallel abstraction; the gateway is for *external* systems |
| `ToolAgentLoop` | FSM-based multi-turn tool-calling loop | Handles internally-controlled tool loops; gateway handles externally-controlled ones |
| `SWEAgentLoop` | External SWE-Agent integration via `ModelProxy` | Closest existing pattern; gateway generalizes this approach |
| `AsyncLLMServerManager` | Inference request management | Gateway routes requests to this (or equivalent) |

The `SWEAgentLoop` + `ModelProxy` pattern is the closest existing precedent: an external process (SWE-Agent) sends model requests to a proxy, veRL captures the trajectory, and the result is used for training. This RFC proposes making that pattern a first-class, reusable abstraction rather than a one-off recipe.

## 3. Proposed Design

### 3.1 Gateway Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     External Agent System                     │
│  (multi-agent orchestrator, SWE-agent, browser agent, etc.)  │
│                                                               │
│  Agents send standard OpenAI API requests to the gateway URL  │
└────────────────────────────┬─────────────────────────────────┘
                             │  HTTP /v1/chat/completions
                             ▼
┌──────────────────────────────────────────────────────────────┐
│                    External Agent Gateway                      │
│                                                               │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐  │
│  │   Request    │  │  Trajectory  │  │   Episode Lifecycle │  │
│  │  Intercept & │  │  Recording   │  │   Management        │  │
│  │  Routing     │  │              │  │                     │  │
│  └──────┬──────┘  └──────┬───────┘  └──────────┬──────────┘  │
│         │                │                      │             │
└─────────┼────────────────┼──────────────────────┼─────────────┘
          │                │                      │
          ▼                ▼                      ▼
┌─────────────────┐  ┌──────────┐  ┌───────────────────────────┐
│ veRL Inference   │  │ Episode  │  │  Export / Adapter Layer   │
│ Backend          │  │ Result   │  │  (→ DataProto)            │
│ (vLLM, etc.)    │  │          │  │                           │
└─────────────────┘  └──────────┘  └───────────────────────────┘
```

### 3.2 Key Interfaces

We sketch the minimal interface surface. Exact API names are open for discussion.

#### Gateway Configuration

```python
@dataclass
class GatewayConfig:
    """Configuration for an External Agent Gateway instance."""
    agent_command: str                          # Command to launch the external agent
    agent_config: dict[str, Any]                # Agent-specific config (gateway injects its URL)
    model_mapping: dict[str, ModelMappingEntry]  # agent_role → inference routing
    timeout: float = 300.0
    host: str = "127.0.0.1"
    port: int = 0                               # 0 = auto-assign
```

#### Model Mapping

```python
@dataclass
class ModelMappingEntry:
    actual_model: str | None = None    # Model name to forward to the inference backend
    backend_url: str | None = None     # Optional per-agent backend URL override
```

This enables the gateway to route different agent roles to different models or inference endpoints — a common requirement when agents in the same system use different model sizes or specializations.

#### Episode Result

The gateway's canonical output is an **episode-level trajectory** — the complete, ordered record of all model interactions that occurred during one episode.

```python
@dataclass
class EpisodeResult:
    episode_id: str
    trajectory: EpisodeTrajectory     # Full trajectory for the episode
    rewards: dict[str, float]         # Per-agent rewards
    final_reward: float | None
    status: str                       # "success" | "failed"
    metadata: dict[str, Any]

@dataclass
class EpisodeTrajectory:
    episode_id: str
    agent_trajectories: dict[str, list[TurnData]]  # Keyed by agent role
    metadata: dict[str, Any]
```

Each `TurnData` entry in the trajectory captures one model interaction:

```python
@dataclass
class TurnData:
    agent_role: str
    turn_index: int
    messages: list[dict]         # The messages sent to the model
    response_text: str           # Model response content
    token_ids: list[int] | None  # Response token IDs (when available)
    logprobs: list[float] | None # Per-token log-probabilities (when available)
    finish_reason: str
    prompt_ids: list[int] | None # Prompt token IDs (when available, see §4)
    timestamp: float
    metadata: dict[str, Any]
```

#### Export to DataProto

An adapter layer converts `EpisodeResult` into veRL's `DataProto`:

```python
def episode_to_dataproto(
    episodes: list[EpisodeResult],
    tokenizer,
    max_prompt_length: int,
    max_response_length: int,
) -> DataProto:
    """Convert gateway episode results to veRL DataProto for training."""
    ...
```

This is where prompt tokenization, response padding, mask construction, and reward assignment happen. The gateway itself does not produce `DataProto` directly — the adapter does.

### 3.3 Episode Lifecycle

A single episode follows this lifecycle:

1. **Gateway starts** — binds an HTTP server exposing `/v1/chat/completions`.
2. **Agent launches** — the external agent process starts, configured to send LLM requests to the gateway URL.
3. **Request interception** — each incoming request is:
   - Identified by agent role (from the `model` field).
   - Routed to the appropriate inference backend.
   - Recorded into the trajectory buffer.
   - Returned to the agent as a standard OpenAI API response.
4. **Agent completes** — the external process exits.
5. **Trajectory assembly** — the recorded buffer is assembled into a structured `EpisodeTrajectory`.
6. **Reward computation** — rewards are computed (via a user-provided reward function).
7. **Result output** — an `EpisodeResult` is returned, ready for export to `DataProto`.

### 3.4 Non-Invasive Integration

A key design principle: **the external agent system requires no code changes**.

The gateway achieves this by:

- Exposing a standard OpenAI-compatible HTTP endpoint. Any agent that can call `/v1/chat/completions` can work with the gateway.
- Injecting the gateway URL into the agent's configuration at launch time (replacing whatever LLM endpoint the agent was configured to use).
- Using the `model` field in the request body as the agent identity signal — a convention already natural to most agent frameworks.

This means existing agent systems can be integrated by simply changing their LLM endpoint URL in configuration. No SDK, no wrapper library, no protocol changes.

## 4. Reducing Tokenization Drift

When the gateway captures trajectory data for training, a practical concern is **tokenization drift**: the prompt tokens used during inference may not match what the training side would produce by re-tokenizing the same messages.

The gateway can reduce this risk through several engineering boundaries:

### 4.1 Recording Runtime Token Metadata

When the inference backend returns token-level information (response `token_ids`, `logprobs`), the gateway records it as-is. This avoids the need for training-side re-tokenization of responses.

### 4.2 Canonical Prompt Rendering

When a local tokenizer is available, the gateway can render prompt `messages` into `prompt_ids` at request time using the same chat template and tokenizer that the inference backend uses. This recorded `prompt_ids` can then be consumed directly by the training adapter, avoiding a separate re-render on the training side.

### 4.3 Render Fingerprinting

The gateway records a `render_fingerprint` alongside token data — capturing the tokenizer identity, chat template parameters, and generation-prompt configuration used at render time. This allows downstream consumers to verify consistency or diagnose mismatches.

### 4.4 Scope of This Approach

This is not a claim that the gateway can achieve strict token-level equivalence in all external multi-turn scenarios. When the external agent system controls context assembly (e.g., context compression, history truncation, custom message formatting), the gateway can only observe and record what it sees — it cannot guarantee that the observed messages perfectly represent the model's actual input state.

The goal is pragmatic: **reduce drift where possible, record enough metadata to detect it, and avoid requiring training-side re-tokenization as the default path**.

## 5. Extensibility

The gateway abstraction is designed to support future enhancements without changing the core interface:

| Extension | How it fits |
|---|---|
| **Multi-agent routing** | Already supported via `model_mapping`; each agent role maps to a policy for training. |
| **Tree / branching rollouts** | Can be built on top of the gateway by running multiple episodes with prefix replay. Not part of the minimal contract. |
| **Parallel episode collection** | Multiple gateway instances can run concurrently for different prompts. |
| **Direct inference integration** | The gateway's inference backend is pluggable; can route to vLLM via HTTP or to veRL's `AsyncLLMServerManager` directly. |

## 6. Reference Implementation

We have built and validated a reference implementation of this gateway abstraction in the [MATE](https://github.com/xxx/MATE-reboot) project. Key components:

| Component | Role |
|---|---|
| `ModelMonitor` | HTTP server implementing the gateway's request interception, routing, and recording |
| `AgentPipe` | Episode lifecycle orchestrator (launch agent, collect trajectory, compute reward) |
| `VLLMBackend` / `VerlBackend` | Pluggable inference backends (OpenAI-compatible HTTP / direct veRL integration) |
| `ChatRenderer` | Canonical prompt rendering with fingerprinting |
| `mate_dataproto_adapter` | Proven adapter converting `EpisodeResult` → veRL `DataProto` |

This implementation has been validated with:

- A 3-agent search/verification/answering pipeline (Search MAS).
- Real vLLM inference backend.
- End-to-end trajectory collection → `DataProto` conversion → training consumption.

The reference implementation is offered as evidence that this abstraction is feasible and practical, not as the only valid implementation. We expect the upstream interface to evolve through community discussion.

## 7. Proposed Integration Path

We suggest a phased approach:

### Phase 1: Interface Agreement

- Agree on the gateway abstraction and its minimal interface surface.
- Agree on the `EpisodeResult` → `DataProto` adapter contract.
- This RFC's primary goal.

### Phase 2: Experimental Integration

- Land the gateway as an `experimental` component (similar to `verl.experimental.agent_loop`).
- Provide one concrete example (e.g., external single-agent system integration).
- Validate with existing veRL CI / training pipeline.

### Phase 3: Stabilization

- Iterate on the interface based on community feedback.
- Add documentation and additional examples.
- Consider whether the gateway and internal `AgentLoopBase` should share any common base interface.

## 8. Open Questions

We'd like community input on:

1. **Naming**: Is "External Agent Gateway" the right name, or should it be "Agent Proxy", "Trajectory Gateway", etc.?
2. **Interface location**: Should the gateway live under `verl.experimental.agent_gateway`, `verl.recipe`, or somewhere else?
3. **Inference backend coupling**: Should the gateway mandate using `AsyncLLMServerManager`, or remain backend-agnostic with an adapter?
4. **DataProto adapter ownership**: Should the `EpisodeResult → DataProto` adapter live in the gateway package, or in the trainer/recipe layer?
5. **Relationship to SWEAgentLoop**: `SWEAgentLoop` + `ModelProxy` is the closest existing pattern. Should the gateway eventually subsume this, or coexist?

## 9. Summary

| Aspect | Position |
|---|---|
| **What we propose** | A standardized External Agent Gateway abstraction for veRL |
| **What it does** | Intercepts external agent model requests, records trajectories, outputs `DataProto`-ready results |
| **What it doesn't do** | Replace internal `AgentLoopBase`, mandate multi-agent, change the trainer |
| **Integration model** | Non-invasive: external agents only need to point their LLM endpoint at the gateway URL |
| **Training output** | `EpisodeResult` → adapter → `DataProto` |
| **Anti-drift** | Record runtime token metadata, canonical prompt rendering, render fingerprinting |
| **Multi-agent** | Supported via model mapping, but not the primary pitch |
| **Reference implementation** | MATE — validated with real inference and multi-agent pipelines |
| **Ask** | Agree that this abstraction layer should exist in veRL |

We welcome feedback, alternative design ideas, and questions. Happy to provide more implementation details or run demos on request.
