# [RFC] Standardized Agent Trajectory Gateway

## Summary

This RFC proposes a standardized **Trajectory Gateway** for VERL that enables non-invasive trajectory collection from external agent systems. The Gateway implements the OpenAI Chat Completions API as an HTTP proxy, transparently intercepting all LLM calls from any agent framework without requiring modifications to agent code. Collected trajectories are exported as `DataProto` for direct training consumption.

## Motivation

VERL's current agent integration (`agent_loop`) is tightly coupled to specific agent implementations (e.g., SWE-Agent). Each new agent framework requires dedicated adapter code, and trajectory collection logic is embedded within the agent loop itself. As the community adopts increasingly diverse agent frameworks, a standardized interface is needed to:

1. **Support arbitrary agent systems** — any framework using the OpenAI Chat Completions API can be integrated without code changes.
2. **Preserve agent-framework independence** — the agent system runs as a black-box subprocess; VERL does not need to understand or embed agent-specific logic.
3. **Collect training-ready trajectories** — including `token_ids`, `logprobs`, and canonical `prompt_ids`, with runtime token-truth guarantees.

## Design Overview

### Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                          VERL Training Loop                          │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │                        Agent Pipe                              │   │
│  │                                                                │   │
│  │  ┌────────────┐    ┌──────────────┐    ┌─────────────────┐    │   │
│  │  │  Launcher   │    │   Gateway     │    │   Collector     │    │   │
│  │  │ (subprocess │    │ (HTTP proxy + │    │ (trajectory     │    │   │
│  │  │  manager)   │    │  tokenizer)   │    │  assembly)      │    │   │
│  │  └──────┬─────┘    └──────┬───────┘    └────────┬────────┘    │   │
│  └─────────┼─────────────────┼─────────────────────┼─────────────┘   │
│            │                 │                     │                  │
│            │                 │               ┌─────▼──────┐          │
│            │                 │               │  Exporter   │──▶ Training
│            │                 │               │ (DataProto) │    Engine │
│            │                 │               └────────────┘          │
│            │           ┌─────▼──────────┐                            │
│            │           │ AsyncLLMServer  │                            │
│            │           │ Manager (vLLM)  │                            │
│            │           └────────────────┘                            │
│  ┌─────────▼────────────────────────────────────────────────────┐   │
│  │              External Agent System (subprocess)              │   │
│  │                                                              │   │
│  │   POST /v1/chat/completions ──────────────────▶ Gateway      │   │
│  └──────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
```

### Core Components

**Agent Pipe**

One Agent Pipe is instantiated per prompt sample and manages the full lifecycle of a single episode — from launching the agent subprocess to collecting and exporting the final trajectory.

**Launcher**

The Launcher takes an agent program (a Python script or an SDK entry point) along with its configuration, and constructs the command to start an agent interaction instance. It injects the Gateway's HTTP address as the agent's LLM endpoint, requiring zero changes to the agent code.

**Gateway (HTTP Proxy)**

An HTTP server exposing `/v1/chat/completions`. The Gateway is the core interception layer:

- Tokenizes the request messages and preprocesses multimodal data if present. Records the canonical `prompt_ids`.
- Routes the request to VERL's `AsyncLLMServerManager` for inference.
- Detokenizes the response tokens to reconstruct a standard OpenAI response (including structured tool calls). Records the `response_ids` and `logprobs`.
- Returns the response to the agent — the agent system is unaware of the interception.
- Appends each interaction record to the per-episode buffer.

**Trajectory Collector & Exporter**

After an episode completes, the Collector assembles the interaction buffer into a full-sequence trajectory and the Exporter converts it to `DataProto` for consumption by VERL's training engine:

```python
# Per-episode DataProto fields
{
    "prompt_ids": [...],        # full-sequence canonical prompt tokens
    "response_ids": [...],      # full-sequence response tokens
    "response_logprobs": [...], # per-token log probabilities
    "response_mask": [...],     # 1 for response tokens, 0 for prompt
    "reward": 0.5,
}
```

When **context compression** occurs during an episode (e.g., the agent summarizes earlier turns to stay within context limits), the compressed context is treated as a new prompt — the Gateway starts a fresh trajectory from the compressed messages. This avoids storing stale pre-compression token sequences that no longer match the model's actual input.

For multi-agent systems, trajectories are grouped by agent role.

## Token Drift Prevention

A key challenge in agent RL is ensuring that the token sequences consumed by training exactly match the tokens used during rollout. The Gateway addresses this through a **canonical rendering** contract:

1. **Single-owner Renderer**: A `ChatRenderer` holds the tokenizer and chat template. All `messages → prompt_ids` conversions go through this single path, producing a `render_fingerprint` (model name, template config hash) for auditability.

2. **Runtime Validation**: Hard invariants are checked at each turn:
   - `prompt_ids` must be non-null on the canonical path.
   - `len(token_ids) == len(logprobs)` for every response.
   - The new `prompt_ids_N` prefix must match `prompt_ids_{N-1} + response_ids_{N-1}` (plus inter-turn template tokens). Mismatches are flagged for diagnostic review, or the episode is dropped in strict mode.

3. **Zero Re-tokenization Export**: The Exporter uses recorded `prompt_ids` and `token_ids` directly — it never falls back to re-tokenizing from `messages`. This eliminates the primary source of prompt tokenization drift.

4. **Drift Diagnostics**: A non-blocking diagnostic layer compares runtime `prompt_ids` against a re-rendered baseline after each call, producing mismatch artifacts for offline analysis without impacting latency.

### Open Question: Multi-Turn Token Boundary Consistency

Despite these safeguards, achieving a full token-in-token-out workflow remains challenging. When the external agent system controls context assembly, the Gateway can only observe and record the messages it receives — it cannot guarantee that these messages perfectly represent the model's actual input state. This issue is amplified in multi-turn interactions where context may be modified, truncated, or compressed by the agent between turns.

We would like the community's feedback on: **what other factors may cause token drift in this setting, and what designs can further mitigate the issue?**

