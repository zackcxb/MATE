# [RFC] Standardized Agent Trajectory Gateway

## Summary

This RFC proposes adding a standardized **Trajectory Gateway** to VERL for non-invasive trajectory collection from external agent systems. The Gateway acts as an HTTP proxy implementing the OpenAI Chat Completions API, transparently intercepting all LLM calls from any agent framework — multi-agent or single-agent — without requiring modifications to the agent code. Collected trajectories are exported as `DataProto` for direct training consumption.

## Motivation

VERL's current agent integration (`agent_loop`) is tightly coupled to specific agent implementations (e.g., SWE-Agent). Each new agent framework requires dedicated adapter code, and the trajectory collection logic is embedded within the agent loop itself. As the community adopts increasingly diverse agent frameworks, a standardized interface is needed to:

1. **Support arbitrary agent systems** — any framework that calls the OpenAI Chat Completions API can be integrated without code changes.
2. **Preserve agent-framework independence** — the agent system runs as a black-box subprocess; VERL does not need to understand or embed agent-specific logic.
3. **Collect training-ready trajectories** — including `token_ids`, `logprobs`, and canonical `prompt_ids`, with runtime token-truth guarantees.

## Design Overview

### Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        VERL Training Loop                        │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │                    Trajectory Gateway                        │ │
│  │                                                              │ │
│  │  ┌────────────┐    ┌──────────────┐    ┌─────────────────┐  │ │
│  │  │  Renderer   │───▶│    Monitor    │───▶│   Collector     │  │ │
│  │  │ (tokenizer) │    │ (HTTP proxy)  │    │ (trajectory     │  │ │
│  │  └────────────┘    └──────┬───────┘    │  assembly)      │  │ │
│  │                           │            └────────┬────────┘  │ │
│  │                     ┌─────▼─────┐               │           │ │
│  │                     │  Backend   │        ┌─────▼─────┐     │ │
│  │                     │ (vLLM /    │        │  Exporter  │     │ │
│  │                     │  direct)   │        │ (DataProto)│     │ │
│  │                     └───────────┘        └───────────┘     │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │              External Agent System (subprocess)              │ │
│  │                                                              │ │
│  │   POST /v1/chat/completions ──────────────────▶ Gateway      │ │
│  └──────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

### Core Components

**Monitor (HTTP Proxy)**

An HTTP server exposing `/v1/chat/completions`. All agent LLM calls are directed here by injecting the Gateway URL into the agent system's configuration at launch. The Monitor:

- Routes requests to the configured backend model.
- Records each interaction (messages, response, token data) into a per-episode buffer.
- Returns a standard OpenAI response — the agent system is unaware of the interception.

**Backend Abstraction**

Two backend modes are supported:

| Mode | Interface | Use Case |
|------|-----------|----------|
| **API Backend** | HTTP `/v1/chat/completions` to vLLM / compatible services | General-purpose, compatible with any OpenAI API server |
| **Direct Backend** | `prompt_ids → generate()` via VERL's inference manager | Zero-hop integration with VERL's rollout workers |

Both return `token_ids`, `logprobs`, and `finish_reason` per generation call.

**Collector & Exporter**

After an episode completes, the Collector assembles the buffer into a structured trajectory. The Exporter converts it to `DataProto`:

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

### Trajectory Storage

Trajectories are stored as **full sequences per episode** rather than per-turn records. Each episode produces a flattened token stream consisting of the canonical prompt prefix followed by all response tokens.

When **context compression** is applied during an episode (e.g., summarizing earlier turns to stay within context limits), the compressed context is treated as a new prompt — the Gateway records a fresh trajectory starting from the compressed prompt. This avoids storing stale pre-compression token sequences that no longer match the actual generation input.

## Token Drift Prevention

A key challenge in agent RL is ensuring that the token sequences consumed by training exactly match the tokens used during rollout. The Gateway addresses this through a **canonical rendering** contract:

1. **Single-owner Renderer**: A `ChatRenderer` holds the tokenizer and chat template. All `messages → prompt_ids` conversions go through this single path, producing a `render_fingerprint` (model name, template config hash) for auditability.

2. **Runtime Validation**: Hard invariants are enforced at generation time:
   - `prompt_ids` must be non-null on the canonical path.
   - `len(token_ids) == len(logprobs)` for every response.

3. **Zero Re-tokenization Export**: The Exporter uses recorded `prompt_ids` and `token_ids` directly — it never falls back to re-tokenizing from `messages`. This eliminates the primary source of prompt tokenization drift documented in VERL's existing training pipelines.

4. **Drift Diagnostics**: A non-blocking diagnostic layer compares runtime `prompt_ids` against a re-rendered baseline after each call, producing mismatch artifacts for offline analysis without impacting latency.

### Open Question: Multi-Turn Token Boundary Consistency

The canonical rendering approach works cleanly for single-turn scenarios. In **multi-turn** episodes, a potential inconsistency arises:

- At turn N, the Renderer produces `prompt_ids_N` by rendering the full accumulated `messages[0..N]` through the chat template.
- For full-sequence training, the trainer needs to construct a contiguous token stream: `prompt_ids_0 + response_ids_0 + ... + response_ids_N`.
- However, the tokens representing `response_text_0` **inside** `prompt_ids_1` (as part of the assistant message history) may not be identical to the independently recorded `response_ids_0`. The chat template wraps each historical message with role tags and special tokens, and the tokenizer may produce different tokenization when the response text appears as part of a longer sequence context.

This means **per-turn canonical `prompt_ids` may not compose into a consistent full-sequence token stream** — concatenating per-turn segments can introduce token boundary mismatches at turn boundaries.

Possible mitigations include:

- **Full-sequence render at episode end**: Render the complete message history once at the end, producing a single authoritative `prompt_ids` for the entire sequence. Per-turn `prompt_ids` are used only for runtime generation, not for training export.
- **Boundary validation**: After each turn, verify that the new `prompt_ids_N` prefix matches the previous `prompt_ids_{N-1} + response_ids_{N-1}` (plus template tokens). Flag mismatches for diagnostic review.
- **Last-response-only training**: For scenarios that only compute loss on the final response, use the last turn's `prompt_ids` as the full prompt — no cross-turn composition is needed.

**We would like community feedback on**: Which mitigation (or combination) best fits VERL's multi-turn training patterns? Are there existing conventions we should align with?

## Discussion Topics

### Context Compression

When an agent system applies context compression (e.g., summarizing early turns to fit context length), the original token history becomes invalid. The proposed approach:

- Treat the compressed context as a **new episode start** — render the compressed messages into fresh `prompt_ids`.
- Link the pre- and post-compression episodes via metadata for reward attribution.
- Do **not** attempt to maintain token continuity across compression boundaries.

Feedback requested: Should compressed episodes inherit the parent episode's reward, or should reward assignment be handled differently?

### Multimodal Inputs

The Gateway currently handles text-only `messages`. For multimodal models (e.g., vision-language), agent messages may contain image/audio references. Key considerations:

- Image tokens are model-specific (varying counts, wrapping tokens like `<image>` / `</image>`).
- The Renderer would need a model-specific processor to produce correct multimodal `prompt_ids`.
- Storage cost increases significantly with embedded image tokens.

This is not yet in scope but we want to understand the community's prioritization.

### Tool Calls (JSON Function Calling)

OpenAI-format `tool_call` responses return structured JSON rather than plain text. The Gateway needs to handle:

- Recording `tool_calls` content as the response (rather than `content` which may be null).
- Tokenization of tool call JSON — the chat template's tool-call formatting may produce different tokens than naively tokenizing the JSON string.
- Multi-step tool-use patterns where the agent receives `tool` role messages containing execution results.

The proposed approach is to rely on the chat template's native tool-call rendering (via `apply_chat_template`) to produce correct token sequences, rather than implementing custom JSON tokenization logic. We welcome input on edge cases the community has encountered.

## Integration Path

### Phase 1: Core Gateway

1. Add the Gateway module under `verl/workers/agent/gateway/`.
2. Expose a `GatewayRolloutWorker` wrapping VERL's existing inference manager.
3. Output `DataProto` directly consumable by existing trainers.
4. Validate with single-agent workflows (e.g., SWE-Agent, ReAct).

### Phase 2: Extensions

1. Multi-agent support: per-role model routing and per-role `DataProto` batch assembly.
2. Branching rollout for multi-sample algorithms (e.g., GRPO).

### Evaluation Criteria

- **Correctness**: `prompt_ids` from the Gateway match what the inference engine actually used.
- **Compatibility**: Any OpenAI-API-compatible agent system works without code changes.
- **Performance**: Gateway overhead < 5% of total rollout time.
- **Data quality**: Exported `DataProto` passes VERL's existing training pipeline validation.
