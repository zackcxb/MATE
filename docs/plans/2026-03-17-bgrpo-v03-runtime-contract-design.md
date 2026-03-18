# V0.3 Design: Runtime Canonical Token Contract For MATE

> Status: **Design approved for implementation planning**
> Date: 2026-03-17
> Scope: **MATE runtime contract, drift research design, output/export boundaries**
> Supersedes as design conclusion: `docs/plans/2026-03-16-bgrpo-v03-design.md`

## 1. Purpose

This document freezes the V0.3 design conclusions reached after the 2026-03-17 brainstorming cycle.

The goal of V0.3 is not "record more side-channel `prompt_ids`". The goal is to make MATE own a **runtime canonical token contract** so that:

1. rollout-time prompt/response token truth is explicit and authoritative
2. confirmed token-drift issues are avoided by design whenever possible
3. downstream training can consume token truth without mandatory re-tokenization
4. tree rollout consumers receive explicit branch semantics instead of reconstructing them heuristically

This document is a design conclusion, not an execution-ready implementation plan.

## 2. Decision Summary

### 2.1 Core Recommendation

V0.3 adopts **dual backends plus a runtime canonical token contract**:

1. `VerlBackend` becomes the primary runtime path for training and research scenarios.
2. `VLLMBackend` remains as a compatibility, debugging, and comparison path.
3. MATE promotes runtime token truth to a first-class internal contract:
   - `runtime_prompt_ids`
   - `response_ids`
   - `response_logprobs`
   - `finish_reason`
   - optional runtime metadata such as routed experts

### 2.2 Scope Boundary

V0.3 does **not** turn MATE into a trainer-native data engine.

Specifically:

1. `DataProto` is not promoted to MATE native output.
2. full-sequence `tokens`-first is not adopted as the only primary MATE data model in V0.3.
3. `global_turn_index` is not promoted into the runtime core contract.
4. `swe_agent`-style strict replay reconstruction is not promoted into a mandatory hard gate.

### 2.3 Design Principle

For confirmed issues, prefer **design elimination** over post-hoc validation.

For hypotheses, prefer **diagnostic instrumentation** over prematurely expanding the core contract.

## 3. Confirmed Facts

### 3.1 Local Code Facts

1. Current MATE runtime still uses OpenAI-style `messages -> /v1/chat/completions`; it does not use direct token generation as the primary path.
2. Current `prompt_ids` are recorded after rendering messages; they are not the actual runtime generation input on the main path.
3. Current replay behavior depends on `(agent_role, turn_index)` plus `messages_hash`, not on token-level replay reconstruction.
4. Current OrchRL training consumption re-renders prompt token ids from `messages` instead of treating MATE-recorded prompt ids as the authoritative source.
5. Tree rollout semantics are already consumed downstream, but branch consumers currently reconstruct some ordering semantics on the OrchRL side.

### 3.2 External Baseline Facts

1. `swe_agent` uses direct `prompt_ids -> generate(...)` and then performs strict replay/alignment because its final training sample is a reconstructed flattened token stream.
2. Slime's primary strategy is different: preserve token truth directly and avoid training-side re-tokenization, rather than relying on post-episode strict replay reconstruction.
3. Official tokenizer and inference docs confirm that chat-template parameters, special-token handling, and stop-token configuration can change token sequences and therefore must be treated as contract-relevant inputs.

## 4. Confirmed Issues To Design Around

The following issues are treated as **confirmed** and must be handled by V0.3 design directly, not merely diagnosed after the fact.

### 4.1 Training-Side Re-Render Drift

Current downstream training reconstructs prompt ids from `messages`, which can diverge from rollout-time prompt token truth.

V0.3 response:

1. record runtime prompt ids from the canonical render path
2. make downstream export paths prefer recorded prompt ids over re-rendered prompt ids

### 4.2 Chat Template / Generation Prompt Drift

Prompt rendering depends on chat-template behavior and render options such as generation-prompt handling.

V0.3 response:

1. introduce a single renderer ownership boundary
2. record render fingerprint alongside runtime token truth

### 4.3 Special Token Handling Drift

Special-token insertion or duplication can occur when text rendering and tokenization are split across multiple paths.

V0.3 response:

1. use a canonical render path for prompt ids
2. avoid downstream default fallback to free re-tokenization when token truth is available

### 4.4 Stop Token / Sampling Default Drift

Runtime generation behavior depends not only on prompt tokens but also on stop-token and sampling parameter semantics.

V0.3 response:

1. capture runtime sampling fingerprint
2. treat stop-token configuration as part of runtime truth metadata

## 5. Hypotheses To Verify, Not Yet Freeze

The following are considered plausible drift sources, but are not promoted to "confirmed issue" status yet.

1. tokenizer version/config drift beyond currently observed training-side rerendering
2. decode/re-encode boundary drift in MATE-specific pipelines
3. tool-call or message normalization drift in MATE-specific multi-agent flows
4. branch replay token drift beyond the already confirmed need for explicit branch semantics

These belong in V0.3 research and diagnostics, not in the minimum runtime core contract.

## 6. Adjacent But Separate Issues

The following must not be conflated with token drift:

1. branch ordering / replay semantics
2. MoE rollout-training condition consistency

They matter for V0.3, but they are separate design axes.

## 7. Candidate Architectures

### 7.1 Option A: Conservative Dual Backend

Keep dual backends, but keep `messages` as the practical top-level contract and treat token truth mostly as enriched metadata.

Pros:

1. minimum surface change
2. easier short-term compatibility

Cons:

1. token truth remains secondary in practice
2. easier to regress into downstream rerender dependence

### 7.2 Option B: Dual Backend Plus Runtime Canonical Token Contract

Keep `VLLMBackend` and `VerlBackend`, but make runtime token truth authoritative inside MATE. `messages` remain for interaction semantics and debugging; token truth becomes the preferred training/export substrate.

Pros:

1. fixes confirmed issues without forcing a full data-model rewrite
2. supports zero re-tokenize export naturally
3. preserves current MATE role as trajectory engine
4. keeps room for future full-sequence export

Cons:

1. requires a clearer boundary between renderer, backend, validator, and exporter
2. requires more explicit metadata and runtime fingerprinting

### 7.3 Option C: Native Full-Sequence Tokens-First Contract

Move MATE fully toward a Slime-style primary data model where complete sequence tokens are the native output and per-turn trajectory structures become secondary.

Pros:

1. strongest path toward zero re-tokenize everywhere
2. cleanest long-term trainer-facing token contract

Cons:

1. turns V0.3 into a data-model rewrite
2. over-rotates MATE toward trainer-native structure too early
3. creates unnecessary risk for current tree and trajectory consumers

### 7.4 Recommended Option

Choose **Option B**.

Reason:

1. it solves confirmed runtime-contract problems directly
2. it supports zero re-tokenize export without requiring MATE to become trainer-native at V0.3
3. it preserves current MATE trajectory semantics and tree rollout role
4. it keeps `VerlBackend` as the core runtime path while preserving `VLLMBackend` for compatibility and comparison

## 8. Responsibilities And Boundaries

### 8.1 Backend

Backend is responsible only for executing generation and returning runtime truth.

`VerlBackend` responsibilities:

1. accept canonical prompt ids as runtime input
2. call direct token generation
3. return token ids, logprobs, finish reason, and optional runtime metadata

`VLLMBackend` responsibilities:

1. support OpenAI-compatible compatibility path
2. normalize its outputs into the same response schema as far as practical
3. serve as comparison/fallback path, not the primary correctness proof path

Backend is not responsible for:

1. training batch assembly
2. exporter-specific formatting
3. heavyweight post-episode replay reconstruction

### 8.2 Renderer

Renderer owns the only canonical `messages -> prompt_ids` path.

Renderer responsibilities:

1. apply chat template
2. own tokenizer and template configuration
3. own special-token policy relevant to prompt rendering
4. emit a render fingerprint

The renderer must be the single runtime authority for prompt rendering used by `VerlBackend`.

### 8.3 Validator

Validator splits into two layers.

Hard validator:

1. verify runtime prompt ids exist on canonical paths
2. verify response ids exist where training/export requires them
3. verify `len(response_ids) == len(response_logprobs)`
4. verify replay cache hits respect `messages_hash`

Diagnostic validator:

1. compare rerendered prompt ids to runtime prompt ids
2. compare decode/re-encode paths
3. inspect assistant replay span mismatch patterns
4. inspect template trailing-token mismatch patterns
5. inspect tokenizer/template/config permutations

### 8.4 Exporter

Exporter is responsible for downstream-facing contract adaptation.

V0.3 exporters may target:

1. MATE native per-turn trajectory structures
2. tokenized turn records for OrchRL consumption
3. optional `DataProto` export

Zero re-tokenize is an exporter capability built on runtime truth, not a backend responsibility.

## 9. Replay And Alignment Decision

### 9.1 What V0.3 Rejects

V0.3 does **not** adopt `swe_agent`-style strict replay/alignment as a mandatory hard gate.

Reason:

1. `swe_agent` needs it because it reconstructs a flattened multi-turn training token stream
2. current MATE consumption remains per-turn
3. copying the `swe_agent` validator wholesale would import checks whose value is tied to a different final contract

### 9.2 What V0.3 Keeps

V0.3 keeps and strengthens only runtime-relevant invariants:

1. canonical runtime prompt ids
2. response id/logprob integrity
3. replay cache message-hash integrity

### 9.3 What V0.3 Studies

Replay/alignment analysis remains useful as a **diagnostic research tool**:

1. sampled prompt rerender equality checks
2. assistant replay diagnostics
3. trailing template token diagnostics

These diagnostics should produce artifacts, not block the main runtime path by default.

## 10. Token-Drift Research Design

### 10.1 Taxonomy

The V0.3 token-drift taxonomy is split into:

1. confirmed issues
2. hypotheses
3. adjacent issues

This replaces the previous practice of treating older internal analysis as evidence by default.

### 10.2 Research Outputs

The research program must produce:

1. taxonomy table with evidence category
2. experiment matrix
3. diagnostics artifact schema
4. success criteria
5. unresolved limitations list

### 10.3 Experiment Matrix

At minimum, compare:

1. current side-channel `prompt_ids` path
2. `VerlBackend` direct-token path
3. exporter path consuming recorded prompt ids
4. rerender fallback path
5. tree replay scenarios
6. multi-agent / tool-call scenarios
7. tokenizer/template/config variants

### 10.4 Artifact Requirements

Per turn or sample, artifacts should be able to include:

1. `messages`
2. `runtime_prompt_ids`
3. rerendered prompt ids
4. `response_ids`
5. `response_logprobs`
6. render fingerprint
7. sampling fingerprint
8. mismatch reason codes
9. optional routed expert metadata

### 10.5 Success Criteria

V0.3 research is considered successful when:

1. confirmed issues are avoided by design on the canonical path
2. downstream export no longer depends by default on training-side prompt rerendering
3. hard runtime invariants are explicit and enforceable
4. hypotheses are either reproduced with evidence or downgraded

## 11. Output Contract Evolution

### 11.1 Native Output

MATE native output remains its own trajectory-oriented structures.

### 11.2 `DataProto`

`DataProto` is recommended as an **optional exporter**, not a native MATE output.

Reason:

1. it lowers downstream adapter burden
2. it preserves MATE's role as trajectory engine
3. it avoids binding the core MATE contract to one trainer data structure too early

### 11.3 Zero Re-Tokenize Support

V0.3 should explicitly support zero re-tokenize downstream consumption by ensuring exporters prefer runtime token truth over `messages`-based rerendering.

Zero re-tokenize is therefore:

1. supported by design
2. not the same thing as making `DataProto` native output

## 12. `rollout_routed_experts`

### 12.1 Decision

Include routed-expert data as an **optional field in schema**, but do not make it part of the minimal required runtime contract for every rollout.

### 12.2 Collection Policy

Use a request-side capability flag such as `return_routed_experts`.

Recommended behavior:

1. supported backends may return routed experts
2. unsupported backends may omit the field
3. dense-model paths should not pay collection cost by default

### 12.3 Rationale

This field has real value for MoE rollout-training consistency research, but it introduces:

1. runtime payload cost
2. serialization/storage cost
3. backend capability divergence

Therefore it should be optional and capability-gated.

## 13. Tree Rollout Contract

### 13.1 `global_turn_index` Decision

`global_turn_index` does not enter the runtime core contract for V0.3.

Current evidence shows it is only a downstream-derived need for tree rollout consumers, not a generation-correctness primitive.

### 13.2 Replacement Strategy

Instead of exporting a fragile cross-role global turn index, MATE should export explicit branch semantics.

Branch-level fields to preserve:

1. `branch_turn`
2. `branch_agent_role`
3. `parent_episode_id`

Per-turn fields to add for branch episodes:

1. `replayed: bool`
2. `branch_phase: "replay_prefix" | "branch_point" | "post_branch"`

Optional derived fields:

1. `branch_id`
2. `branch_step_index`

### 13.3 Why This Is Better

Tree consumers actually need to know:

1. which turns are replayed prefix
2. which turn is the branch point
3. which turns belong to post-branch continuation

These semantics are better represented directly than indirectly through a reconstructed global index.

## 14. Non-Goals

V0.3 does not require:

1. making `DataProto` the native MATE output
2. adopting full-sequence tokens-first as the only MATE data model
3. enforcing `swe_agent`-style strict replay as a universal hard gate
4. promoting `global_turn_index` into the runtime core contract
5. implementing OrchRL trainer changes inside this design doc phase

## 15. Implementation-Planning Entry Conditions

Implementation planning may begin after this design doc is accepted.

The next plan must preserve these decisions:

1. `VerlBackend` is the primary runtime path
2. renderer owns canonical prompt rendering
3. validator is split into hard invariants and diagnostics
4. exporter owns zero re-tokenize and optional `DataProto`
5. routed experts are optional and capability-gated
6. tree rollout exports explicit branch semantics instead of `global_turn_index`
