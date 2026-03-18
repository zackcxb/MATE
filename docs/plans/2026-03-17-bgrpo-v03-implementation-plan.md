# V0.3 Runtime Canonical Token Contract Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the MATE-side V0.3 runtime contract so rollout token truth is canonical, tree rollout exposes explicit branch semantics, and optional routed-expert capture is plumbed without making `DataProto` a native output.

**Architecture:** Extend the trajectory runtime around four boundaries: renderer, backend, validator, and exporter-facing data structures. Keep `VLLMBackend` compatible, add a primary `VerlBackend` path, and push explicit branch semantics into tree outputs so downstream consumers stop reconstructing global ordering heuristically.

**Tech Stack:** Python 3.12, aiohttp, httpx, pytest, transformers tokenizer APIs, existing MATE trajectory modules.

---

### Task 1: Add Runtime Contract Datatypes And Branch Semantics

**Files:**
- Modify: `mate/trajectory/datatypes.py`
- Modify: `mate/trajectory/__init__.py`
- Test: `tests/trajectory/test_datatypes.py`

**Step 1: Write the failing tests**

Add tests that assert:

```python
def test_model_request_supports_runtime_prompt_ids_and_fingerprint():
    req = ModelRequest(
        request_id="r1",
        agent_role="verifier",
        messages=[{"role": "user", "content": "hi"}],
        generation_params={},
        prompt_ids=[1, 2, 3],
        render_fingerprint={"tokenizer": "tok-v1"},
    )
    assert req.prompt_ids == [1, 2, 3]
    assert req.render_fingerprint == {"tokenizer": "tok-v1"}


def test_turn_data_supports_branch_semantics_and_optional_routed_experts():
    turn = TurnData(
        agent_role="searcher",
        turn_index=0,
        messages=[],
        response_text="ok",
        token_ids=[4],
        logprobs=[-0.1],
        finish_reason="stop",
        timestamp=1.0,
        prompt_ids=[1, 2, 3],
        metadata={},
        replayed=True,
        branch_phase="replay_prefix",
        routed_experts=[[[7, 8]]],
    )
    assert turn.replayed is True
    assert turn.branch_phase == "replay_prefix"
    assert turn.routed_experts == [[[7, 8]]]
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/trajectory/test_datatypes.py -v`
Expected: FAIL because the new fields do not exist yet.

**Step 3: Write minimal implementation**

Update `mate/trajectory/datatypes.py` to add:

```python
@dataclass
class ModelRequest:
    request_id: str
    agent_role: str
    messages: list[dict[str, Any]]
    generation_params: dict[str, Any]
    prompt_ids: list[int] | None = None
    render_fingerprint: dict[str, Any] = field(default_factory=dict)
    sampling_fingerprint: dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelResponse:
    content: str
    token_ids: list[int] | None
    logprobs: list[float] | None
    finish_reason: str
    prompt_ids: list[int] | None = None
    routed_experts: list[Any] | None = None
    runtime_metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class TurnData:
    ...
    prompt_ids: list[int] | None = None
    replayed: bool = False
    branch_phase: str | None = None
    routed_experts: list[Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

Export any new public types from `mate/trajectory/__init__.py` if needed.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/trajectory/test_datatypes.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add mate/trajectory/datatypes.py mate/trajectory/__init__.py tests/trajectory/test_datatypes.py
git commit -m "feat: add v0.3 runtime contract fields"
```

### Task 2: Introduce Canonical Renderer And Fingerprints

**Files:**
- Create: `mate/trajectory/renderer.py`
- Modify: `mate/trajectory/backend.py`
- Modify: `mate/trajectory/__init__.py`
- Test: `tests/trajectory/test_backend.py`
- Test: `tests/trajectory/test_monitor.py`

**Step 1: Write the failing tests**

Add backend tests that assert:

```python
async def test_chat_renderer_renders_prompt_ids_and_fingerprint():
    renderer = ChatRenderer.from_tokenizer(FakeTokenizer(), model_name="Qwen")
    prompt_ids, fingerprint = renderer.render(
        [{"role": "user", "content": "hi"}],
        add_generation_prompt=True,
    )
    assert prompt_ids == [101, 102]
    assert fingerprint["model_name"] == "Qwen"
    assert "add_generation_prompt" in fingerprint


async def test_vllm_backend_uses_precomputed_prompt_ids_when_present(monkeypatch):
    req = ModelRequest(
        request_id="r1",
        agent_role="verifier",
        messages=[{"role": "user", "content": "hi"}],
        generation_params={},
        prompt_ids=[9, 9, 9],
    )
    resp = await backend.generate(req)
    assert resp.prompt_ids == [9, 9, 9]
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/trajectory/test_backend.py -v`
Expected: FAIL because `ChatRenderer` does not exist and `ModelRequest.prompt_ids` is not consumed.

**Step 3: Write minimal implementation**

Create `mate/trajectory/renderer.py` with a small renderer abstraction:

```python
class ChatRenderer:
    def __init__(self, tokenizer, *, model_name: str | None = None):
        self._tokenizer = tokenizer
        self._model_name = model_name

    def render(self, messages, *, add_generation_prompt: bool) -> tuple[list[int], dict[str, Any]]:
        prompt_ids = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
        )
        if hasattr(prompt_ids, "tolist"):
            prompt_ids = prompt_ids.tolist()
        return [int(token_id) for token_id in prompt_ids], {
            "model_name": self._model_name,
            "add_generation_prompt": add_generation_prompt,
            "tokenizer_class": type(self._tokenizer).__name__,
        }
```

Update `VLLMBackend` so it:
- accepts an optional renderer
- prefers `request.prompt_ids` if present
- otherwise uses the renderer to derive prompt ids and fingerprint
- returns `prompt_ids` from runtime truth rather than re-deriving them silently in multiple places

**Step 4: Run tests to verify they pass**

Run: `pytest tests/trajectory/test_backend.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add mate/trajectory/renderer.py mate/trajectory/backend.py mate/trajectory/__init__.py tests/trajectory/test_backend.py tests/trajectory/test_monitor.py
git commit -m "feat: add canonical chat renderer"
```

### Task 3: Implement `VerlBackend` And Hard Runtime Invariants

**Files:**
- Modify: `mate/trajectory/backend.py`
- Create: `mate/trajectory/validator.py`
- Modify: `mate/trajectory/monitor.py`
- Modify: `mate/trajectory/pipe.py`
- Modify: `mate/trajectory/__init__.py`
- Test: `tests/trajectory/test_backend.py`
- Test: `tests/trajectory/test_monitor.py`
- Test: `tests/trajectory/test_pipe.py`

**Step 1: Write the failing tests**

Add tests that assert:

```python
async def test_verl_backend_calls_direct_generate_with_prompt_ids():
    manager = FakeServerManager(token_ids=[11, 12], log_probs=[-0.1, -0.2])
    backend = VerlBackend(server_manager=manager)
    req = ModelRequest(
        request_id="r1",
        agent_role="verifier",
        messages=[{"role": "user", "content": "hi"}],
        generation_params={"max_tokens": 16},
        prompt_ids=[1, 2, 3],
    )
    resp = await backend.generate(req)
    assert manager.calls[0]["prompt_ids"] == [1, 2, 3]
    assert resp.token_ids == [11, 12]
    assert resp.logprobs == [-0.1, -0.2]


def test_validate_runtime_response_rejects_logprob_length_mismatch():
    with pytest.raises(ValueError, match="logprob"):
        validate_runtime_response(
            ModelResponse(content="x", token_ids=[1, 2], logprobs=[-0.1], finish_reason="stop")
        )
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/trajectory/test_backend.py tests/trajectory/test_monitor.py tests/trajectory/test_pipe.py -v`
Expected: FAIL because `VerlBackend` and the validator do not exist.

**Step 3: Write minimal implementation**

Add `VerlBackend` to `mate/trajectory/backend.py`:

```python
class VerlBackend(InferenceBackend):
    def __init__(self, server_manager):
        self._server_manager = server_manager

    async def generate(self, request: ModelRequest) -> ModelResponse:
        if not request.prompt_ids:
            raise ValueError("VerlBackend requires canonical prompt_ids")
        output = await self._server_manager.generate(
            request_id=request.request_id,
            prompt_ids=request.prompt_ids,
            sampling_params=request.generation_params,
        )
        return ModelResponse(
            content="",
            token_ids=list(output.token_ids),
            logprobs=list(output.log_probs) if output.log_probs is not None else None,
            finish_reason=getattr(output, "stop_reason", "stop"),
            prompt_ids=list(request.prompt_ids),
        )
```

Create `mate/trajectory/validator.py` with hard invariant helpers:

```python
def validate_runtime_request(request: ModelRequest) -> None:
    if request.prompt_ids is None:
        raise ValueError("runtime prompt_ids are required on canonical token paths")


def validate_runtime_response(response: ModelResponse) -> None:
    if response.token_ids is None or not response.token_ids:
        raise ValueError("response token_ids must not be empty")
    if response.logprobs is not None and len(response.token_ids) != len(response.logprobs):
        raise ValueError("response token_ids/logprobs length mismatch")
```

Use these validators in `ModelMonitor` and `AgentPipe` only on canonical-token paths so failures are explicit and localized.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/trajectory/test_backend.py tests/trajectory/test_monitor.py tests/trajectory/test_pipe.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add mate/trajectory/backend.py mate/trajectory/validator.py mate/trajectory/monitor.py mate/trajectory/pipe.py mate/trajectory/__init__.py tests/trajectory/test_backend.py tests/trajectory/test_monitor.py tests/trajectory/test_pipe.py
git commit -m "feat: add verl backend and runtime invariants"
```

### Task 4: Plumb Optional Routed Experts End-To-End

**Files:**
- Modify: `mate/trajectory/backend.py`
- Modify: `mate/trajectory/monitor.py`
- Modify: `mate/trajectory/replay_cache.py`
- Modify: `mate/trajectory/collector.py`
- Test: `tests/trajectory/test_backend.py`
- Test: `tests/trajectory/test_monitor.py`
- Test: `tests/trajectory/test_replay_cache.py`
- Test: `tests/trajectory/test_collector.py`

**Step 1: Write the failing tests**

Add tests that assert:

```python
async def test_vllm_backend_passes_return_routed_experts_when_enabled(monkeypatch):
    req = ModelRequest(
        request_id="r1",
        agent_role="verifier",
        messages=[{"role": "user", "content": "hi"}],
        generation_params={"return_routed_experts": True},
    )
    await backend.generate(req)
    assert captured_payload["return_routed_experts"] is True


async def test_monitor_records_routed_experts_when_backend_returns_them():
    response = ModelResponse(
        content="ok",
        token_ids=[1],
        logprobs=[-0.1],
        finish_reason="stop",
        routed_experts=[[[3, 4]]],
    )
    ...
    assert monitor.get_buffer()[0].metadata["routed_experts"] == [[[3, 4]]]
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/trajectory/test_backend.py tests/trajectory/test_monitor.py tests/trajectory/test_replay_cache.py tests/trajectory/test_collector.py -v`
Expected: FAIL because routed experts are not plumbed through yet.

**Step 3: Write minimal implementation**

- Keep collection optional and capability-gated.
- In `VLLMBackend`, forward `return_routed_experts=True` only when requested.
- In `VerlBackend`, surface routed expert metadata only if returned by the backend.
- Preserve routed experts through `InteractionRecord`, `ReplayCache`, and `TurnData`.
- Store it either in an explicit field or mirrored in metadata, but do not require it for every path.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/trajectory/test_backend.py tests/trajectory/test_monitor.py tests/trajectory/test_replay_cache.py tests/trajectory/test_collector.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add mate/trajectory/backend.py mate/trajectory/monitor.py mate/trajectory/replay_cache.py mate/trajectory/collector.py tests/trajectory/test_backend.py tests/trajectory/test_monitor.py tests/trajectory/test_replay_cache.py tests/trajectory/test_collector.py
git commit -m "feat: add optional routed expert capture"
```

### Task 5: Add Explicit Tree Branch Semantics

**Files:**
- Modify: `mate/trajectory/tree.py`
- Modify: `mate/trajectory/collector.py`
- Modify: `mate/trajectory/datatypes.py`
- Test: `tests/trajectory/test_tree.py`
- Test: `tests/trajectory/test_tree_integration.py`

**Step 1: Write the failing tests**

Add tests that assert branch turn outputs now carry explicit semantics:

```python
async def test_tree_rollout_marks_replayed_and_branch_phases(monkeypatch):
    result = await tree_rollout(...)
    branch_turns = result.branch_results[0].episode_result.trajectory.agent_trajectories["searcher"]
    assert branch_turns[0].replayed is True
    assert branch_turns[0].branch_phase == "replay_prefix"
    assert any(turn.branch_phase == "branch_point" for turn in branch_turns)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/trajectory/test_tree.py tests/trajectory/test_tree_integration.py -v`
Expected: FAIL because replayed/branch-phase fields are not set.

**Step 3: Write minimal implementation**

Implement explicit branch semantics instead of relying on downstream global-order reconstruction.

Recommended logic:

```python
def _annotate_branch_buffer(buffer, branch_turn: int):
    sorted_buffer = sorted(buffer, key=lambda record: record.timestamp)
    for idx, record in enumerate(sorted_buffer):
        if idx < branch_turn:
            record.metadata["replayed"] = True
            record.metadata["branch_phase"] = "replay_prefix"
        elif idx == branch_turn:
            record.metadata["replayed"] = False
            record.metadata["branch_phase"] = "branch_point"
        else:
            record.metadata["replayed"] = False
            record.metadata["branch_phase"] = "post_branch"
```

Ensure `TrajectoryCollector` copies these semantics into `TurnData.replayed` and `TurnData.branch_phase`.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/trajectory/test_tree.py tests/trajectory/test_tree_integration.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add mate/trajectory/tree.py mate/trajectory/collector.py mate/trajectory/datatypes.py tests/trajectory/test_tree.py tests/trajectory/test_tree_integration.py
git commit -m "feat: add explicit tree branch semantics"
```

### Task 6: Add Diagnostic Drift Artifact Hooks

**Files:**
- Create: `mate/trajectory/diagnostics.py`
- Modify: `mate/trajectory/monitor.py`
- Modify: `mate/trajectory/pipe.py`
- Modify: `mate/trajectory/__init__.py`
- Test: `tests/trajectory/test_monitor.py`
- Test: `tests/trajectory/test_pipe.py`

**Step 1: Write the failing tests**

Add tests that assert diagnostics can be emitted without blocking the runtime path:

```python
def test_build_drift_artifact_captures_runtime_and_rerender_ids():
    artifact = build_drift_artifact(
        messages=[{"role": "user", "content": "hi"}],
        runtime_prompt_ids=[1, 2],
        rerendered_prompt_ids=[1, 3],
        response_ids=[4],
        response_logprobs=[-0.1],
        render_fingerprint={"tokenizer": "tok-v1"},
    )
    assert artifact["runtime_prompt_ids"] == [1, 2]
    assert artifact["mismatch"] is True
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/trajectory/test_monitor.py tests/trajectory/test_pipe.py -v`
Expected: FAIL because no diagnostics helper exists.

**Step 3: Write minimal implementation**

Create a helper module that only builds structured diagnostics payloads; do not fail the rollout on these checks by default.

```python
def build_drift_artifact(...):
    return {
        "messages": messages,
        "runtime_prompt_ids": runtime_prompt_ids,
        "rerendered_prompt_ids": rerendered_prompt_ids,
        "response_ids": response_ids,
        "response_logprobs": response_logprobs,
        "render_fingerprint": render_fingerprint,
        "sampling_fingerprint": sampling_fingerprint,
        "mismatch": runtime_prompt_ids != rerendered_prompt_ids,
    }
```

Wire it so diagnostics can be attached to metadata or emitted by future exporters, but do not block the main path.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/trajectory/test_monitor.py tests/trajectory/test_pipe.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add mate/trajectory/diagnostics.py mate/trajectory/monitor.py mate/trajectory/pipe.py mate/trajectory/__init__.py tests/trajectory/test_monitor.py tests/trajectory/test_pipe.py
git commit -m "feat: add drift diagnostics artifacts"
```

### Task 7: Add Exporter Boundary For Token-Truth Consumption

**Files:**
- Create: `mate/trajectory/exporters.py`
- Modify: `mate/trajectory/__init__.py`
- Test: `tests/trajectory/test_orchrl_integration.py`
- Create: `tests/trajectory/test_exporters.py`

**Step 1: Write the failing tests**

Add tests for a minimal exporter helper that prefers recorded prompt ids:

```python
def test_tokenized_turn_export_prefers_recorded_prompt_ids():
    turn = TurnData(
        agent_role="verifier",
        turn_index=0,
        messages=[{"role": "user", "content": "hi"}],
        response_text="ok",
        token_ids=[5, 6],
        logprobs=[-0.1, -0.2],
        finish_reason="stop",
        timestamp=1.0,
        prompt_ids=[1, 2, 3],
    )
    record = export_tokenized_turn(turn)
    assert record["prompt_ids"] == [1, 2, 3]
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/trajectory/test_exporters.py -v`
Expected: FAIL because exporter helpers do not exist.

**Step 3: Write minimal implementation**

Create exporter helpers that make the design decision explicit:

```python
def export_tokenized_turn(turn: TurnData) -> dict[str, Any]:
    if turn.prompt_ids is None:
        raise ValueError("recorded prompt_ids required for token-truth export")
    if turn.token_ids is None:
        raise ValueError("response token_ids required for token-truth export")
    return {
        "agent_role": turn.agent_role,
        "turn_index": turn.turn_index,
        "prompt_ids": list(turn.prompt_ids),
        "response_ids": list(turn.token_ids),
        "response_logprobs": list(turn.logprobs) if turn.logprobs is not None else None,
        "replayed": turn.replayed,
        "branch_phase": turn.branch_phase,
        "routed_experts": turn.routed_experts,
    }
```

Do not implement native `DataProto` here. Keep the exporter generic and MATE-owned.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/trajectory/test_exporters.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add mate/trajectory/exporters.py mate/trajectory/__init__.py tests/trajectory/test_exporters.py tests/trajectory/test_orchrl_integration.py
git commit -m "feat: add token truth exporter boundary"
```

### Task 8: Run Verification And Refresh Entry Docs

**Files:**
- Modify: `docs/project-context.md`
- Modify: `docs/plans/2026-03-17-bgrpo-v03-runtime-contract-design.md`
- Test: `tests/trajectory/test_backend.py`
- Test: `tests/trajectory/test_monitor.py`
- Test: `tests/trajectory/test_tree.py`
- Test: `tests/trajectory/test_tree_integration.py`
- Test: `tests/trajectory/test_pipe.py`
- Test: `tests/trajectory/test_datatypes.py`
- Test: `tests/trajectory/test_replay_cache.py`
- Test: `tests/trajectory/test_collector.py`
- Test: `tests/trajectory/test_exporters.py`

**Step 1: Run targeted verification**

Run:

```bash
pytest \
  tests/trajectory/test_datatypes.py \
  tests/trajectory/test_backend.py \
  tests/trajectory/test_monitor.py \
  tests/trajectory/test_replay_cache.py \
  tests/trajectory/test_collector.py \
  tests/trajectory/test_pipe.py \
  tests/trajectory/test_tree.py \
  tests/trajectory/test_tree_integration.py \
  tests/trajectory/test_exporters.py -v
```

Expected: PASS.

**Step 2: Run the broader trajectory suite**

Run: `pytest tests/trajectory -v`
Expected: PASS, with any OrchRL-dependent tests skipped only for missing environment prerequisites.

**Step 3: Refresh entry documents if needed**

Update only factual status lines in `docs/project-context.md` and the V0.3 design doc if implementation details changed names during coding.

**Step 4: Commit**

```bash
git add docs/project-context.md docs/plans/2026-03-17-bgrpo-v03-runtime-contract-design.md
git commit -m "docs: refresh v0.3 runtime contract status"
```
