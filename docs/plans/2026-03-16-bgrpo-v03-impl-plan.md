# V0.3 BGRPO Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable Branching-GRPO (BGRPO) algorithm by adding `prompt_ids` + `global_turn_index` to MATE data contracts and fixing GRPO grouping in OrchRL.

**Architecture:** MATE-first approach — extend MATE datatypes/monitor/backend/collector with two new fields (`global_turn_index`, `prompt_ids`), then fix OrchRL UID grouping and skip_turn_predicate to enable correct BGRPO group formation (做法 C: only branch-point turns enter training batch).

**Tech Stack:** Python dataclasses, aiohttp (MATE monitor), HuggingFace tokenizers, PyTorch tensors (OrchRL), verl DataProto

**Design doc:** `docs/plans/2026-03-16-bgrpo-v03-design.md`

---

## Chunk 1: MATE 侧数据契约扩展

### File Structure

| Action | File | Responsibility |
|--------|------|---------------|
| Modify | `mate/trajectory/datatypes.py` | Add `prompt_ids` and `global_turn_index` to ModelResponse, InteractionRecord, TurnData |
| Modify | `mate/trajectory/monitor.py` | Add `_global_turn_counter`, pass both new fields into InteractionRecord |
| Modify | `mate/trajectory/backend.py` | Generate `prompt_ids` via local tokenizer in VLLMBackend.generate() |
| Modify | `mate/trajectory/collector.py` | Propagate new fields from InteractionRecord to TurnData |
| Modify | `mate/trajectory/tree.py` | Use `global_turn_index` in `_sorted_buffer` instead of timestamp |
| Modify | `tests/trajectory/test_monitor.py` | Add tests for global_turn_index |
| Modify | `tests/trajectory/test_backend.py` | Add tests for prompt_ids generation |
| Modify | `tests/trajectory/test_collector.py` | Add test for field propagation |

### Task 1: Extend datatypes — add new fields to ModelResponse, InteractionRecord, TurnData

**Files:**
- Modify: `mate/trajectory/datatypes.py:21-54`

- [ ] **Step 1: Add `prompt_ids` to ModelResponse**

In `mate/trajectory/datatypes.py`, add `prompt_ids` field to `ModelResponse` (after `finish_reason`):

```python
@dataclass
class ModelResponse:
    content: str
    token_ids: list[int] | None
    logprobs: list[float] | None
    finish_reason: str
    prompt_ids: list[int] | None = None
```

- [ ] **Step 2: Add `global_turn_index` and `prompt_ids` to InteractionRecord**

In `mate/trajectory/datatypes.py`, add two fields to `InteractionRecord` (after `turn_index`):

```python
@dataclass
class InteractionRecord:
    agent_role: str
    turn_index: int
    global_turn_index: int
    timestamp: float
    messages: list[dict[str, Any]]
    generation_params: dict[str, Any]
    response_text: str
    token_ids: list[int] | None
    logprobs: list[float] | None
    finish_reason: str
    episode_id: str
    prompt_ids: list[int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

Note: `global_turn_index` is non-optional (int) — always assigned by Monitor; every code path that creates InteractionRecord goes through Monitor. `prompt_ids` is optional (placed before `metadata` default field). Backward compatibility for old serialized data is handled at the OrchRL adapter layer (`_turn_global_positions` fallback), not via defaults on the dataclass itself.

- [ ] **Step 3: Add `global_turn_index` and `prompt_ids` to TurnData**

In `mate/trajectory/datatypes.py`, add two fields to `TurnData`:

```python
@dataclass
class TurnData:
    agent_role: str
    turn_index: int
    global_turn_index: int
    messages: list[dict[str, Any]]
    response_text: str
    token_ids: list[int] | None
    logprobs: list[float] | None
    finish_reason: str
    timestamp: float
    prompt_ids: list[int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

Note: `global_turn_index` is non-optional, placed after `turn_index`. `prompt_ids` is optional, placed before `metadata`.

- [ ] **Step 4: Run existing tests to see what breaks**

Run: `cd /home/cxb/MATE-reboot && python -m pytest tests/trajectory tests/scripts -q 2>&1 | tail -20`

Expected: Multiple failures due to missing `global_turn_index` positional argument in test helper constructors. This confirms the datatype change is detected.

- [ ] **Step 5: Fix all test helpers that construct TurnData**

Search all test files for `TurnData(` constructor calls. Each must add `global_turn_index=` parameter. For test helpers that create TurnData, add a `global_turn_index` parameter with a default:

In `tests/trajectory/test_monitor.py` — the monitor tests construct `InteractionRecord` objects implicitly via the HTTP endpoint, so they should pass once Monitor is fixed (Task 2). But any direct `InteractionRecord(...)` or `TurnData(...)` construction in tests must add the new field.

Search every test file for direct `TurnData(` and `InteractionRecord(` and `ModelResponse(` constructor calls and add the new fields. The full list of affected test files:

- `tests/trajectory/test_collector.py`
- `tests/trajectory/test_tree.py`
- `tests/trajectory/test_tree_integration.py`
- `tests/trajectory/test_orchrl_integration.py`
- `tests/trajectory/test_parallel.py`
- `tests/trajectory/test_pipe.py`
- `tests/trajectory/test_datatypes.py`
- `tests/trajectory/test_replay_cache.py`
- `tests/scripts/test_run_real_validation.py`
- `tests/scripts/test_trajectory_utils.py`

For each `TurnData(...)` call, add `global_turn_index=<value>` after `turn_index=`. For `InteractionRecord(...)` calls, add `global_turn_index=<value>` after `turn_index=`. For `ModelResponse(...)` calls, `prompt_ids` has a default so no change needed.

- [ ] **Step 6: Run tests again to verify datatype changes compile**

Run: `cd /home/cxb/MATE-reboot && python -m pytest tests/trajectory tests/scripts -q 2>&1 | tail -20`

Expected: Some tests may still fail (Monitor/Backend not yet updated), but no `TypeError` from missing positional args.

- [ ] **Step 7: Commit**

```bash
cd /home/cxb/MATE-reboot
git add mate/trajectory/datatypes.py tests/
git commit -m "feat(datatypes): add global_turn_index and prompt_ids to ModelResponse, InteractionRecord, TurnData"
```

### Task 2: Extend Monitor — assign global_turn_index, propagate prompt_ids

**Files:**
- Modify: `mate/trajectory/monitor.py:16-36` (init), `mate/trajectory/monitor.py:74-78` (clear_buffer), `mate/trajectory/monitor.py:104-145` (handle_chat_completions)
- Modify: `tests/trajectory/test_monitor.py`

- [ ] **Step 1: Write the failing test for global_turn_index**

Add to `tests/trajectory/test_monitor.py`:

```python
async def test_global_turn_index_increments_across_agents():
    """global_turn_index should increment monotonically across all agents in an episode."""
    backend = RecordingBackend()
    monitor = ModelMonitor(
        backend=backend,
        model_mapping={"verifier": ModelMappingEntry(), "searcher": ModelMappingEntry()},
    )
    port = await monitor.start()
    base = f"http://127.0.0.1:{port}"

    async with httpx.AsyncClient() as client:
        # verifier turn 0 → global 0
        await client.post(f"{base}/v1/chat/completions", json={"model": "verifier", "messages": [{"role": "user", "content": "v0"}]})
        # searcher turn 0 → global 1
        await client.post(f"{base}/v1/chat/completions", json={"model": "searcher", "messages": [{"role": "user", "content": "s0"}]})
        # verifier turn 1 → global 2
        await client.post(f"{base}/v1/chat/completions", json={"model": "verifier", "messages": [{"role": "user", "content": "v1"}]})

    buf = monitor.get_buffer()
    await monitor.stop()

    assert len(buf) == 3
    assert buf[0].global_turn_index == 0
    assert buf[1].global_turn_index == 1
    assert buf[2].global_turn_index == 2
```

- [ ] **Step 2: Write the failing test for global_turn_index reset on clear_buffer**

```python
async def test_global_turn_index_resets_on_clear_buffer():
    """clear_buffer should reset global_turn_counter to 0."""
    backend = RecordingBackend()
    monitor = ModelMonitor(
        backend=backend,
        model_mapping={"verifier": ModelMappingEntry()},
    )
    port = await monitor.start()
    base = f"http://127.0.0.1:{port}"

    async with httpx.AsyncClient() as client:
        await client.post(f"{base}/v1/chat/completions", json={"model": "verifier", "messages": [{"role": "user", "content": "v0"}]})

    monitor.clear_buffer()

    async with httpx.AsyncClient() as client:
        await client.post(f"{base}/v1/chat/completions", json={"model": "verifier", "messages": [{"role": "user", "content": "v1"}]})

    buf = monitor.get_buffer()
    await monitor.stop()

    assert len(buf) == 1
    assert buf[0].global_turn_index == 0
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /home/cxb/MATE-reboot && python -m pytest tests/trajectory/test_monitor.py::test_global_turn_index_increments_across_agents tests/trajectory/test_monitor.py::test_global_turn_index_resets_on_clear_buffer -v 2>&1 | tail -20`

Expected: FAIL — `InteractionRecord.__init__() missing required positional argument: 'global_turn_index'`

- [ ] **Step 4: Implement Monitor changes**

In `mate/trajectory/monitor.py`, make three changes:

**4a.** In `__init__` (after line 30), add counter:
```python
        self._global_turn_counter = 0
```

**4b.** In `clear_buffer` (after line 77), reset counter:
```python
            self._global_turn_counter = 0
```

**4c.** In `_handle_chat_completions`, inside the first `with self._state_lock:` block (lines 104-106), add global counter:
```python
        with self._state_lock:
            turn_index = self._turn_counters.get(agent_role, 0)
            self._turn_counters[agent_role] = turn_index + 1
            global_turn_index = self._global_turn_counter
            self._global_turn_counter += 1
            generation_snapshot = self._buffer_generation
```

**4d.** In the `InteractionRecord(...)` constructor call (around line 130), add:
```python
            global_turn_index=global_turn_index,
```
after `turn_index=turn_index,`

**4e.** Add `prompt_ids` from response:
```python
            prompt_ids=response.prompt_ids if hasattr(response, 'prompt_ids') else None,
```
Note: Use `getattr` pattern since `ModelResponse.prompt_ids` has a default, but `ReplayCache` returns `ModelResponse` without it in tests. Safer to use:
```python
            prompt_ids=getattr(response, 'prompt_ids', None),
```

- [ ] **Step 5: Run the new monitor tests**

Run: `cd /home/cxb/MATE-reboot && python -m pytest tests/trajectory/test_monitor.py::test_global_turn_index_increments_across_agents tests/trajectory/test_monitor.py::test_global_turn_index_resets_on_clear_buffer -v`

Expected: PASS

- [ ] **Step 6: Run all monitor tests for regression**

Run: `cd /home/cxb/MATE-reboot && python -m pytest tests/trajectory/test_monitor.py -v 2>&1 | tail -30`

Expected: All monitor tests pass.

- [ ] **Step 7: Commit**

```bash
cd /home/cxb/MATE-reboot
git add mate/trajectory/monitor.py tests/trajectory/test_monitor.py
git commit -m "feat(monitor): assign global_turn_index and propagate prompt_ids in InteractionRecord"
```

### Task 3: Extend Backend — generate prompt_ids via local tokenizer

**Files:**
- Modify: `mate/trajectory/backend.py:56-119` (VLLMBackend.generate)
- Modify: `tests/trajectory/test_backend.py`

- [ ] **Step 1: Write the failing test for prompt_ids with tokenizer**

Add to `tests/trajectory/test_backend.py`. Use the existing `monkeypatch` + `FakeAsyncClient` pattern (the project does NOT use `pytest-httpx`):

```python
async def test_generate_populates_prompt_ids_when_tokenizer_available(monkeypatch):
    """VLLMBackend should produce prompt_ids via apply_chat_template when tokenizer is set."""

    class FakeTokenizer:
        def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True):
            return [10, 20, 30]

    class FakeAsyncClient:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): return False
        async def post(self, url, json):
            return httpx.Response(
                200,
                json={"choices": [{"message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}]},
                request=httpx.Request("POST", url),
            )

    monkeypatch.setattr("mate.trajectory.backend.httpx.AsyncClient", FakeAsyncClient)

    backend = VLLMBackend(backend_url="http://fake:8000", tokenizer=FakeTokenizer())
    request = ModelRequest(
        request_id="r1",
        agent_role="verifier",
        messages=[{"role": "user", "content": "test"}],
        generation_params={},
    )
    response = await backend.generate(request)

    assert response.prompt_ids == [10, 20, 30]
```

- [ ] **Step 2: Write the failing test for prompt_ids without tokenizer**

```python
async def test_generate_returns_none_prompt_ids_when_no_tokenizer(monkeypatch):
    """VLLMBackend without tokenizer should return prompt_ids=None."""

    class FakeAsyncClient:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): return False
        async def post(self, url, json):
            return httpx.Response(
                200,
                json={"choices": [{"message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}]},
                request=httpx.Request("POST", url),
            )

    monkeypatch.setattr("mate.trajectory.backend.httpx.AsyncClient", FakeAsyncClient)

    backend = VLLMBackend(backend_url="http://fake:8000")
    request = ModelRequest(
        request_id="r1",
        agent_role="verifier",
        messages=[{"role": "user", "content": "test"}],
        generation_params={},
    )
    response = await backend.generate(request)

    assert response.prompt_ids is None
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /home/cxb/MATE-reboot && python -m pytest tests/trajectory/test_backend.py::test_generate_populates_prompt_ids_when_tokenizer_available tests/trajectory/test_backend.py::test_generate_returns_none_prompt_ids_when_no_tokenizer -v 2>&1 | tail -20`

Expected: FAIL — response has no `prompt_ids` attribute or it's always None.

- [ ] **Step 4: Implement Backend changes**

In `mate/trajectory/backend.py`, modify `VLLMBackend.generate()`. Before the `return ModelResponse(...)` at the end (around line 113):

```python
        prompt_ids: list[int] | None = None
        if self._tokenizer is not None:
            try:
                prompt_ids = self._tokenizer.apply_chat_template(
                    request.messages, add_generation_prompt=True, tokenize=True,
                )
                if isinstance(prompt_ids, list):
                    prompt_ids = [int(t) for t in prompt_ids]
                else:
                    prompt_ids = None
            except Exception:
                prompt_ids = None

        return ModelResponse(
            content=content,
            token_ids=token_ids,
            logprobs=logprobs,
            finish_reason=finish_reason,
            prompt_ids=prompt_ids,
        )
```

Note: the `try/except` guards against tokenizer failures (e.g. malformed messages). If `apply_chat_template` returns a Tensor, the `isinstance(list)` check will catch it — but in practice, `tokenize=True` returns a list. If the tokenizer returns a Tensor (some do), add:

```python
                import torch as _torch
                if isinstance(prompt_ids, _torch.Tensor):
                    prompt_ids = prompt_ids.tolist()
```

However, to avoid importing torch in the backend module (it's not currently imported), keep the simpler version and rely on `isinstance(list)` — if it's not a list, set to None. This is safe because OrchRL has a fallback.

- [ ] **Step 5: Run the new backend tests**

Run: `cd /home/cxb/MATE-reboot && python -m pytest tests/trajectory/test_backend.py::test_generate_populates_prompt_ids_when_tokenizer_available tests/trajectory/test_backend.py::test_generate_returns_none_prompt_ids_when_no_tokenizer -v`

Expected: PASS

- [ ] **Step 6: Run all backend tests for regression**

Run: `cd /home/cxb/MATE-reboot && python -m pytest tests/trajectory/test_backend.py -v 2>&1 | tail -30`

Expected: All backend tests pass.

- [ ] **Step 7: Commit**

```bash
cd /home/cxb/MATE-reboot
git add mate/trajectory/backend.py tests/trajectory/test_backend.py
git commit -m "feat(backend): generate prompt_ids via local tokenizer in VLLMBackend"
```

### Task 4: Extend Collector — propagate new fields

**Files:**
- Modify: `mate/trajectory/collector.py:18-35` (_to_turn_data)
- Modify: `tests/trajectory/test_collector.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/trajectory/test_collector.py`:

```python
def test_to_turn_data_propagates_global_turn_index_and_prompt_ids():
    """Collector should propagate global_turn_index and prompt_ids from InteractionRecord to TurnData."""
    record = InteractionRecord(
        agent_role="verifier",
        turn_index=0,
        global_turn_index=3,
        timestamp=1.0,
        messages=[{"role": "user", "content": "test"}],
        generation_params={},
        response_text="response",
        token_ids=[1, 2],
        logprobs=[-0.1],
        finish_reason="stop",
        episode_id="ep1",
        prompt_ids=[10, 20, 30],
    )
    collector = TrajectoryCollector()
    trajectory = collector.build([record], episode_id="ep1")
    turn = trajectory.agent_trajectories["verifier"][0]

    assert turn.global_turn_index == 3
    assert turn.prompt_ids == [10, 20, 30]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/cxb/MATE-reboot && python -m pytest tests/trajectory/test_collector.py::test_to_turn_data_propagates_global_turn_index_and_prompt_ids -v`

Expected: FAIL — TurnData doesn't have the right values (collector doesn't pass them through yet).

- [ ] **Step 3: Implement Collector changes**

In `mate/trajectory/collector.py`, modify `_to_turn_data` to propagate the new fields:

```python
    @staticmethod
    def _to_turn_data(record: InteractionRecord, episode_id: str) -> TurnData:
        metadata = dict(record.metadata)
        metadata.setdefault("episode_id", episode_id)
        metadata.setdefault("agent_role", record.agent_role)
        metadata.setdefault("turn_index", record.turn_index)
        metadata.setdefault("timestamp", record.timestamp)
        return TurnData(
            agent_role=record.agent_role,
            turn_index=record.turn_index,
            global_turn_index=record.global_turn_index,
            messages=record.messages,
            response_text=record.response_text,
            token_ids=record.token_ids,
            logprobs=record.logprobs,
            finish_reason=record.finish_reason,
            timestamp=record.timestamp,
            prompt_ids=record.prompt_ids,
            metadata=metadata,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/cxb/MATE-reboot && python -m pytest tests/trajectory/test_collector.py -v`

Expected: All collector tests pass.

- [ ] **Step 5: Commit**

```bash
cd /home/cxb/MATE-reboot
git add mate/trajectory/collector.py tests/trajectory/test_collector.py
git commit -m "feat(collector): propagate global_turn_index and prompt_ids to TurnData"
```

### Task 5: Update tree.py — use global_turn_index for sorting

**Files:**
- Modify: `mate/trajectory/tree.py` (`_sorted_buffer` function)

Note: `replay_cache.py` is explicitly **not changed** per design doc boundary ("不改 ReplayCache"). ReplayCache continues to use timestamp sorting, which is correct — it only needs to replay turns in the same agent-local order, not global order.

- [ ] **Step 1: Modify `_sorted_buffer` to use `global_turn_index`**

In `mate/trajectory/tree.py`, change the `_sorted_buffer` function:

```python
def _sorted_buffer(buffer: list[InteractionRecord]) -> list[InteractionRecord]:
    return sorted(buffer, key=lambda record: record.global_turn_index)
```

- [ ] **Step 2: Run all tree tests for regression**

Run: `cd /home/cxb/MATE-reboot && python -m pytest tests/trajectory/test_tree.py tests/trajectory/test_tree_integration.py -v 2>&1 | tail -30`

Expected: All tests pass.

- [ ] **Step 3: Run full test suite**

Run: `cd /home/cxb/MATE-reboot && python -m pytest tests/trajectory tests/scripts -q`

Expected: All tests pass (91+ passed). This is the Phase 1 completion gate.

- [ ] **Step 4: Commit**

```bash
cd /home/cxb/MATE-reboot
git add mate/trajectory/tree.py
git commit -m "refactor(tree): sort buffer by global_turn_index instead of timestamp"
```

---

## Chunk 2: OrchRL 侧 BGRPO 分组修复

### File Structure

| Action | File | Responsibility |
|--------|------|---------------|
| Modify | `/home/cxb/OrchRL/trajectory/datatypes.py` | Sync vendored datatypes with MATE V0.3 |
| Modify | `/home/cxb/OrchRL/trajectory/monitor.py` | Sync vendored monitor with MATE V0.3 |
| Modify | `/home/cxb/OrchRL/trajectory/backend.py` | Sync vendored backend with MATE V0.3 |
| Modify | `/home/cxb/OrchRL/trajectory/collector.py` | Sync vendored collector with MATE V0.3 |
| Modify | `/home/cxb/OrchRL/trajectory/tree.py` | Sync vendored tree with MATE V0.3 |
| Modify | `/home/cxb/OrchRL/orchrl/trainer/mate_dataproto_adapter.py` | Fix UID scheme + skip_turn_predicate + prompt_ids consumption |
| Modify | `/home/cxb/OrchRL/orchrl/trainer/mate_rollout_adapter.py` | Ensure VLLMBackend has tokenizer in tree mode |
| Modify | `/home/cxb/OrchRL/tests/orchrl/trainer/test_mate_dataproto_adapter.py` | Update existing tests + add BGRPO grouping test |

### Task 6: Sync vendored trajectory/ to MATE V0.3

**Files:**
- Modify: All `.py` files in `/home/cxb/OrchRL/trajectory/`

- [ ] **Step 1: Copy changed files from MATE to OrchRL vendored directory**

Copy the following files from MATE, then fix relative imports:

```bash
cd /home/cxb
for f in datatypes.py monitor.py backend.py collector.py tree.py; do
  cp MATE-reboot/mate/trajectory/$f OrchRL/trajectory/$f
done
```

Note: `replay_cache.py` is NOT copied — it was not modified in MATE V0.3 (per design doc boundary).

- [ ] **Step 2: Fix imports in vendored files**

In each copied file, change `from mate.trajectory.` imports to relative imports `from .`:

- `monitor.py`: `from mate.trajectory.backend` → `from .backend`, etc.
- `backend.py`: `from mate.trajectory.datatypes` → `from .datatypes`
- `collector.py`: `from mate.trajectory.datatypes` → `from .datatypes`
- `tree.py`: `from mate.trajectory.backend` → `from .backend`, etc.

Verify: `grep -r "from mate\." /home/cxb/OrchRL/trajectory/` should return no results.

- [ ] **Step 3: Run OrchRL tests for vendored trajectory regression**

Run: `cd /home/cxb/OrchRL && python -m pytest tests/ -q 2>&1 | tail -20`

Expected: Some test failures (test helpers need `global_turn_index`), but no import errors.

- [ ] **Step 4: Fix OrchRL test helpers**

Update all `TurnData(...)` and `InteractionRecord(...)` calls in OrchRL test files to include `global_turn_index=<value>`.

Key files:
- `tests/orchrl/trainer/test_mate_dataproto_adapter.py` — `_turn()` helper
- `tests/orchrl/trainer/test_mate_rollout_adapter.py` — `_turn()` helper
- `tests/orchrl/trainer/test_multi_agents_ppo_trainer_mate.py` — `_turn()` helper

For the `_turn()` helper in `test_mate_dataproto_adapter.py`:

```python
def _turn(role: str, turn_index: int, timestamp: float, *, replayed: bool = False, global_turn_index: int = 0) -> TurnData:
    metadata = {}
    if replayed:
        metadata["replayed"] = True
    return TurnData(
        agent_role=role,
        turn_index=turn_index,
        global_turn_index=global_turn_index,
        messages=[{"role": "user", "content": f"{role}-{turn_index}"}],
        response_text=f"{role}-response-{turn_index}",
        token_ids=[turn_index + 1, turn_index + 2],
        logprobs=[-0.1, -0.2],
        finish_reason="stop",
        timestamp=timestamp,
        metadata=metadata,
    )
```

For `test_mate_rollout_adapter.py`:

```python
def _turn(role: str, turn_index: int, timestamp: float, *, global_turn_index: int = 0) -> TurnData:
    return TurnData(
        agent_role=role,
        turn_index=turn_index,
        global_turn_index=global_turn_index,
        messages=[{"role": "user", "content": f"{role}-{turn_index}"}],
        response_text=f"{role}-response-{turn_index}",
        token_ids=[turn_index + 1],
        logprobs=[-0.1],
        finish_reason="stop",
        timestamp=timestamp,
        metadata={},
    )
```

For `test_multi_agents_ppo_trainer_mate.py` — apply the same pattern to its `_turn()` helper (add `global_turn_index: int = 0` keyword arg and pass `global_turn_index=global_turn_index` to TurnData).

- [ ] **Step 5: Run OrchRL tests again**

Run: `cd /home/cxb/OrchRL && python -m pytest tests/ -q 2>&1 | tail -20`

Expected: All tests pass (vendored sync is transparent).

- [ ] **Step 6: Commit**

```bash
cd /home/cxb/OrchRL
git add trajectory/ tests/
git commit -m "sync: vendored trajectory/ to MATE V0.3 (global_turn_index + prompt_ids)"
```

### Task 7: Fix UID scheme and skip_turn_predicate in mate_dataproto_adapter

**Files:**
- Modify: `/home/cxb/OrchRL/orchrl/trainer/mate_dataproto_adapter.py:42-93` (tree_episodes_to_policy_batches), `:153` (prompt_ids consumption), `:179-193` (_turn_global_positions, _tree_uid)

- [ ] **Step 1: Write the failing test for BGRPO grouping**

Add to `tests/orchrl/trainer/test_mate_dataproto_adapter.py`:

```python
def test_bgrpo_grouping_pilot_and_branch_share_uid_at_branch_point():
    """BGRPO: pilot turn and branch turns at the same branch point must share the same uid."""
    pilot = _episode(
        "pilot",
        {
            "verifier": [_turn("verifier", 0, 1.0, global_turn_index=0), _turn("verifier", 1, 3.0, global_turn_index=2)],
            "searcher": [_turn("searcher", 0, 2.0, global_turn_index=1), _turn("searcher", 1, 4.0, global_turn_index=3)],
        },
    )
    # Branch at global position 1 (searcher turn 0)
    branch_a = _episode(
        "branch-a",
        {
            "verifier": [_turn("verifier", 0, 5.0, replayed=True, global_turn_index=0)],
            "searcher": [_turn("searcher", 0, 6.0, global_turn_index=1), _turn("searcher", 1, 7.0, global_turn_index=2)],
        },
        sample_idx=1,
    )
    branch_b = _episode(
        "branch-b",
        {
            "verifier": [_turn("verifier", 0, 8.0, replayed=True, global_turn_index=0)],
            "searcher": [_turn("searcher", 0, 9.0, global_turn_index=1), _turn("searcher", 1, 10.0, global_turn_index=2)],
        },
        sample_idx=2,
    )
    tree_episode = TreeEpisodeResult(
        pilot_result=pilot,
        branch_results=[
            BranchResult(episode_result=branch_a, branch_turn=1, branch_agent_role="searcher", parent_episode_id="pilot"),
            BranchResult(episode_result=branch_b, branch_turn=1, branch_agent_role="searcher", parent_episode_id="pilot"),
        ],
        prompt="prompt",
        tree_metadata={},
    )

    batches = tree_episodes_to_policy_batches(
        episodes=[tree_episode],
        tokenizer_dict={"policy_v": _Tokenizer(), "policy_s": _Tokenizer()},
        role_policy_mapping={"verifier": "policy_v", "searcher": "policy_s"},
        role_index_mapping={"verifier": 0, "searcher": 1},
        max_prompt_length=32,
        max_response_length=32,
    )

    searcher_uids = list(batches["policy_s"].non_tensor_batch["uid"])
    # Pilot searcher turns: bp0, bp1, bp2, bp3
    # Branch-a: only branch-point turn (global_turn_index==1) enters batch → uid matches pilot bp1
    # Branch-b: only branch-point turn (global_turn_index==1) enters batch → uid matches pilot bp1

    # Find all uids that correspond to branch point 1 (searcher, global position 1)
    bp1_uids = [uid for uid in searcher_uids if ":bp1" in uid and ":c" not in uid]
    assert len(bp1_uids) == 3  # 1 pilot + 2 branches, all same uid
    assert len(set(bp1_uids)) == 1  # All identical
```

- [ ] **Step 2: Write failing test for 做法 C — continuation turns excluded**

```python
def test_bgrpo_continuation_turns_excluded_from_batch():
    """做法 C: only branch-point turn enters batch; continuation turns are excluded."""
    pilot = _episode(
        "pilot",
        {
            "verifier": [_turn("verifier", 0, 1.0, global_turn_index=0)],
            "searcher": [_turn("searcher", 0, 2.0, global_turn_index=1)],
        },
    )
    branch = _episode(
        "branch",
        {
            # replayed prefix
            "verifier": [_turn("verifier", 0, 3.0, replayed=True, global_turn_index=0)],
            # branch-point turn (global 1) + continuation turn (global 2)
            "searcher": [_turn("searcher", 0, 4.0, global_turn_index=1), _turn("searcher", 1, 5.0, global_turn_index=2)],
        },
        sample_idx=1,
    )
    tree_episode = TreeEpisodeResult(
        pilot_result=pilot,
        branch_results=[
            BranchResult(episode_result=branch, branch_turn=1, branch_agent_role="searcher", parent_episode_id="pilot"),
        ],
        prompt="prompt",
        tree_metadata={},
    )

    batches = tree_episodes_to_policy_batches(
        episodes=[tree_episode],
        tokenizer_dict={"policy_v": _Tokenizer(), "policy_s": _Tokenizer()},
        role_policy_mapping={"verifier": "policy_v", "searcher": "policy_s"},
        role_index_mapping={"verifier": 0, "searcher": 1},
        max_prompt_length=32,
        max_response_length=32,
    )

    branch_searcher_episodes = [
        eid for eid in batches["policy_s"].non_tensor_batch["episode_id"] if eid == "branch"
    ]
    # Only 1 branch turn in batch (the branch-point), not 2 (no continuation)
    assert len(branch_searcher_episodes) == 1

    # Verifier from branch should be entirely excluded (replayed prefix + no branch-point)
    branch_verifier_episodes = [
        eid for eid in batches["policy_v"].non_tensor_batch["episode_id"] if eid == "branch"
    ]
    assert len(branch_verifier_episodes) == 0
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /home/cxb/OrchRL && python -m pytest tests/orchrl/trainer/test_mate_dataproto_adapter.py::test_bgrpo_grouping_pilot_and_branch_share_uid_at_branch_point tests/orchrl/trainer/test_mate_dataproto_adapter.py::test_bgrpo_continuation_turns_excluded_from_batch -v 2>&1 | tail -20`

Expected: FAIL — uids don't match, continuation turns still in batch.

- [ ] **Step 4: Implement UID fix — pilot uid_factory**

In `mate_dataproto_adapter.py`, change the pilot `uid_factory` in `tree_episodes_to_policy_batches` (line 66):

From:
```python
            uid_factory=lambda *, prompt_group_id, agent_idx, **_: f"{prompt_group_id}:{agent_idx}",
```

To:
```python
            uid_factory=lambda *, prompt_group_id, agent_idx, global_turn_index, **_: f"{prompt_group_id}:{agent_idx}:bp{global_turn_index}",
```

This also requires `global_turn_index_lookup` for pilot episodes. Add before the pilot `_append_episode_records` call:

```python
        pilot_positions = _turn_global_positions(tree_episode.pilot_result)
```

And pass it:
```python
            global_turn_index_lookup=pilot_positions,
```

- [ ] **Step 5: Implement UID fix — branch uid_factory and skip_turn_predicate**

In `mate_dataproto_adapter.py`, change the branch `_append_episode_records` call:

Change `_tree_uid`:
```python
def _tree_uid(*, prompt_group_id: str, agent_idx: int, branch_turn: int, global_turn_index: int) -> str:
    if global_turn_index == branch_turn:
        return f"{prompt_group_id}:{agent_idx}:bp{branch_turn}"
    return f"{prompt_group_id}:{agent_idx}:bp{branch_turn}:c{global_turn_index}"
```

Change both `uid_factory` and `skip_turn_predicate`. Use default-arg capture (`_bt=branch.branch_turn`) on **both** lambdas to avoid the closure-over-loop-variable bug:

```python
                uid_factory=lambda *, prompt_group_id, agent_idx, global_turn_index, _bt=branch.branch_turn, **_: _tree_uid(
                    prompt_group_id=prompt_group_id,
                    agent_idx=agent_idx,
                    branch_turn=_bt,
                    global_turn_index=global_turn_index,
                ),
                skip_turn_predicate=lambda turn, *, role, global_turn_index, _bt=branch.branch_turn, **_kw: global_turn_index != _bt,
```

Note: the `_bt=branch.branch_turn` default arg capture is critical on BOTH lambdas — `branch` is the loop variable in `for branch in tree_episode.branch_results:`. Without capture, all lambdas would reference the last `branch` value.

- [ ] **Step 6: Implement _turn_global_positions to use global_turn_index field**

Since TurnData now has `global_turn_index`, simplify `_turn_global_positions`:

```python
def _turn_global_positions(episode) -> dict[int, int]:
    result = {}
    for role, turns in episode.trajectory.agent_trajectories.items():
        for turn in turns:
            if hasattr(turn, 'global_turn_index') and turn.global_turn_index >= 0:
                result[id(turn)] = turn.global_turn_index
            else:
                # Fallback for old data without global_turn_index
                result[id(turn)] = -1
    if any(v < 0 for v in result.values()):
        # Fallback: reconstruct from timestamps (backward compat)
        flattened_turns = []
        for role, turns in episode.trajectory.agent_trajectories.items():
            for turn in turns:
                flattened_turns.append((turn.timestamp, turn.turn_index, role, turn))
        flattened_turns.sort(key=lambda item: (item[0], item[1], item[2]))
        return {id(turn): index for index, (_, _, _, turn) in enumerate(flattened_turns)}
    return result
```

- [ ] **Step 7: Implement prompt_ids priority consumption**

In `_append_episode_records` (line 153), change:

```python
            prompt_ids = _tokenize_messages(tokenizer, turn.messages, max_prompt_length)
```

to:

```python
            if getattr(turn, 'prompt_ids', None) is not None:
                prompt_ids = [int(t) for t in turn.prompt_ids][-max_prompt_length:]
            else:
                prompt_ids = _tokenize_messages(tokenizer, turn.messages, max_prompt_length)
```

- [ ] **Step 8: Run the new BGRPO tests**

Run: `cd /home/cxb/OrchRL && python -m pytest tests/orchrl/trainer/test_mate_dataproto_adapter.py::test_bgrpo_grouping_pilot_and_branch_share_uid_at_branch_point tests/orchrl/trainer/test_mate_dataproto_adapter.py::test_bgrpo_continuation_turns_excluded_from_batch -v`

Expected: PASS

- [ ] **Step 9: Update existing tests to match new UID format**

The existing tests expect the old UID format. Update:

1. `test_tree_episodes_to_policy_batches_keeps_pilot_uid_compatible`:
   - Old: `"prompt-7:0"` → New: `"prompt-7:0:bp0"` (verifier global 0), `"prompt-7:1:bp1"` (searcher global 1)

2. `test_tree_episodes_to_policy_batches_emits_branch_aware_uids`:
   - Old: `"prompt-7:1:b1"` → New format with bp prefix
   - Old: `"prompt-7:0:b1:t2"` → New format with bp/c prefix

3. `test_tree_episodes_to_policy_batches_skips_replayed_prefix_turns`:
   - The branch searcher turn that was previously included (continuation) may now be excluded by 做法 C

These updates must reflect the exact expected UIDs after the new scheme. Work through each test's tree structure to compute the correct expected UIDs.

- [ ] **Step 10: Run all adapter tests for regression**

Run: `cd /home/cxb/OrchRL && python -m pytest tests/orchrl/trainer/test_mate_dataproto_adapter.py -v 2>&1 | tail -30`

Expected: All tests pass.

- [ ] **Step 11: Run full OrchRL test suite**

Run: `cd /home/cxb/OrchRL && python -m pytest tests/ -q 2>&1 | tail -20`

Expected: All tests pass.

- [ ] **Step 12: Commit**

```bash
cd /home/cxb/OrchRL
git add orchrl/trainer/mate_dataproto_adapter.py tests/orchrl/trainer/test_mate_dataproto_adapter.py
git commit -m "feat(adapter): fix BGRPO UID grouping + 做法C skip_turn_predicate + prompt_ids priority"
```

### Task 8: Ensure VLLMBackend has tokenizer in tree mode

**Files:**
- Modify: `/home/cxb/OrchRL/orchrl/trainer/mate_rollout_adapter.py:162-172` (_build_backend)

- [ ] **Step 1: Assess current _build_backend**

Current code (lines 162-172):
```python
    def _build_backend(self, pipe_config: AgentPipeConfig) -> VLLMBackend:
        default_url = next(
            (entry.backend_url for entry in pipe_config.model_mapping.values() if entry.backend_url),
            None,
        )
        if default_url is None:
            raise ValueError("no backend_url available for MATE rollout backend")
        return VLLMBackend(
            backend_url=default_url,
            timeout=float(self._config.get("backend_timeout", self._config.get("timeout", 120.0))),
        )
```

This creates VLLMBackend **without** tokenizer. To enable prompt_ids, it should use `VLLMBackend.with_tokenizer()` when a model_path is available.

- [ ] **Step 2: Modify _build_backend to use tokenizer when available**

```python
    def _build_backend(self, pipe_config: AgentPipeConfig) -> VLLMBackend:
        default_url = next(
            (entry.backend_url for entry in pipe_config.model_mapping.values() if entry.backend_url),
            None,
        )
        if default_url is None:
            raise ValueError("no backend_url available for MATE rollout backend")

        model_path = self._config.get("model_path")
        timeout = float(self._config.get("backend_timeout", self._config.get("timeout", 120.0)))

        if model_path:
            return VLLMBackend.with_tokenizer(
                backend_url=default_url,
                model_path=model_path,
                timeout=timeout,
            )

        return VLLMBackend(
            backend_url=default_url,
            timeout=timeout,
        )
```

This is backward compatible: if `model_path` is not in config, behavior is unchanged.

- [ ] **Step 3: Run rollout adapter tests**

Run: `cd /home/cxb/OrchRL && python -m pytest tests/orchrl/trainer/test_mate_rollout_adapter.py -v`

Expected: All tests pass (tests mock tree_rollout/parallel_rollout, so backend construction is not exercised).

- [ ] **Step 4: Commit**

```bash
cd /home/cxb/OrchRL
git add orchrl/trainer/mate_rollout_adapter.py
git commit -m "feat(rollout_adapter): use VLLMBackend.with_tokenizer when model_path configured"
```

---

## Chunk 3: 验证与收口

### Task 9: Level 1 — Full unit test regression on both repos

- [ ] **Step 1: Run MATE full test suite**

Run: `cd /home/cxb/MATE-reboot && python -m pytest tests/trajectory tests/scripts -q`

Expected: All tests pass (91+ passed).

- [ ] **Step 2: Run OrchRL full test suite**

Run: `cd /home/cxb/OrchRL && python -m pytest tests/ -q`

Expected: All tests pass.

- [ ] **Step 3: Document results**

Record test counts and any skips. Both suites must be green before proceeding to Level 2.

### Task 10: Level 2 — SearchMAS tree smoke (2 training steps)

This task requires vLLM + retrieval service and 8-card GPU environment.

- [ ] **Step 1: Identify the existing smoke runner script**

The existing smoke script is colocated with the SearchMAS example in OrchRL. Locate it:

```bash
find /home/cxb/OrchRL -name "*smoke*" -o -name "*search_mas*tree*" | head -10
```

- [ ] **Step 2: Run smoke with 2 training steps**

Follow the existing smoke runner configuration, setting training steps to 2. Verify:
- No crash or exception
- GRPO advantage values are logged and non-zero
- Loss values are finite
- prompt_ids in TurnData is non-None (check via logging or debug print)

- [ ] **Step 3: Document smoke results**

Record the key metrics from the 2-step smoke. This is a pass/fail gate.

### Task 11: Level 3a — 8-card short run diagnostic (50-100 steps)

This task requires the full 8-card GPU environment.

- [ ] **Step 1: Configure SearchMAS BGRPO run for 50-100 steps**

Extend the smoke configuration to run 50-100 training steps with logging to tensorboard/wandb.

- [ ] **Step 2: Run the training job**

Execute the 50-100 step training run.

- [ ] **Step 3: Diagnose results against 4 criteria**

Check:
1. **Reward trend**: episode reward mean — not flat or diverging
2. **Advantage distribution**: GRPO group advantages have nonzero variance
3. **Branch-point loss**: only branch-point turns contribute gradients
4. **prompt_ids consistency**: sample comparison of stored vs re-tokenized prompt_ids

- [ ] **Step 4: Document diagnostic results**

Write results to `docs/retros/2026-03-XX-bgrpo-v03-diagnostic.md`.

Pass criteria: all 4 items show no anomaly. Reward need not rise in 50-100 steps, but must not show NaN, divergence, or all-zero advantage.

### Task 12: Update project-context.md and commit final state

- [ ] **Step 1: Update docs/project-context.md**

Mark V0.3 as implemented and verified. Update milestone status, add new doc references.

- [ ] **Step 2: Final commit**

```bash
cd /home/cxb/MATE-reboot
git add docs/project-context.md
git commit -m "docs: update project context for V0.3 BGRPO completion"
```
