# Trajectory Engine V0.2 实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在现有 AgentPipe 架构上实现重放式树状分支采样，输出 TreeEpisodeResult 供训练侧 GRPO 分组消费。

**Architecture:** pilot run 采集完整轨迹并缓存 LLM 交互记录（ReplayCache），branch run 复用同一 MAS 子进程模式但 Monitor 在分支点之前返回缓存响应。tree_rollout 函数编排 pilot + branches，输出平坦存储 + 树索引元数据。

**Tech Stack:** Python 3.10+, aiohttp, asyncio, dataclasses, hashlib, pytest, pytest-asyncio

**前置文档：**
- 设计方向：`docs/plans/2026-03-09-trajectory-engine-v02-design-direction.md`
- V0 设计（已冻结）：`docs/plans/2026-03-04-trajectory-engine-v0-design.md`
- 设计审核修正：简化 BranchCoordinator 为 tree_rollout 函数（YAGNI），删除 shared_prefix_turns 冗余字段

**代码规模预估：** ~300 行新增代码 + ~180 行测试代码

---

## Task 1: 新增数据结构

**Files:**
- Modify: `mate/trajectory/datatypes.py`
- Test: `tests/trajectory/test_datatypes.py`

### Step 1: 编写 failing test

```python
# tests/trajectory/test_datatypes.py 中新增

def test_branch_result_fields():
    from mate.trajectory import EpisodeResult, EpisodeTrajectory, TurnData
    pilot = EpisodeResult(
        trajectory=EpisodeTrajectory(episode_id="pilot-001", agent_trajectories={}, metadata={}),
        rewards={}, final_reward=0.0, metadata={},
    )
    from mate.trajectory.datatypes import BranchResult
    branch = BranchResult(
        episode_result=EpisodeResult(
            trajectory=EpisodeTrajectory(episode_id="branch-001", agent_trajectories={}, metadata={}),
            rewards={}, final_reward=0.0, metadata={},
        ),
        branch_turn=2,
        branch_agent_role="searcher",
        parent_episode_id="pilot-001",
    )
    assert branch.branch_turn == 2
    assert branch.branch_agent_role == "searcher"
    assert branch.parent_episode_id == "pilot-001"


def test_tree_episode_result_fields():
    from mate.trajectory.datatypes import BranchResult, TreeEpisodeResult
    from mate.trajectory import EpisodeResult, EpisodeTrajectory
    pilot = EpisodeResult(
        trajectory=EpisodeTrajectory(episode_id="pilot-001", agent_trajectories={}, metadata={}),
        rewards={}, final_reward=0.0, metadata={},
    )
    tree = TreeEpisodeResult(
        pilot_result=pilot,
        branch_results=[],
        prompt="test prompt",
    )
    assert tree.pilot_result is pilot
    assert tree.prompt == "test prompt"
    assert tree.tree_metadata == {}


def test_episode_result_status_default():
    from mate.trajectory import EpisodeResult, EpisodeTrajectory
    result = EpisodeResult(
        trajectory=EpisodeTrajectory(episode_id="e1", agent_trajectories={}, metadata={}),
        rewards={}, final_reward=0.0, metadata={},
    )
    assert result.status == "success"
    assert result.failure_info is None
```

### Step 2: 运行测试验证失败

Run: `cd /home/cxb/MATE-reboot && python -m pytest tests/trajectory/test_datatypes.py -v -k "branch_result or tree_episode or status_default"`
Expected: FAIL — `ImportError` / `AttributeError`

### Step 3: 实现数据结构

在 `mate/trajectory/datatypes.py` 末尾添加：

```python
@dataclass
class EpisodeResult:
    trajectory: EpisodeTrajectory
    rewards: dict[str, float | list[float]]
    final_reward: float | None
    metadata: dict[str, Any] = field(default_factory=dict)
    status: str = "success"
    failure_info: dict[str, Any] | None = None


@dataclass
class BranchResult:
    episode_result: EpisodeResult
    branch_turn: int
    branch_agent_role: str
    parent_episode_id: str


@dataclass
class TreeEpisodeResult:
    pilot_result: EpisodeResult
    branch_results: list[BranchResult]
    prompt: str
    tree_metadata: dict[str, Any] = field(default_factory=dict)
```

注意：`EpisodeResult` 是修改现有 dataclass（添加 `status` 和 `failure_info` 字段，带默认值），不是新建。

### Step 4: 运行测试验证通过

Run: `cd /home/cxb/MATE-reboot && python -m pytest tests/trajectory/test_datatypes.py -v`
Expected: ALL PASS（包括原有测试）

### Step 5: 运行全量回归

Run: `cd /home/cxb/MATE-reboot && python -m pytest tests/ -q --timeout=120`
Expected: 全部通过（status 和 failure_info 有默认值，不影响现有代码）

### Step 6: Commit

```bash
git add mate/trajectory/datatypes.py tests/trajectory/test_datatypes.py
git commit -m "feat: add BranchResult, TreeEpisodeResult, EpisodeResult.status for V0.2 tree branching"
```

---

## Task 2: ReplayCache

**Files:**
- Create: `mate/trajectory/replay_cache.py`
- Test: `tests/trajectory/test_replay_cache.py`

### Step 1: 编写 failing test

```python
# tests/trajectory/test_replay_cache.py

import pytest
from mate.trajectory.datatypes import InteractionRecord, ModelResponse


def _make_record(agent_role: str, turn_index: int, response_text: str = "cached") -> InteractionRecord:
    return InteractionRecord(
        agent_role=agent_role,
        turn_index=turn_index,
        timestamp=float(turn_index),
        messages=[{"role": "user", "content": f"turn {turn_index}"}],
        generation_params={},
        response_text=response_text,
        token_ids=[1, 2, 3],
        logprobs=[-0.1, -0.2, -0.3],
        finish_reason="stop",
        episode_id="pilot-001",
    )


def test_replay_cache_from_buffer_and_lookup():
    from mate.trajectory.replay_cache import ReplayCache
    buffer = [
        _make_record("verifier", 0),
        _make_record("searcher", 0),
        _make_record("verifier", 1),
        _make_record("answerer", 0),
    ]
    cache = ReplayCache.from_buffer(buffer)
    hit = cache.lookup("verifier", 0)
    assert hit is not None
    assert hit.content == "cached"
    assert hit.token_ids == [1, 2, 3]


def test_replay_cache_miss():
    from mate.trajectory.replay_cache import ReplayCache
    buffer = [_make_record("verifier", 0)]
    cache = ReplayCache.from_buffer(buffer)
    assert cache.lookup("searcher", 0) is None
    assert cache.lookup("verifier", 1) is None


def test_replay_cache_truncated_at_branch_point():
    from mate.trajectory.replay_cache import ReplayCache
    buffer = [
        _make_record("verifier", 0),   # global pos 0
        _make_record("searcher", 0),   # global pos 1
        _make_record("verifier", 1),   # global pos 2
        _make_record("answerer", 0),   # global pos 3
    ]
    # Branch at global position 2 (verifier:1) → cache only positions 0,1
    cache = ReplayCache.from_buffer(buffer, branch_at_global_position=2)
    assert cache.lookup("verifier", 0) is not None
    assert cache.lookup("searcher", 0) is not None
    assert cache.lookup("verifier", 1) is None   # branch point excluded
    assert cache.lookup("answerer", 0) is None    # after branch point


def test_replay_cache_size():
    from mate.trajectory.replay_cache import ReplayCache
    buffer = [_make_record("v", 0), _make_record("s", 0)]
    cache = ReplayCache.from_buffer(buffer)
    assert len(cache) == 2
```

### Step 2: 运行测试验证失败

Run: `cd /home/cxb/MATE-reboot && python -m pytest tests/trajectory/test_replay_cache.py -v`
Expected: FAIL — `ModuleNotFoundError`

### Step 3: 实现 ReplayCache

```python
# mate/trajectory/replay_cache.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .datatypes import InteractionRecord, ModelResponse


class ReplayCache:
    """Stores pilot episode LLM interactions for replay during branch runs."""

    def __init__(self, entries: dict[tuple[str, int], ModelResponse]) -> None:
        self._entries = entries

    @classmethod
    def from_buffer(
        cls,
        buffer: list[InteractionRecord],
        branch_at_global_position: int | None = None,
    ) -> ReplayCache:
        sorted_records = sorted(buffer, key=lambda r: r.timestamp)
        entries: dict[tuple[str, int], ModelResponse] = {}
        for global_pos, record in enumerate(sorted_records):
            if branch_at_global_position is not None and global_pos >= branch_at_global_position:
                break
            entries[(record.agent_role, record.turn_index)] = ModelResponse(
                content=record.response_text,
                token_ids=record.token_ids,
                logprobs=record.logprobs,
                finish_reason=record.finish_reason,
            )
        return cls(entries)

    def lookup(self, agent_role: str, turn_index: int) -> ModelResponse | None:
        return self._entries.get((agent_role, turn_index))

    def __len__(self) -> int:
        return len(self._entries)
```

### Step 4: 运行测试验证通过

Run: `cd /home/cxb/MATE-reboot && python -m pytest tests/trajectory/test_replay_cache.py -v`
Expected: ALL PASS

### Step 5: Commit

```bash
git add mate/trajectory/replay_cache.py tests/trajectory/test_replay_cache.py
git commit -m "feat: add ReplayCache for pilot episode response caching"
```

---

## Task 3: Monitor 支持 ReplayCache

**Files:**
- Modify: `mate/trajectory/monitor.py`
- Test: `tests/trajectory/test_monitor.py`

### Step 1: 编写 failing test

```python
# tests/trajectory/test_monitor.py 中新增

@pytest.mark.asyncio
async def test_monitor_uses_replay_cache():
    """When replay_cache is set, Monitor returns cached response without calling backend."""
    from mate.trajectory.replay_cache import ReplayCache
    from mate.trajectory.datatypes import ModelResponse

    mock_backend = MockBackend(response_text="fresh-response")
    cache_entries = {
        ("verifier", 0): ModelResponse(
            content="cached-response",
            token_ids=[10, 20],
            logprobs=[-0.5, -0.6],
            finish_reason="stop",
        )
    }
    replay_cache = ReplayCache(cache_entries)

    monitor = ModelMonitor(
        backend=mock_backend,
        model_mapping={"verifier": ModelMappingEntry()},
        replay_cache=replay_cache,
    )
    port = await monitor.start()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                json={"model": "verifier", "messages": [{"role": "user", "content": "test"}]},
            ) as resp:
                data = await resp.json()
                assert data["choices"][0]["message"]["content"] == "cached-response"

        # Backend should NOT have been called
        assert mock_backend.call_count == 0

        # Buffer should still contain the replayed interaction
        buffer = monitor.get_buffer()
        assert len(buffer) == 1
        assert buffer[0].metadata.get("replayed") is True
        assert buffer[0].token_ids == [10, 20]
    finally:
        await monitor.stop()
```

需要在已有 test fixtures 基础上添加（`MockBackend` 等应已存在；如无 `call_count` 则需适配）。

### Step 2: 运行测试验证失败

Run: `cd /home/cxb/MATE-reboot && python -m pytest tests/trajectory/test_monitor.py -v -k "replay_cache"`
Expected: FAIL — `TypeError: __init__() got unexpected keyword argument 'replay_cache'`

### Step 3: 修改 Monitor

在 `mate/trajectory/monitor.py` 中：

1. 添加 `replay_cache` 构造参数（默认 `None`，保持向后兼容）
2. 在 `_handle_chat_completions` 中，`backend.generate()` 调用之前检查 replay cache
3. 缓存命中时仍写入 buffer，但标记 `metadata["replayed"] = True`

修改点：
- `__init__` 签名添加 `replay_cache: ReplayCache | None = None`
- `_handle_chat_completions` 在 `try: response = await self._backend.generate(...)` 前添加缓存查询分支

### Step 4: 运行测试验证通过

Run: `cd /home/cxb/MATE-reboot && python -m pytest tests/trajectory/test_monitor.py -v`
Expected: ALL PASS（包括原有测试，因为 `replay_cache=None` 是默认值）

### Step 5: 全量回归

Run: `cd /home/cxb/MATE-reboot && python -m pytest tests/ -q --timeout=120`
Expected: ALL PASS

### Step 6: Commit

```bash
git add mate/trajectory/monitor.py tests/trajectory/test_monitor.py
git commit -m "feat: add replay_cache support to ModelMonitor for branch runs"
```

---

## Task 4: AgentPipe 优雅降级

**Files:**
- Modify: `mate/trajectory/pipe.py`
- Test: `tests/trajectory/test_pipe.py`

### Step 1: 编写 failing test

```python
# tests/trajectory/test_pipe.py 中新增

@pytest.mark.asyncio
async def test_pipe_returns_partial_result_on_mas_failure():
    """AgentPipe returns partial result (status='failed') instead of raising on MAS failure."""
    # 使用配置使 MAS 命令会失败的场景
    config = AgentPipeConfig(
        mas_command_template="python -c \"import sys; sys.exit(1)\"",
        config_template={"llm": {"base_url": "placeholder"}, "agents": {}},
        model_mapping={},
        timeout=10.0,
    )
    pipe = AgentPipe(config=config, backend=mock_backend)
    result = await pipe.run(prompt="test", reward_provider=mock_reward, allow_partial=True)
    assert result.status == "failed"
    assert result.failure_info is not None
    assert "exit_code" in result.failure_info
```

### Step 2: 运行测试验证失败

Run: `cd /home/cxb/MATE-reboot && python -m pytest tests/trajectory/test_pipe.py -v -k "partial_result"`
Expected: FAIL

### Step 3: 修改 AgentPipe.run()

在 `AgentPipe.run()` 签名中添加 `allow_partial: bool = False` 参数。

当 `allow_partial=True` 且 MAS 进程非零退出时，不 raise，而是返回一个 `status="failed"` 的 `EpisodeResult`：

```python
if exit_code != 0:
    if not allow_partial:
        raise RuntimeError(f"MAS process exited with non-zero exit code {exit_code}")
    trajectory = self._collector.build(buffer=monitor.get_buffer(), episode_id=episode_id)
    return EpisodeResult(
        trajectory=trajectory,
        rewards={},
        final_reward=None,
        metadata={"exit_code": exit_code},
        status="failed",
        failure_info={"exit_code": exit_code, "reason": "MAS non-zero exit"},
    )
```

### Step 4: 运行测试验证通过

Run: `cd /home/cxb/MATE-reboot && python -m pytest tests/trajectory/test_pipe.py -v`
Expected: ALL PASS（`allow_partial` 默认 False，不影响现有测试）

### Step 5: 全量回归

Run: `cd /home/cxb/MATE-reboot && python -m pytest tests/ -q --timeout=120`
Expected: ALL PASS

### Step 6: Commit

```bash
git add mate/trajectory/pipe.py tests/trajectory/test_pipe.py
git commit -m "feat: add allow_partial to AgentPipe for graceful degradation"
```

---

## Task 5: tree_rollout 函数

**Files:**
- Create: `mate/trajectory/tree.py`
- Test: `tests/trajectory/test_tree.py`

这是 V0.2 的核心任务，需要编排 pilot run + branch runs。

### Step 1: 编写 failing test（基础场景）

```python
# tests/trajectory/test_tree.py

import pytest
from unittest.mock import AsyncMock, patch
from mate.trajectory.datatypes import (
    BranchResult, EpisodeResult, EpisodeTrajectory,
    InteractionRecord, TreeEpisodeResult, TurnData,
)
from mate.trajectory.pipe import AgentPipeConfig
from mate.trajectory.backend import InferenceBackend


def _make_episode_result(episode_id: str, agent_turns: dict[str, int]) -> EpisodeResult:
    """Helper: create an EpisodeResult with N turns per agent."""
    trajectories = {}
    for role, n_turns in agent_turns.items():
        trajectories[role] = [
            TurnData(
                agent_role=role, turn_index=i,
                messages=[{"role": "user", "content": f"turn-{i}"}],
                response_text=f"response-{role}-{i}",
                token_ids=[i * 10 + j for j in range(5)],
                logprobs=[-0.1] * 5,
                finish_reason="stop",
                timestamp=float(i),
            )
            for i in range(n_turns)
        ]
    return EpisodeResult(
        trajectory=EpisodeTrajectory(episode_id=episode_id, agent_trajectories=trajectories),
        rewards={role: 1.0 for role in agent_turns},
        final_reward=1.0,
    )


@pytest.mark.asyncio
async def test_tree_rollout_structure():
    from mate.trajectory.tree import tree_rollout

    pilot = _make_episode_result("pilot-001", {"verifier": 2, "searcher": 1, "answerer": 1})
    branch_results = [
        _make_episode_result(f"branch-{i}", {"verifier": 2, "searcher": 1, "answerer": 1})
        for i in range(4)
    ]

    # Mock AgentPipe.run to return pilot on first call, then branches
    all_results = [pilot] + branch_results
    call_idx = {"i": 0}

    async def mock_pipe_run(self, prompt, reward_provider, **kwargs):
        result = all_results[call_idx["i"]]
        call_idx["i"] += 1
        return result

    with patch("mate.trajectory.tree.AgentPipe.run", mock_pipe_run):
        tree_result = await tree_rollout(
            prompt="test question",
            reward_provider=lambda traj: {"agent_rewards": {}, "final_reward": 0.0},
            config=AgentPipeConfig(
                mas_command_template="echo test",
                config_template={},
                model_mapping={},
            ),
            backend=AsyncMock(spec=InferenceBackend),
            k_branches=1,
        )

    assert isinstance(tree_result, TreeEpisodeResult)
    assert tree_result.pilot_result.trajectory.episode_id == "pilot-001"
    assert tree_result.prompt == "test question"
    assert len(tree_result.branch_results) == 4  # 4 turns × k=1
    for br in tree_result.branch_results:
        assert isinstance(br, BranchResult)
        assert br.parent_episode_id == "pilot-001"
```

### Step 2: 运行测试验证失败

Run: `cd /home/cxb/MATE-reboot && python -m pytest tests/trajectory/test_tree.py -v -k "tree_rollout_structure"`
Expected: FAIL — `ModuleNotFoundError: No module named 'mate.trajectory.tree'`

### Step 3: 实现 tree_rollout

```python
# mate/trajectory/tree.py
from __future__ import annotations

import asyncio
import logging
from typing import Any

from .backend import InferenceBackend
from .datatypes import BranchResult, EpisodeResult, TreeEpisodeResult
from .pipe import AgentPipe, AgentPipeConfig
from .replay_cache import ReplayCache
from .reward import RewardProvider

_LOGGER = logging.getLogger(__name__)


async def tree_rollout(
    prompt: str,
    reward_provider: RewardProvider,
    config: AgentPipeConfig,
    backend: InferenceBackend,
    k_branches: int = 3,
    max_concurrent_branches: int | None = None,
) -> TreeEpisodeResult:
    if k_branches < 1:
        raise ValueError("k_branches must be >= 1")

    # 1. Pilot run
    pilot_pipe = AgentPipe(config=config, backend=backend)
    pilot_result = await pilot_pipe.run(prompt=prompt, reward_provider=reward_provider)
    pilot_buffer = pilot_pipe.last_buffer()

    if pilot_result.status != "success" or not pilot_buffer:
        return TreeEpisodeResult(
            pilot_result=pilot_result,
            branch_results=[],
            prompt=prompt,
            tree_metadata={"reason": "pilot_failed_or_empty"},
        )

    # 2. Determine branch points (every agent turn in pilot)
    sorted_records = sorted(pilot_buffer, key=lambda r: r.timestamp)
    branch_points = [
        {"global_position": i, "agent_role": rec.agent_role, "turn_index": rec.turn_index}
        for i, rec in enumerate(sorted_records)
    ]

    # 3. Run branches
    semaphore = asyncio.Semaphore(max_concurrent_branches) if max_concurrent_branches else None

    async def run_branch(bp: dict) -> list[BranchResult]:
        cache = ReplayCache.from_buffer(pilot_buffer, branch_at_global_position=bp["global_position"])
        results = []
        for _ in range(k_branches):
            branch_pipe = AgentPipe(config=config, backend=backend, replay_cache=cache)
            try:
                if semaphore:
                    async with semaphore:
                        br_result = await branch_pipe.run(
                            prompt=prompt, reward_provider=reward_provider, allow_partial=True,
                        )
                else:
                    br_result = await branch_pipe.run(
                        prompt=prompt, reward_provider=reward_provider, allow_partial=True,
                    )
            except Exception as exc:
                _LOGGER.warning("Branch at %s failed: %s", bp, exc)
                continue
            results.append(BranchResult(
                episode_result=br_result,
                branch_turn=bp["global_position"],
                branch_agent_role=bp["agent_role"],
                parent_episode_id=pilot_result.trajectory.episode_id,
            ))
        return results

    branch_tasks = [run_branch(bp) for bp in branch_points]
    gathered = await asyncio.gather(*branch_tasks, return_exceptions=True)

    all_branches: list[BranchResult] = []
    for item in gathered:
        if isinstance(item, list):
            all_branches.extend(item)
        elif isinstance(item, Exception):
            _LOGGER.warning("Branch task failed: %s", item)

    total_turns = sum(
        len(turns)
        for turns in pilot_result.trajectory.agent_trajectories.values()
    )
    return TreeEpisodeResult(
        pilot_result=pilot_result,
        branch_results=all_branches,
        prompt=prompt,
        tree_metadata={
            "n_branch_points": len(branch_points),
            "k_branches": k_branches,
            "total_branches_collected": len(all_branches),
            "pilot_total_turns": total_turns,
        },
    )
```

**注意**：此实现需要 `AgentPipe` 暴露 `last_buffer()` 方法（返回最近一次 run 的 Monitor buffer），以及接受 `replay_cache` 参数传给 Monitor。这些修改在 Step 3 中一并完成。

需要对 `AgentPipe` 做的修改：
1. `__init__` 添加 `replay_cache: ReplayCache | None = None`
2. `run()` 创建 Monitor 时传入 `replay_cache`
3. 添加 `last_buffer()` 方法，返回最近一次 run 结束时的 buffer 快照

### Step 4: 运行测试验证通过

Run: `cd /home/cxb/MATE-reboot && python -m pytest tests/trajectory/test_tree.py -v`
Expected: PASS

### Step 5: 添加更多测试

在 `tests/trajectory/test_tree.py` 中继续添加：

```python
@pytest.mark.asyncio
async def test_tree_rollout_pilot_failure_returns_empty_tree():
    """When pilot fails, tree_rollout returns a tree with no branches."""
    ...

@pytest.mark.asyncio
async def test_tree_rollout_branch_failure_is_graceful():
    """When a branch fails, it's skipped and other branches continue."""
    ...

@pytest.mark.asyncio
async def test_tree_rollout_k_branches_param():
    """k_branches controls how many branches per turn."""
    ...
```

### Step 6: 全量回归

Run: `cd /home/cxb/MATE-reboot && python -m pytest tests/ -q --timeout=120`
Expected: ALL PASS

### Step 7: Commit

```bash
git add mate/trajectory/tree.py mate/trajectory/pipe.py tests/trajectory/test_tree.py
git commit -m "feat: add tree_rollout for replay-based branching sampling"
```

---

## Task 6: Package 导出与版本更新

**Files:**
- Modify: `mate/trajectory/__init__.py`
- Modify: `pyproject.toml`

### Step 1: 更新 `__init__.py`

在现有 imports 后添加：

```python
from .datatypes import BranchResult, TreeEpisodeResult
from .replay_cache import ReplayCache
from .tree import tree_rollout
```

在 `__all__` 中添加：`"BranchResult"`, `"TreeEpisodeResult"`, `"ReplayCache"`, `"tree_rollout"`

### Step 2: 更新版本号

在 `pyproject.toml` 中将 `version` 从当前值改为 `"0.2.0"`。

### Step 3: 验证导入

Run: `cd /home/cxb/MATE-reboot && python -c "from mate.trajectory import tree_rollout, TreeEpisodeResult, BranchResult, ReplayCache; print('OK')"`
Expected: `OK`

### Step 4: 全量回归

Run: `cd /home/cxb/MATE-reboot && python -m pytest tests/ -q --timeout=120`
Expected: ALL PASS

### Step 5: Commit

```bash
git add mate/trajectory/__init__.py pyproject.toml
git commit -m "chore: export V0.2 tree branching APIs and bump version to 0.2.0"
```

---

## Task 7: 集成测试

**Files:**
- Create: `tests/trajectory/test_tree_integration.py`

### Step 1: 编写集成测试

端到端测试：使用 mock MAS 脚本（Python 子进程），验证 tree_rollout 能正确完成 pilot + branch runs，ReplayCache 缓存命中，TreeEpisodeResult 结构完整。

```python
# tests/trajectory/test_tree_integration.py

"""Integration test: tree_rollout with a real subprocess MAS (mock script)."""

import pytest
import textwrap
from pathlib import Path
from mate.trajectory import (
    AgentPipeConfig, ModelMappingEntry, VLLMBackend, tree_rollout,
    TreeEpisodeResult, BranchResult,
)


@pytest.fixture
def mock_mas_script(tmp_path: Path) -> Path:
    """Create a minimal MAS script that makes 2 LLM calls."""
    script = tmp_path / "mock_mas.py"
    script.write_text(textwrap.dedent('''
        import sys, json, urllib.request

        config_path = sys.argv[1]
        question = sys.argv[2]

        with open(config_path) as f:
            import yaml
            config = yaml.safe_load(f)

        base_url = config["llm"]["base_url"]

        for role in ["verifier", "answerer"]:
            req = urllib.request.Request(
                f"{base_url}/chat/completions",
                data=json.dumps({
                    "model": role,
                    "messages": [{"role": "user", "content": f"{role}: {question}"}],
                }).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req) as resp:
                result = json.loads(resp.read())
                print(f"{role}: {result['choices'][0]['message']['content']}")
    '''))
    return script


# Test implementation depends on mock backend setup.
# The test verifies:
# 1. tree_rollout completes without error
# 2. TreeEpisodeResult has pilot + branches
# 3. Branch results have correct parent_episode_id
# 4. Replayed turns are marked in buffer
```

### Step 2: 运行集成测试

Run: `cd /home/cxb/MATE-reboot && python -m pytest tests/trajectory/test_tree_integration.py -v --timeout=60`
Expected: PASS

### Step 3: Commit

```bash
git add tests/trajectory/test_tree_integration.py
git commit -m "test: add tree_rollout integration test with mock MAS"
```

---

## Task 8: 更新设计文档（融入审核修正）

**Files:**
- Modify: `docs/plans/2026-03-09-trajectory-engine-v02-design-direction.md`

### Step 1: 融入审核发现

更新设计方向文档，在相关章节中：

1. **§2.1**：添加工具调用确定性假设声明，以及缓存未命中时的降级语义（标记 `degraded`，不静默回退）
2. **§2.2**：简化为 `tree_rollout` 函数描述，删除 BranchCoordinator/BranchStrategy 策略模式
3. **§2.3**：添加完整的 uid 赋值规则伪代码，覆盖重放 turn / 分支点 turn / 后续 turn；删除 `shared_prefix_turns` 字段
4. **§2.3**：添加 messages_hash 规范（sha256 + json.dumps sort_keys）
5. **§3.1**：明确 token 估算策略（`len(json.dumps(messages)) // 3`）和 `max_context_tokens` 来源
6. **§3.2**：添加树状分支失败模式矩阵
7. **§4.1**：添加固定 LLM 调用数的对照实验

### Step 2: Commit

```bash
git add docs/plans/2026-03-09-trajectory-engine-v02-design-direction.md
git commit -m "docs: incorporate design review findings into V0.2 direction doc"
```

---

## 依赖关系与执行顺序

```
Task 1 (数据结构) ──┐
                    ├── Task 5 (tree_rollout) ── Task 6 (exports) ── Task 7 (集成测试)
Task 2 (ReplayCache)┤
                    │
Task 3 (Monitor)  ──┘
Task 4 (优雅降级)  ──── 可独立执行（但 Task 5 依赖 allow_partial）
Task 8 (文档更新)  ──── 可随时执行
```

建议执行顺序：`1 → 2 → 3 → 4 → 5 → 6 → 7 → 8`

---

## uid 赋值规则伪代码（审核 BLOCKER-2 修正）

```python
def assign_uids_for_tree(tree_result: TreeEpisodeResult, prompt_group_id: str) -> list[dict]:
    """将 TreeEpisodeResult 转为训练记录，每条记录包含 uid 字段。"""
    records = []

    # 1. Pilot turns: 使用标准 uid（与 V0 兼容）
    for role, turns in tree_result.pilot_result.trajectory.agent_trajectories.items():
        agent_idx = ROLE_INDEX[role]
        for turn in turns:
            records.append({
                "uid": f"{prompt_group_id}:{agent_idx}",
                "source": "pilot",
                **turn_to_record(turn),
            })

    # 2. Branch turns
    for br in tree_result.branch_results:
        for role, turns in br.episode_result.trajectory.agent_trajectories.items():
            agent_idx = ROLE_INDEX[role]
            for turn in turns:
                is_replayed = turn.metadata.get("replayed", False)
                if is_replayed:
                    continue  # 重放 turn 不进入训练 batch（与 pilot 重复）

                # 分支点 turn 和后续 turn 使用分组 uid
                records.append({
                    "uid": f"{prompt_group_id}:{agent_idx}:b{br.branch_turn}",
                    "source": "branch",
                    "branch_turn": br.branch_turn,
                    **turn_to_record(turn),
                })

    return records
```

分组语义：同一 `uid` 下的记录属于同一 GRPO 分组，共享相同的上下文前缀。
