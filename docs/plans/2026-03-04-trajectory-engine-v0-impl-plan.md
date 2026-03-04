# Trajectory Engine V0 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement AgentPipe trajectory collection module and validate end-to-end with DrMAS Search (OrchRL) scenario.

**Architecture:** Strategy-pattern ModelMonitor with pluggable InferenceBackend (VLLMBackend for V0), MASLauncher for subprocess lifecycle, TrajectoryCollector for data assembly, RewardWorker for reward computation, AgentPipe as top-level orchestrator. Design doc: `docs/plans/2026-03-04-trajectory-engine-v0-design.md`.

**Tech Stack:** Python 3.10+, aiohttp (HTTP server), httpx (async HTTP client), pytest + pytest-asyncio (testing), PyYAML (config)

**Workspace:** `/home/cxb/MATE-reboot`

**Reference code:**
- OrchRL Search MAS: `/home/cxb/OrchRL/examples/mas_app/search/`
- SWE-agent recipe (ModelProxy reference): `/home/cxb/rl_framework/verl/recipe/swe_agent/model_proxy/proxy_server.py`
- DrMAS advantage: `/home/cxb/multi-agent/DrMAS/verl/trainer/ppo/core_algos.py`

---

## Task 1: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `mate/__init__.py`
- Create: `mate/trajectory/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/trajectory/__init__.py`

**Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "mate"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "aiohttp>=3.9.0",
    "httpx>=0.27.0",
    "PyYAML>=6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23.0",
    "pytest-timeout>=2.2.0",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

**Step 2: Create package directories**

```bash
mkdir -p mate/trajectory tests/trajectory
touch mate/__init__.py mate/trajectory/__init__.py tests/__init__.py tests/trajectory/__init__.py
```

**Step 3: Install in dev mode and verify**

Run: `cd /home/cxb/MATE-reboot && pip install -e ".[dev]"`
Expected: successful install, `import mate` works.

**Step 4: Commit**

```bash
git add pyproject.toml mate/ tests/
git commit -m "feat: scaffold mate package with dev dependencies"
```

---

## Task 2: Data Structures

**Files:**
- Create: `mate/trajectory/datatypes.py`
- Create: `tests/trajectory/test_datatypes.py`

**Step 1: Write the failing test**

```python
# tests/trajectory/test_datatypes.py
from mate.trajectory.datatypes import (
    ModelMappingEntry,
    ModelRequest,
    ModelResponse,
    InteractionRecord,
    TurnData,
    EpisodeTrajectory,
    EpisodeResult,
)


def test_model_request_creation():
    req = ModelRequest(
        request_id="r1",
        agent_role="verifier",
        messages=[{"role": "user", "content": "hello"}],
        generation_params={"temperature": 0.7},
    )
    assert req.agent_role == "verifier"
    assert req.request_id == "r1"


def test_model_response_with_none_tokens():
    resp = ModelResponse(content="hi", token_ids=None, logprobs=None, finish_reason="stop")
    assert resp.token_ids is None
    assert resp.finish_reason == "stop"


def test_interaction_record_fields():
    rec = InteractionRecord(
        agent_role="searcher",
        turn_index=0,
        timestamp=1000.0,
        messages=[{"role": "user", "content": "q"}],
        generation_params={"temperature": 0.6},
        response_text="answer",
        token_ids=[1, 2, 3],
        logprobs=[-0.1, -0.2, -0.3],
        finish_reason="stop",
        episode_id="ep1",
        metadata={},
    )
    assert rec.agent_role == "searcher"
    assert len(rec.token_ids) == 3


def test_episode_trajectory_agent_grouping():
    t1 = TurnData(
        agent_role="verifier", turn_index=0,
        messages=[], response_text="no", token_ids=None,
        logprobs=None, finish_reason="stop", timestamp=1.0, metadata={},
    )
    t2 = TurnData(
        agent_role="searcher", turn_index=0,
        messages=[], response_text="query", token_ids=None,
        logprobs=None, finish_reason="stop", timestamp=2.0, metadata={},
    )
    traj = EpisodeTrajectory(
        episode_id="ep1",
        agent_trajectories={"verifier": [t1], "searcher": [t2]},
        metadata={},
    )
    assert set(traj.agent_trajectories.keys()) == {"verifier", "searcher"}


def test_episode_result_with_per_turn_rewards():
    traj = EpisodeTrajectory(episode_id="ep1", agent_trajectories={}, metadata={})
    result = EpisodeResult(
        trajectory=traj,
        rewards={"verifier": [0.5, 0.8], "searcher": 1.0},
        final_reward=1.0,
        metadata={},
    )
    assert isinstance(result.rewards["verifier"], list)
    assert isinstance(result.rewards["searcher"], float)
```

**Step 2: Run test to verify it fails**

Run: `cd /home/cxb/MATE-reboot && python -m pytest tests/trajectory/test_datatypes.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write the implementation**

```python
# mate/trajectory/datatypes.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelMappingEntry:
    actual_model: str | None = None
    backend_url: str | None = None


@dataclass
class ModelRequest:
    request_id: str
    agent_role: str
    messages: list[dict[str, Any]]
    generation_params: dict[str, Any]


@dataclass
class ModelResponse:
    content: str
    token_ids: list[int] | None
    logprobs: list[float] | None
    finish_reason: str


@dataclass
class InteractionRecord:
    agent_role: str
    turn_index: int
    timestamp: float
    messages: list[dict[str, Any]]
    generation_params: dict[str, Any]
    response_text: str
    token_ids: list[int] | None
    logprobs: list[float] | None
    finish_reason: str
    episode_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TurnData:
    agent_role: str
    turn_index: int
    messages: list[dict[str, Any]]
    response_text: str
    token_ids: list[int] | None
    logprobs: list[float] | None
    finish_reason: str
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeTrajectory:
    episode_id: str
    agent_trajectories: dict[str, list[TurnData]]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeResult:
    trajectory: EpisodeTrajectory
    rewards: dict[str, float | list[float]]
    final_reward: float | None
    metadata: dict[str, Any] = field(default_factory=dict)
```

**Step 4: Run test to verify it passes**

Run: `cd /home/cxb/MATE-reboot && python -m pytest tests/trajectory/test_datatypes.py -v`
Expected: all 5 tests PASS

**Step 5: Commit**

```bash
git add mate/trajectory/datatypes.py tests/trajectory/test_datatypes.py
git commit -m "feat: add trajectory data structures"
```

---

## Task 3: InferenceBackend Interface + VLLMBackend

**Files:**
- Create: `mate/trajectory/backend.py`
- Create: `tests/trajectory/test_backend.py`

**Step 1: Write the failing test**

```python
# tests/trajectory/test_backend.py
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase, unittest_run_loop

from mate.trajectory.backend import InferenceBackend, VLLMBackend
from mate.trajectory.datatypes import ModelRequest, ModelResponse


def test_inference_backend_is_abstract():
    with pytest.raises(TypeError):
        InferenceBackend()


async def test_vllm_backend_generate(aiohttp_server):
    """Spin up a fake vLLM server, verify VLLMBackend correctly forwards and parses."""

    async def fake_chat_completions(request: web.Request) -> web.Response:
        body = await request.json()
        assert body["model"] == "Qwen3-4B"
        assert body["logprobs"] is True
        return web.json_response({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "test response"},
                "finish_reason": "stop",
                "logprobs": {
                    "content": [
                        {"token": "test", "logprob": -0.5, "bytes": None},
                        {"token": " response", "logprob": -0.3, "bytes": None},
                    ]
                },
            }],
        })

    app = web.Application()
    app.router.add_post("/v1/chat/completions", fake_chat_completions)
    server = await aiohttp_server(app)

    backend = VLLMBackend(
        backend_url=f"http://localhost:{server.port}",
        actual_model="Qwen3-4B",
    )
    req = ModelRequest(
        request_id="r1",
        agent_role="verifier",
        messages=[{"role": "user", "content": "hello"}],
        generation_params={"temperature": 0.7, "max_tokens": 100},
    )
    resp = await backend.generate(req)

    assert isinstance(resp, ModelResponse)
    assert resp.content == "test response"
    assert resp.finish_reason == "stop"
    assert resp.logprobs == [-0.5, -0.3]


async def test_vllm_backend_injects_logprobs(aiohttp_server):
    """Verify VLLMBackend always sets logprobs=True in forwarded request."""
    captured = {}

    async def capture_handler(request: web.Request) -> web.Response:
        captured["body"] = await request.json()
        return web.json_response({
            "choices": [{
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
                "logprobs": None,
            }],
        })

    app = web.Application()
    app.router.add_post("/v1/chat/completions", capture_handler)
    server = await aiohttp_server(app)

    backend = VLLMBackend(backend_url=f"http://localhost:{server.port}")
    req = ModelRequest(
        request_id="r1", agent_role="a",
        messages=[{"role": "user", "content": "x"}],
        generation_params={},
    )
    await backend.generate(req)
    assert captured["body"]["logprobs"] is True


async def test_vllm_backend_replaces_model(aiohttp_server):
    """Verify VLLMBackend replaces model field with actual_model."""
    captured = {}

    async def capture_handler(request: web.Request) -> web.Response:
        captured["body"] = await request.json()
        return web.json_response({
            "choices": [{
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
                "logprobs": None,
            }],
        })

    app = web.Application()
    app.router.add_post("/v1/chat/completions", capture_handler)
    server = await aiohttp_server(app)

    backend = VLLMBackend(
        backend_url=f"http://localhost:{server.port}",
        actual_model="real-model-name",
    )
    req = ModelRequest(
        request_id="r1", agent_role="verifier",
        messages=[{"role": "user", "content": "x"}],
        generation_params={"model": "verifier"},
    )
    await backend.generate(req)
    assert captured["body"]["model"] == "real-model-name"
```

**Step 2: Run test to verify it fails**

Run: `cd /home/cxb/MATE-reboot && python -m pytest tests/trajectory/test_backend.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write the implementation**

```python
# mate/trajectory/backend.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import httpx

from .datatypes import ModelRequest, ModelResponse


class InferenceBackend(ABC):
    @abstractmethod
    async def generate(self, request: ModelRequest) -> ModelResponse:
        ...


class VLLMBackend(InferenceBackend):
    """HTTP reverse-proxy backend for standalone vLLM/SGLang services."""

    def __init__(
        self,
        backend_url: str,
        actual_model: str | None = None,
        timeout: float = 120.0,
    ):
        self.backend_url = backend_url.rstrip("/")
        self.actual_model = actual_model
        self.timeout = timeout

    async def generate(self, request: ModelRequest) -> ModelResponse:
        payload: dict[str, Any] = {
            "messages": request.messages,
            "logprobs": True,
            "top_logprobs": 1,
            **request.generation_params,
        }
        if self.actual_model:
            payload["model"] = self.actual_model
        elif "model" not in payload:
            payload["model"] = request.agent_role

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.backend_url}/v1/chat/completions",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        choice = data["choices"][0]
        content = choice["message"]["content"] or ""
        finish_reason = choice.get("finish_reason", "stop")

        token_ids = None
        logprobs = None
        lp_data = choice.get("logprobs")
        if lp_data and isinstance(lp_data.get("content"), list):
            logprobs = [item["logprob"] for item in lp_data["content"]]

        return ModelResponse(
            content=content,
            token_ids=token_ids,
            logprobs=logprobs,
            finish_reason=finish_reason,
        )
```

**Step 4: Run test to verify it passes**

Run: `cd /home/cxb/MATE-reboot && python -m pytest tests/trajectory/test_backend.py -v`
Expected: all 4 tests PASS

**Step 5: Commit**

```bash
git add mate/trajectory/backend.py tests/trajectory/test_backend.py
git commit -m "feat: add InferenceBackend interface and VLLMBackend"
```

---

## Task 4: ModelMonitor HTTP Server

**Files:**
- Create: `mate/trajectory/monitor.py`
- Create: `tests/trajectory/test_monitor.py`

**Step 1: Write the failing test**

```python
# tests/trajectory/test_monitor.py
import pytest
import httpx

from mate.trajectory.monitor import ModelMonitor
from mate.trajectory.backend import InferenceBackend
from mate.trajectory.datatypes import (
    ModelMappingEntry, ModelRequest, ModelResponse, InteractionRecord,
)


class EchoBackend(InferenceBackend):
    """Test backend that echoes the agent_role as content."""
    async def generate(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            content=f"echo:{request.agent_role}",
            token_ids=[101, 102],
            logprobs=[-0.1, -0.2],
            finish_reason="stop",
        )


@pytest.fixture
async def monitor():
    mapping = {
        "verifier": ModelMappingEntry(actual_model="test-model"),
        "searcher": ModelMappingEntry(actual_model="test-model"),
    }
    mon = ModelMonitor(backend=EchoBackend(), model_mapping=mapping)
    port = await mon.start()
    yield mon, port
    await mon.stop()


async def test_monitor_routes_by_model_field(monitor):
    mon, port = monitor
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            json={
                "model": "verifier",
                "messages": [{"role": "user", "content": "hello"}],
                "temperature": 0.5,
            },
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["choices"][0]["message"]["content"] == "echo:verifier"


async def test_monitor_collects_interaction_record(monitor):
    mon, port = monitor
    async with httpx.AsyncClient() as client:
        await client.post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            json={
                "model": "searcher",
                "messages": [{"role": "user", "content": "query"}],
            },
        )
    buffer = mon.get_buffer()
    assert len(buffer) == 1
    rec = buffer[0]
    assert isinstance(rec, InteractionRecord)
    assert rec.agent_role == "searcher"
    assert rec.response_text == "echo:searcher"
    assert rec.token_ids == [101, 102]
    assert rec.logprobs == [-0.1, -0.2]
    assert rec.turn_index == 0


async def test_monitor_increments_turn_index(monitor):
    mon, port = monitor
    async with httpx.AsyncClient() as client:
        for _ in range(3):
            await client.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                json={"model": "verifier", "messages": [{"role": "user", "content": "q"}]},
            )
    buffer = mon.get_buffer()
    assert len(buffer) == 3
    assert [r.turn_index for r in buffer] == [0, 1, 2]


async def test_monitor_unknown_agent_returns_error(monitor):
    mon, port = monitor
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            json={"model": "unknown_agent", "messages": []},
        )
    assert resp.status_code == 400


async def test_monitor_clear_buffer(monitor):
    mon, port = monitor
    async with httpx.AsyncClient() as client:
        await client.post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            json={"model": "verifier", "messages": [{"role": "user", "content": "q"}]},
        )
    assert len(mon.get_buffer()) == 1
    mon.clear_buffer()
    assert len(mon.get_buffer()) == 0
```

**Step 2: Run test to verify it fails**

Run: `cd /home/cxb/MATE-reboot && python -m pytest tests/trajectory/test_monitor.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write the implementation**

```python
# mate/trajectory/monitor.py
from __future__ import annotations

import time
import uuid
from typing import Any

from aiohttp import web

from .backend import InferenceBackend
from .datatypes import (
    InteractionRecord,
    ModelMappingEntry,
    ModelRequest,
    ModelResponse,
)


class ModelMonitor:
    def __init__(
        self,
        backend: InferenceBackend,
        model_mapping: dict[str, ModelMappingEntry],
        episode_id: str | None = None,
    ):
        self._backend = backend
        self._model_mapping = model_mapping
        self._episode_id = episode_id or uuid.uuid4().hex
        self._buffer: list[InteractionRecord] = []
        self._turn_counters: dict[str, int] = {}
        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self._port: int | None = None

    async def start(self, host: str = "127.0.0.1", port: int = 0) -> int:
        self._app = web.Application()
        self._app.router.add_post("/v1/chat/completions", self._handle_chat_completions)
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, host, port)
        await self._site.start()
        # Resolve actual bound port
        sockets = self._site._server.sockets
        self._port = sockets[0].getsockname()[1]
        return self._port

    async def stop(self):
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
            self._site = None
            self._app = None

    def get_buffer(self) -> list[InteractionRecord]:
        return list(self._buffer)

    def clear_buffer(self):
        self._buffer.clear()
        self._turn_counters.clear()

    async def _handle_chat_completions(self, http_request: web.Request) -> web.Response:
        try:
            body = await http_request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)

        agent_role = body.get("model", "")
        if agent_role not in self._model_mapping:
            return web.json_response(
                {"error": f"unknown agent role: {agent_role}"},
                status=400,
            )

        messages = body.get("messages", [])
        gen_params = {
            k: v for k, v in body.items()
            if k not in ("model", "messages")
        }

        request_id = uuid.uuid4().hex
        model_request = ModelRequest(
            request_id=request_id,
            agent_role=agent_role,
            messages=messages,
            generation_params=gen_params,
        )

        try:
            response = await self._backend.generate(model_request)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=502)

        turn_index = self._turn_counters.get(agent_role, 0)
        self._turn_counters[agent_role] = turn_index + 1

        record = InteractionRecord(
            agent_role=agent_role,
            turn_index=turn_index,
            timestamp=time.time(),
            messages=messages,
            generation_params=gen_params,
            response_text=response.content,
            token_ids=response.token_ids,
            logprobs=response.logprobs,
            finish_reason=response.finish_reason,
            episode_id=self._episode_id,
            metadata={},
        )
        self._buffer.append(record)

        openai_response = {
            "id": f"chatcmpl-{request_id}",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": response.content},
                "finish_reason": response.finish_reason,
            }],
        }
        return web.json_response(openai_response)
```

**Step 4: Run test to verify it passes**

Run: `cd /home/cxb/MATE-reboot && python -m pytest tests/trajectory/test_monitor.py -v`
Expected: all 5 tests PASS

**Step 5: Commit**

```bash
git add mate/trajectory/monitor.py tests/trajectory/test_monitor.py
git commit -m "feat: add ModelMonitor HTTP server with strategy-pattern backend"
```

---

## Task 5: MASLauncher

**Files:**
- Create: `mate/trajectory/launcher.py`
- Create: `tests/trajectory/test_launcher.py`

**Step 1: Write the failing test**

```python
# tests/trajectory/test_launcher.py
import json
from pathlib import Path

import pytest
import yaml

from mate.trajectory.launcher import MASLauncher


@pytest.fixture
def tmp_config(tmp_path):
    config = {
        "llm": {
            "base_url": "http://original:8000/v1",
            "model": "Qwen3-4B",
        },
        "agents": {
            "verifier": {"temperature": 0.2},
            "searcher": {"temperature": 0.6},
            "answerer": {"temperature": 0.4},
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(config))
    return path, config


def test_prepare_config_replaces_base_url(tmp_config):
    path, config = tmp_config
    launcher = MASLauncher()
    out = launcher.prepare_config(
        config_template=config,
        monitor_url="http://127.0.0.1:19000/v1",
        agent_roles=["verifier", "searcher", "answerer"],
    )
    result = yaml.safe_load(out.read_text())
    assert result["llm"]["base_url"] == "http://127.0.0.1:19000/v1"


def test_prepare_config_injects_agent_model_names(tmp_config):
    path, config = tmp_config
    launcher = MASLauncher()
    out = launcher.prepare_config(
        config_template=config,
        monitor_url="http://127.0.0.1:19000/v1",
        agent_roles=["verifier", "searcher", "answerer"],
    )
    result = yaml.safe_load(out.read_text())
    assert result["agents"]["verifier"]["model"] == "verifier"
    assert result["agents"]["searcher"]["model"] == "searcher"
    assert result["agents"]["answerer"]["model"] == "answerer"


def test_prepare_config_preserves_other_fields(tmp_config):
    path, config = tmp_config
    launcher = MASLauncher()
    out = launcher.prepare_config(
        config_template=config,
        monitor_url="http://127.0.0.1:19000/v1",
        agent_roles=["verifier", "searcher", "answerer"],
    )
    result = yaml.safe_load(out.read_text())
    assert result["agents"]["verifier"]["temperature"] == 0.2
    assert result["agents"]["searcher"]["temperature"] == 0.6


async def test_launch_and_wait_success():
    launcher = MASLauncher()
    process = await launcher.launch(
        command="python -c \"print('hello')\"",
    )
    exit_code = await launcher.wait(process, timeout=10.0)
    assert exit_code == 0


async def test_launch_and_wait_timeout():
    launcher = MASLauncher()
    process = await launcher.launch(command="sleep 60")
    exit_code = await launcher.wait(process, timeout=0.5)
    assert exit_code != 0  # killed
```

**Step 2: Run test to verify it fails**

Run: `cd /home/cxb/MATE-reboot && python -m pytest tests/trajectory/test_launcher.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write the implementation**

```python
# mate/trajectory/launcher.py
from __future__ import annotations

import asyncio
import copy
import tempfile
from pathlib import Path
from typing import Any

import yaml


class MASLauncher:
    def __init__(self, work_dir: str | Path | None = None):
        self._work_dir = Path(work_dir) if work_dir else None
        self._temp_files: list[Path] = []

    def prepare_config(
        self,
        config_template: dict[str, Any],
        monitor_url: str,
        agent_roles: list[str],
    ) -> Path:
        config = copy.deepcopy(config_template)

        if "llm" in config:
            config["llm"]["base_url"] = monitor_url

        agents_cfg = config.get("agents", {})
        for role in agent_roles:
            if role in agents_cfg:
                agents_cfg[role]["model"] = role
            else:
                agents_cfg[role] = {"model": role}

        fd = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, prefix="mate_mas_",
        )
        yaml.dump(config, fd, default_flow_style=False)
        fd.close()
        path = Path(fd.name)
        self._temp_files.append(path)
        return path

    async def launch(
        self,
        command: str,
        env_vars: dict[str, str] | None = None,
    ) -> asyncio.subprocess.Process:
        import os
        env = dict(os.environ)
        if env_vars:
            env.update(env_vars)

        process = await asyncio.create_subprocess_shell(
            command,
            cwd=str(self._work_dir) if self._work_dir else None,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        return process

    async def wait(
        self, process: asyncio.subprocess.Process, timeout: float | None = None,
    ) -> int:
        try:
            await asyncio.wait_for(process.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
        return process.returncode

    def cleanup(self):
        for path in self._temp_files:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        self._temp_files.clear()
```

**Step 4: Run test to verify it passes**

Run: `cd /home/cxb/MATE-reboot && python -m pytest tests/trajectory/test_launcher.py -v`
Expected: all 5 tests PASS

**Step 5: Commit**

```bash
git add mate/trajectory/launcher.py tests/trajectory/test_launcher.py
git commit -m "feat: add MASLauncher for subprocess lifecycle management"
```

---

## Task 6: TrajectoryCollector

**Files:**
- Create: `mate/trajectory/collector.py`
- Create: `tests/trajectory/test_collector.py`

**Step 1: Write the failing test**

```python
# tests/trajectory/test_collector.py
from mate.trajectory.collector import TrajectoryCollector
from mate.trajectory.datatypes import InteractionRecord, EpisodeTrajectory


def _make_record(agent_role: str, turn_index: int, ts: float) -> InteractionRecord:
    return InteractionRecord(
        agent_role=agent_role,
        turn_index=turn_index,
        timestamp=ts,
        messages=[{"role": "user", "content": "q"}],
        generation_params={},
        response_text=f"{agent_role}-{turn_index}",
        token_ids=[1, 2],
        logprobs=[-0.1, -0.2],
        finish_reason="stop",
        episode_id="ep1",
        metadata={},
    )


def test_build_groups_by_agent_role():
    records = [
        _make_record("verifier", 0, 1.0),
        _make_record("searcher", 0, 2.0),
        _make_record("verifier", 1, 3.0),
        _make_record("answerer", 0, 4.0),
    ]
    collector = TrajectoryCollector()
    traj = collector.build(records, episode_id="ep1")
    assert isinstance(traj, EpisodeTrajectory)
    assert set(traj.agent_trajectories.keys()) == {"verifier", "searcher", "answerer"}
    assert len(traj.agent_trajectories["verifier"]) == 2
    assert len(traj.agent_trajectories["searcher"]) == 1
    assert len(traj.agent_trajectories["answerer"]) == 1


def test_build_sorts_by_turn_index():
    records = [
        _make_record("verifier", 2, 3.0),
        _make_record("verifier", 0, 1.0),
        _make_record("verifier", 1, 2.0),
    ]
    collector = TrajectoryCollector()
    traj = collector.build(records, episode_id="ep1")
    indices = [t.turn_index for t in traj.agent_trajectories["verifier"]]
    assert indices == [0, 1, 2]


def test_build_empty_buffer():
    collector = TrajectoryCollector()
    traj = collector.build([], episode_id="ep1")
    assert traj.agent_trajectories == {}
    assert traj.episode_id == "ep1"


def test_build_preserves_data_fields():
    records = [_make_record("searcher", 0, 1.0)]
    collector = TrajectoryCollector()
    traj = collector.build(records, episode_id="ep1")
    turn = traj.agent_trajectories["searcher"][0]
    assert turn.response_text == "searcher-0"
    assert turn.token_ids == [1, 2]
    assert turn.logprobs == [-0.1, -0.2]
```

**Step 2: Run test to verify it fails**

Run: `cd /home/cxb/MATE-reboot && python -m pytest tests/trajectory/test_collector.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write the implementation**

```python
# mate/trajectory/collector.py
from __future__ import annotations

from collections import defaultdict

from .datatypes import EpisodeTrajectory, InteractionRecord, TurnData


class TrajectoryCollector:
    def build(
        self, buffer: list[InteractionRecord], episode_id: str,
    ) -> EpisodeTrajectory:
        grouped: dict[str, list[InteractionRecord]] = defaultdict(list)
        for record in buffer:
            grouped[record.agent_role].append(record)

        agent_trajectories: dict[str, list[TurnData]] = {}
        for role, records in grouped.items():
            records.sort(key=lambda r: r.turn_index)
            agent_trajectories[role] = [
                TurnData(
                    agent_role=r.agent_role,
                    turn_index=r.turn_index,
                    messages=r.messages,
                    response_text=r.response_text,
                    token_ids=r.token_ids,
                    logprobs=r.logprobs,
                    finish_reason=r.finish_reason,
                    timestamp=r.timestamp,
                    metadata=r.metadata,
                )
                for r in records
            ]

        return EpisodeTrajectory(
            episode_id=episode_id,
            agent_trajectories=agent_trajectories,
            metadata={},
        )
```

**Step 4: Run test to verify it passes**

Run: `cd /home/cxb/MATE-reboot && python -m pytest tests/trajectory/test_collector.py -v`
Expected: all 4 tests PASS

**Step 5: Commit**

```bash
git add mate/trajectory/collector.py tests/trajectory/test_collector.py
git commit -m "feat: add TrajectoryCollector for agent-grouped data assembly"
```

---

## Task 7: RewardWorker

**Files:**
- Create: `mate/trajectory/reward.py`
- Create: `tests/trajectory/test_reward.py`

**Step 1: Write the failing test**

```python
# tests/trajectory/test_reward.py
from mate.trajectory.reward import RewardWorker, FunctionRewardProvider
from mate.trajectory.datatypes import (
    EpisodeTrajectory, EpisodeResult, TurnData,
)


def _make_trajectory(roles: list[str]) -> EpisodeTrajectory:
    agent_trajectories = {}
    for role in roles:
        agent_trajectories[role] = [
            TurnData(
                agent_role=role, turn_index=0, messages=[], response_text="out",
                token_ids=None, logprobs=None, finish_reason="stop",
                timestamp=1.0, metadata={},
            )
        ]
    return EpisodeTrajectory(episode_id="ep1", agent_trajectories=agent_trajectories)


def test_function_reward_provider():
    def reward_fn(traj):
        return {
            "agent_rewards": {role: 1.0 for role in traj.agent_trajectories},
            "final_reward": 1.0,
        }
    provider = FunctionRewardProvider(reward_fn)
    traj = _make_trajectory(["verifier", "searcher"])
    result = provider.compute(traj)
    assert result["final_reward"] == 1.0
    assert result["agent_rewards"]["verifier"] == 1.0


def test_reward_worker_returns_episode_result():
    def reward_fn(traj):
        return {
            "agent_rewards": {"verifier": 0.5, "answerer": 1.0},
            "final_reward": 1.0,
        }
    worker = RewardWorker()
    traj = _make_trajectory(["verifier", "answerer"])
    result = worker.compute(traj, FunctionRewardProvider(reward_fn))
    assert isinstance(result, EpisodeResult)
    assert result.rewards["verifier"] == 0.5
    assert result.rewards["answerer"] == 1.0
    assert result.final_reward == 1.0
    assert result.trajectory is traj


def test_reward_worker_handles_per_turn_rewards():
    def reward_fn(traj):
        return {
            "agent_rewards": {"verifier": [0.3, 0.7]},
            "final_reward": 0.7,
        }
    worker = RewardWorker()
    traj = _make_trajectory(["verifier"])
    result = worker.compute(traj, FunctionRewardProvider(reward_fn))
    assert result.rewards["verifier"] == [0.3, 0.7]
```

**Step 2: Run test to verify it fails**

Run: `cd /home/cxb/MATE-reboot && python -m pytest tests/trajectory/test_reward.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write the implementation**

```python
# mate/trajectory/reward.py
from __future__ import annotations

from typing import Any, Callable, Protocol

from .datatypes import EpisodeResult, EpisodeTrajectory


class RewardProvider(Protocol):
    def compute(self, trajectory: EpisodeTrajectory) -> dict[str, Any]: ...


class FunctionRewardProvider:
    def __init__(self, func: Callable[[EpisodeTrajectory], dict[str, Any]]):
        self._func = func

    def compute(self, trajectory: EpisodeTrajectory) -> dict[str, Any]:
        return self._func(trajectory)


class RewardWorker:
    def compute(
        self,
        trajectory: EpisodeTrajectory,
        provider: RewardProvider,
    ) -> EpisodeResult:
        result = provider.compute(trajectory)
        return EpisodeResult(
            trajectory=trajectory,
            rewards=result.get("agent_rewards", {}),
            final_reward=result.get("final_reward"),
            metadata={},
        )
```

**Step 4: Run test to verify it passes**

Run: `cd /home/cxb/MATE-reboot && python -m pytest tests/trajectory/test_reward.py -v`
Expected: all 3 tests PASS

**Step 5: Commit**

```bash
git add mate/trajectory/reward.py tests/trajectory/test_reward.py
git commit -m "feat: add RewardWorker with pluggable RewardProvider protocol"
```

---

## Task 8: AgentPipe Orchestrator

**Files:**
- Create: `mate/trajectory/pipe.py`
- Create: `tests/trajectory/test_pipe.py`

**Step 1: Write the failing test**

```python
# tests/trajectory/test_pipe.py
import pytest
import yaml

from mate.trajectory.pipe import AgentPipe, AgentPipeConfig
from mate.trajectory.backend import InferenceBackend
from mate.trajectory.datatypes import (
    ModelMappingEntry, ModelRequest, ModelResponse, EpisodeResult,
)
from mate.trajectory.reward import FunctionRewardProvider


class EchoBackend(InferenceBackend):
    async def generate(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            content=f"echo:{request.agent_role}",
            token_ids=[1, 2, 3],
            logprobs=[-0.1, -0.2, -0.3],
            finish_reason="stop",
        )


def _simple_reward(traj):
    return {
        "agent_rewards": {role: 1.0 for role in traj.agent_trajectories},
        "final_reward": 1.0,
    }


async def test_agent_pipe_end_to_end(tmp_path):
    """Full pipeline: launch a trivial MAS script, collect trajectory, compute reward."""
    # Write a tiny MAS script that calls the monitor
    script = tmp_path / "tiny_mas.py"
    script.write_text('''
import json, sys, yaml
from pathlib import Path
from urllib.request import urlopen, Request

config = yaml.safe_load(Path(sys.argv[1]).read_text())
base_url = config["llm"]["base_url"]
question = sys.argv[2]

for agent in ["verifier", "answerer"]:
    payload = json.dumps({
        "model": agent,
        "messages": [{"role": "user", "content": question}],
    }).encode()
    req = Request(f"{base_url}/chat/completions", data=payload,
                  headers={"Content-Type": "application/json"})
    resp = urlopen(req)
    data = json.loads(resp.read())
    print(f"{agent}: {data['choices'][0]['message']['content']}")
''')

    config_template = {
        "llm": {"base_url": "http://placeholder/v1", "model": "default"},
        "agents": {
            "verifier": {"temperature": 0.2},
            "answerer": {"temperature": 0.4},
        },
    }

    pipe_config = AgentPipeConfig(
        mas_command=f"python {{config_path}} {{prompt}}",
        mas_command_template="python {script} {{config_path}} {{prompt}}".format(script=str(script)),
        config_template=config_template,
        model_mapping={
            "verifier": ModelMappingEntry(),
            "answerer": ModelMappingEntry(),
        },
        timeout=30.0,
    )

    pipe = AgentPipe(config=pipe_config, backend=EchoBackend())
    reward_provider = FunctionRewardProvider(_simple_reward)
    result = await pipe.run(prompt="What is 2+2?", reward_provider=reward_provider)

    assert isinstance(result, EpisodeResult)
    assert "verifier" in result.trajectory.agent_trajectories
    assert "answerer" in result.trajectory.agent_trajectories
    assert len(result.trajectory.agent_trajectories["verifier"]) == 1
    assert len(result.trajectory.agent_trajectories["answerer"]) == 1
    assert result.rewards["verifier"] == 1.0
    assert result.final_reward == 1.0
```

**Step 2: Run test to verify it fails**

Run: `cd /home/cxb/MATE-reboot && python -m pytest tests/trajectory/test_pipe.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write the implementation**

```python
# mate/trajectory/pipe.py
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from .backend import InferenceBackend
from .collector import TrajectoryCollector
from .datatypes import EpisodeResult, ModelMappingEntry
from .launcher import MASLauncher
from .monitor import ModelMonitor
from .reward import RewardProvider, RewardWorker


@dataclass
class AgentPipeConfig:
    mas_command_template: str
    config_template: dict[str, Any]
    model_mapping: dict[str, ModelMappingEntry]
    timeout: float = 300.0
    monitor_host: str = "127.0.0.1"
    monitor_port: int = 0
    mas_work_dir: str | None = None
    mas_command: str = ""  # unused legacy, kept for compat


class AgentPipe:
    def __init__(self, config: AgentPipeConfig, backend: InferenceBackend):
        self._config = config
        self._backend = backend
        self._collector = TrajectoryCollector()
        self._reward_worker = RewardWorker()

    async def run(
        self, prompt: str, reward_provider: RewardProvider,
    ) -> EpisodeResult:
        episode_id = uuid.uuid4().hex

        monitor = ModelMonitor(
            backend=self._backend,
            model_mapping=self._config.model_mapping,
            episode_id=episode_id,
        )
        launcher = MASLauncher(work_dir=self._config.mas_work_dir)

        try:
            port = await monitor.start(
                host=self._config.monitor_host,
                port=self._config.monitor_port,
            )
            monitor_url = f"http://{self._config.monitor_host}:{port}/v1"

            config_path = launcher.prepare_config(
                config_template=self._config.config_template,
                monitor_url=monitor_url,
                agent_roles=list(self._config.model_mapping.keys()),
            )

            command = self._config.mas_command_template.format(
                config_path=str(config_path),
                prompt=prompt,
            )

            process = await launcher.launch(command=command)
            exit_code = await launcher.wait(process, timeout=self._config.timeout)

            trajectory = self._collector.build(monitor.get_buffer(), episode_id)
            result = self._reward_worker.compute(trajectory, reward_provider)
            result.metadata["exit_code"] = exit_code

            return result
        finally:
            await monitor.stop()
            launcher.cleanup()
```

**Step 4: Run test to verify it passes**

Run: `cd /home/cxb/MATE-reboot && python -m pytest tests/trajectory/test_pipe.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add mate/trajectory/pipe.py tests/trajectory/test_pipe.py
git commit -m "feat: add AgentPipe orchestrator wiring all components"
```

---

## Task 9: Package Exports

**Files:**
- Modify: `mate/trajectory/__init__.py`

**Step 1: Update package exports**

```python
# mate/trajectory/__init__.py
from .backend import InferenceBackend, VLLMBackend
from .collector import TrajectoryCollector
from .datatypes import (
    EpisodeResult,
    EpisodeTrajectory,
    InteractionRecord,
    ModelMappingEntry,
    ModelRequest,
    ModelResponse,
    TurnData,
)
from .launcher import MASLauncher
from .monitor import ModelMonitor
from .pipe import AgentPipe, AgentPipeConfig
from .reward import FunctionRewardProvider, RewardProvider, RewardWorker

__all__ = [
    "AgentPipe",
    "AgentPipeConfig",
    "EpisodeResult",
    "EpisodeTrajectory",
    "FunctionRewardProvider",
    "InferenceBackend",
    "InteractionRecord",
    "MASLauncher",
    "ModelMappingEntry",
    "ModelMonitor",
    "ModelRequest",
    "ModelResponse",
    "RewardProvider",
    "RewardWorker",
    "TrajectoryCollector",
    "TurnData",
    "VLLMBackend",
]
```

**Step 2: Verify all imports work**

Run: `cd /home/cxb/MATE-reboot && python -c "from mate.trajectory import AgentPipe, ModelMonitor, VLLMBackend; print('OK')"`
Expected: `OK`

**Step 3: Run full test suite**

Run: `cd /home/cxb/MATE-reboot && python -m pytest tests/ -v`
Expected: all tests PASS

**Step 4: Commit**

```bash
git add mate/trajectory/__init__.py
git commit -m "feat: export all public API from mate.trajectory"
```

---

## Task 10: OrchRL Search MAS Integration Test

End-to-end test using OrchRL Search MAS with a mock vLLM backend. This validates the full pipeline without requiring a real LLM or retrieval service.

**Files:**
- Create: `tests/trajectory/test_orchrl_integration.py`

**Step 1: Write the integration test**

```python
# tests/trajectory/test_orchrl_integration.py
"""
Integration test: AgentPipe + OrchRL Search MAS with mock LLM backend.

Simulates a 1-turn Search MAS execution:
  Verifier → approved (yes) → Answerer → final answer

Uses a mock InferenceBackend that returns scripted responses per agent_role.
"""
import pytest

from mate.trajectory.pipe import AgentPipe, AgentPipeConfig
from mate.trajectory.backend import InferenceBackend
from mate.trajectory.datatypes import (
    ModelMappingEntry, ModelRequest, ModelResponse, EpisodeResult,
)
from mate.trajectory.reward import FunctionRewardProvider


ORCHRL_SEARCH_DIR = "/home/cxb/OrchRL/examples/mas_app/search"

SCRIPTED_RESPONSES = {
    "verifier": "Based on the information available, I can answer this question.\n<verify>yes</verify>",
    "searcher": "Let me search for more information.\n<search>test query</search>",
    "answerer": "The answer is 42.\n<answer>42</answer>",
}


class ScriptedBackend(InferenceBackend):
    async def generate(self, request: ModelRequest) -> ModelResponse:
        content = SCRIPTED_RESPONSES.get(request.agent_role, "unknown agent")
        return ModelResponse(
            content=content,
            token_ids=list(range(len(content.split()))),
            logprobs=[-0.1] * len(content.split()),
            finish_reason="stop",
        )


def _search_reward(traj):
    answer_turns = traj.agent_trajectories.get("answerer", [])
    if not answer_turns:
        return {"agent_rewards": {}, "final_reward": 0.0}
    text = answer_turns[-1].response_text
    has_answer = "<answer>" in text.lower()
    final_reward = 1.0 if has_answer else 0.0
    agent_rewards = {role: final_reward for role in traj.agent_trajectories}
    return {"agent_rewards": agent_rewards, "final_reward": final_reward}


@pytest.fixture
def orchrl_config():
    return {
        "application": {
            "type": "search",
            "max_turns": 4,
            "force_final_answer_on_max_turn": True,
        },
        "llm": {
            "base_url": "http://placeholder/v1",
            "api_key": "EMPTY",
            "timeout": 30,
            "max_retries": 1,
            "retry_backoff_sec": 0.1,
            "model": "default",
        },
        "search": {
            "provider": "disabled",
        },
        "agents": {
            "verifier": {"temperature": 0.2, "max_tokens": 512},
            "searcher": {"temperature": 0.6, "max_tokens": 512},
            "answerer": {"temperature": 0.4, "max_tokens": 512},
        },
    }


@pytest.mark.skipif(
    not __import__("pathlib").Path(ORCHRL_SEARCH_DIR).exists(),
    reason="OrchRL Search MAS not available at expected path",
)
async def test_orchrl_search_mas_e2e(orchrl_config):
    """Full e2e: AgentPipe launches OrchRL Search MAS subprocess, collects trajectory."""
    pipe_config = AgentPipeConfig(
        mas_command_template=(
            "python -m search_mas.scripts.run_search_mas"
            " --config {config_path}"
            " --question \"{prompt}\""
        ),
        config_template=orchrl_config,
        model_mapping={
            "verifier": ModelMappingEntry(),
            "searcher": ModelMappingEntry(),
            "answerer": ModelMappingEntry(),
        },
        timeout=60.0,
        mas_work_dir=ORCHRL_SEARCH_DIR,
    )

    pipe = AgentPipe(config=pipe_config, backend=ScriptedBackend())
    result = await pipe.run(
        prompt="What is the meaning of life?",
        reward_provider=FunctionRewardProvider(_search_reward),
    )

    assert isinstance(result, EpisodeResult)
    assert result.trajectory.episode_id
    assert "verifier" in result.trajectory.agent_trajectories
    assert "answerer" in result.trajectory.agent_trajectories
    verifier_turns = result.trajectory.agent_trajectories["verifier"]
    assert len(verifier_turns) >= 1
    assert "verify" in verifier_turns[0].response_text.lower()
    answerer_turns = result.trajectory.agent_trajectories["answerer"]
    assert len(answerer_turns) >= 1
    assert "answer" in answerer_turns[0].response_text.lower()
    assert result.final_reward == 1.0
    assert result.metadata.get("exit_code") == 0
```

**Step 2: Run to verify the test executes**

Run: `cd /home/cxb/MATE-reboot && python -m pytest tests/trajectory/test_orchrl_integration.py -v --timeout=120`

Expected: PASS if OrchRL is available, SKIP otherwise.

Note: this test may fail on first attempt if OrchRL's `search_client` cannot handle `provider: disabled`. If so, fix by adding a `DisabledSearchClient` fallback in the config or adjusting the config to use `provider: http` with a dummy URL. Debug and iterate.

**Step 3: Commit**

```bash
git add tests/trajectory/test_orchrl_integration.py
git commit -m "test: add OrchRL Search MAS end-to-end integration test"
```

---

## Task 11: Final Validation

**Step 1: Run full test suite**

Run: `cd /home/cxb/MATE-reboot && python -m pytest tests/ -v --timeout=120`
Expected: all tests PASS

**Step 2: Verify git log**

Run: `cd /home/cxb/MATE-reboot && git log --oneline`
Expected: clean linear history with atomic commits for each task.

**Step 3: Update docs/plans with completion status**

Append a completion note to the design doc marking V0 implementation as delivered.
