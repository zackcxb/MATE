import httpx
import pytest

from mate.trajectory.backend import InferenceBackend
from mate.trajectory.datatypes import ModelMappingEntry, ModelRequest, ModelResponse
from mate.trajectory.monitor import ModelMonitor


class RecordingBackend(InferenceBackend):
    def __init__(self, *, raise_error: bool = False):
        self.raise_error = raise_error
        self.requests: list[ModelRequest] = []

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if self.raise_error:
            raise RuntimeError("backend boom")
        return ModelResponse(
            content=f"echo:{request.agent_role}",
            token_ids=[11, 22, 33],
            logprobs=[-0.1, -0.2, -0.3],
            finish_reason="stop",
        )


@pytest.fixture
async def monitor_server():
    mapping = {
        "verifier": ModelMappingEntry(actual_model="m1"),
        "searcher": ModelMappingEntry(actual_model="m2"),
    }
    backend = RecordingBackend()
    monitor = ModelMonitor(backend=backend, model_mapping=mapping)
    port = await monitor.start()
    yield monitor, backend, port
    await monitor.stop()


async def test_routes_by_model_field(monitor_server):
    _, backend, port = monitor_server
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            json={
                "model": "verifier",
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 0.6,
            },
        )

    assert response.status_code == 200
    assert len(backend.requests) == 1
    assert backend.requests[0].agent_role == "verifier"
    assert response.json()["choices"][0]["message"]["content"] == "echo:verifier"


async def test_collects_interaction_record_to_buffer(monitor_server):
    monitor, _, port = monitor_server
    async with httpx.AsyncClient() as client:
        await client.post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            json={
                "model": "searcher",
                "messages": [{"role": "user", "content": "find x"}],
                "top_p": 0.9,
            },
        )

    buf = monitor.get_buffer()
    assert len(buf) == 1
    record = buf[0]
    assert record.response_text == "echo:searcher"
    assert record.token_ids == [11, 22, 33]
    assert record.logprobs == [-0.1, -0.2, -0.3]
    assert record.turn_index == 0


async def test_turn_index_increments_for_same_agent(monitor_server):
    monitor, _, port = monitor_server
    async with httpx.AsyncClient() as client:
        for _ in range(3):
            await client.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                json={
                    "model": "verifier",
                    "messages": [{"role": "user", "content": "q"}],
                },
            )

    turns = [r.turn_index for r in monitor.get_buffer()]
    assert turns == [0, 1, 2]


async def test_unknown_agent_returns_400(monitor_server):
    _, _, port = monitor_server
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            json={"model": "ghost", "messages": []},
        )

    assert response.status_code == 400


async def test_clear_buffer_works(monitor_server):
    monitor, _, port = monitor_server
    async with httpx.AsyncClient() as client:
        await client.post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            json={"model": "verifier", "messages": [{"role": "user", "content": "q"}]},
        )

    assert len(monitor.get_buffer()) == 1
    monitor.clear_buffer()
    assert monitor.get_buffer() == []


async def test_invalid_json_returns_400():
    mapping = {"verifier": ModelMappingEntry(actual_model="m1")}
    monitor = ModelMonitor(backend=RecordingBackend(), model_mapping=mapping)
    port = await monitor.start()
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                content="{bad-json",
                headers={"content-type": "application/json"},
            )
        assert response.status_code == 400
    finally:
        await monitor.stop()


async def test_backend_exception_returns_502():
    mapping = {"verifier": ModelMappingEntry(actual_model="m1")}
    monitor = ModelMonitor(backend=RecordingBackend(raise_error=True), model_mapping=mapping)
    port = await monitor.start()
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                json={"model": "verifier", "messages": [{"role": "user", "content": "q"}]},
            )
        assert response.status_code == 502
    finally:
        await monitor.stop()
