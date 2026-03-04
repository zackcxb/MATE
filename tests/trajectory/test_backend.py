import httpx
import pytest

from mate.trajectory.backend import InferenceBackend, VLLMBackend
from mate.trajectory.datatypes import ModelRequest, ModelResponse


def test_inference_backend_is_abstract():
    with pytest.raises(TypeError):
        InferenceBackend()


async def test_vllm_backend_generate_forwards_and_parses(monkeypatch):
    captured: dict[str, object] = {}

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            captured["client_init_kwargs"] = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json):
            captured["url"] = url
            captured["json"] = json
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "test response"},
                            "finish_reason": "stop",
                            "logprobs": {
                                "content": [
                                    {"token": "test", "logprob": -0.5},
                                    {"token": " response", "logprob": -0.3},
                                ]
                            },
                        }
                    ]
                },
                request=httpx.Request("POST", url),
            )

    monkeypatch.setattr("mate.trajectory.backend.httpx.AsyncClient", FakeAsyncClient)

    backend = VLLMBackend(backend_url="http://fake-vllm", actual_model="Qwen3-4B")
    req = ModelRequest(
        request_id="r1",
        agent_role="verifier",
        messages=[{"role": "user", "content": "hello"}],
        generation_params={"temperature": 0.7, "max_tokens": 128},
    )

    resp = await backend.generate(req)

    assert captured["url"] == "http://fake-vllm/v1/chat/completions"
    assert isinstance(resp, ModelResponse)
    assert resp.content == "test response"
    assert resp.finish_reason == "stop"
    assert resp.logprobs == [-0.5, -0.3]
    assert captured["json"] == {
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0.7,
        "max_tokens": 128,
        "logprobs": True,
        "model": "Qwen3-4B",
    }


async def test_vllm_backend_forces_logprobs_true(monkeypatch):
    captured: dict[str, object] = {}

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json):
            captured["json"] = json
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                            "logprobs": None,
                        }
                    ]
                },
                request=httpx.Request("POST", url),
            )

    monkeypatch.setattr("mate.trajectory.backend.httpx.AsyncClient", FakeAsyncClient)

    backend = VLLMBackend(backend_url="http://fake-vllm")
    req = ModelRequest(
        request_id="r1",
        agent_role="searcher",
        messages=[{"role": "user", "content": "x"}],
        generation_params={"logprobs": False},
    )
    await backend.generate(req)

    assert captured["json"]["logprobs"] is True
    assert captured["json"]["model"] == "searcher"


async def test_vllm_backend_actual_model_overrides_request_model(monkeypatch):
    captured: dict[str, object] = {}

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json):
            captured["json"] = json
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                            "logprobs": None,
                        }
                    ]
                },
                request=httpx.Request("POST", url),
            )

    monkeypatch.setattr("mate.trajectory.backend.httpx.AsyncClient", FakeAsyncClient)

    backend = VLLMBackend(backend_url="http://fake-vllm", actual_model="real-model-name")
    req = ModelRequest(
        request_id="r1",
        agent_role="verifier",
        messages=[{"role": "user", "content": "x"}],
        generation_params={"model": "placeholder-model"},
    )
    await backend.generate(req)

    assert captured["json"]["model"] == "real-model-name"
