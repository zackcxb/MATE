from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import httpx

from .datatypes import ModelRequest, ModelResponse


class InferenceBackend(ABC):
    @abstractmethod
    async def generate(self, request: ModelRequest) -> ModelResponse:
        """Generate a response from an inference backend."""


class VLLMBackend(InferenceBackend):
    """Inference backend that forwards OpenAI-style requests to vLLM."""

    def __init__(
        self,
        backend_url: str,
        actual_model: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.backend_url = backend_url.rstrip("/")
        self.actual_model = actual_model
        self.timeout = timeout

    async def generate(self, request: ModelRequest) -> ModelResponse:
        payload: dict[str, Any] = {
            "messages": request.messages,
            **request.generation_params,
        }
        payload["logprobs"] = True
        if self.actual_model:
            payload["model"] = self.actual_model
        elif "model" not in payload:
            payload["model"] = request.agent_role

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.backend_url}/v1/chat/completions",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        choice = data["choices"][0]
        message = choice.get("message") or {}
        content = message.get("content") or ""
        finish_reason = choice.get("finish_reason") or "stop"

        logprobs: list[float] | None = None
        logprobs_data = choice.get("logprobs")
        if isinstance(logprobs_data, dict) and isinstance(logprobs_data.get("content"), list):
            values: list[float] = []
            for token_info in logprobs_data["content"]:
                if isinstance(token_info, dict) and token_info.get("logprob") is not None:
                    values.append(token_info["logprob"])
            logprobs = values

        return ModelResponse(
            content=content,
            token_ids=None,
            logprobs=logprobs,
            finish_reason=finish_reason,
        )
