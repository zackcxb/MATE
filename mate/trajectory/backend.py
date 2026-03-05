from __future__ import annotations

from abc import ABC, abstractmethod
import math
from typing import Any

import httpx

from .datatypes import ModelRequest, ModelResponse

BACKEND_URL_OVERRIDE_KEY = "_backend_url"


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
        generation_params = dict(request.generation_params)
        backend_url_override = generation_params.pop(BACKEND_URL_OVERRIDE_KEY, None)
        target_backend_url = self.backend_url
        if isinstance(backend_url_override, str) and backend_url_override:
            target_backend_url = backend_url_override.rstrip("/")

        payload: dict[str, Any] = {
            "messages": request.messages,
            **generation_params,
        }
        payload["logprobs"] = True
        payload["return_token_ids"] = True
        if self.actual_model:
            payload["model"] = self.actual_model
        elif "model" not in payload:
            payload["model"] = request.agent_role

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{target_backend_url}/v1/chat/completions",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        choices = data.get("choices") if isinstance(data, dict) else None
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ValueError("malformed response: missing or invalid choices")

        choice = choices[0]
        message = choice.get("message") or {}
        content = message.get("content") or ""
        finish_reason = choice.get("finish_reason") or "stop"
        token_ids = choice.get("token_ids")

        logprobs: list[float] | None = None
        logprobs_data = choice.get("logprobs")
        if isinstance(logprobs_data, dict) and isinstance(logprobs_data.get("content"), list):
            values: list[float] = []
            for token_info in logprobs_data["content"]:
                if not isinstance(token_info, dict):
                    continue
                value = token_info.get("logprob")
                if isinstance(value, bool):
                    continue
                if isinstance(value, (int, float)) and math.isfinite(value):
                    values.append(float(value))
            logprobs = values

        return ModelResponse(
            content=content,
            token_ids=token_ids if isinstance(token_ids, list) else None,
            logprobs=logprobs,
            finish_reason=finish_reason,
        )
