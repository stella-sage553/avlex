"""Optional OpenAI chat-completions backend.

This is the one place avlex talks to a hosted model. ``openai`` is an optional
dependency (``pip install avlex[openai]``) and is imported lazily so the core
never needs it. Not exercised in CI — it requires network and credentials.
"""

from __future__ import annotations

import os

from avlex.llm.base import GenerationConfig, LLMClient, LLMResponse, Message


class OpenAIClient(LLMClient):  # pragma: no cover - requires network + credentials
    """Adapter over the OpenAI Chat Completions API."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None) -> None:
        self.model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")

    def _client(self):  # noqa: ANN202 - third-party type
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "OpenAIClient needs the 'openai' extra: pip install avlex[openai]"
            ) from exc
        return OpenAI(api_key=self._api_key)

    def generate(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
    ) -> LLMResponse:
        config = config or GenerationConfig()
        payload = [{"role": m.role, "content": m.content} for m in messages]
        response = self._client().chat.completions.create(
            model=self.model,
            messages=payload,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )
        text = response.choices[0].message.content or ""
        return LLMResponse(text=text, meta={"backend": "openai", "model": self.model})
