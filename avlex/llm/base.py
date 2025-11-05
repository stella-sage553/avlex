"""LLM client abstraction.

avlex talks to language models through a tiny interface so the offline
:class:`~avlex.llm.template_llm.TemplateLLM` and a real hosted model are
interchangeable from the pipeline's point of view.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Message:
    """One chat turn."""

    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class GenerationConfig:
    """Decoding knobs passed to a client."""

    max_tokens: int = 128
    temperature: float = 0.0
    seed: int = 0


@dataclass
class LLMResponse:
    """A model reply plus any backend metadata."""

    text: str
    meta: dict = field(default_factory=dict)


class LLMClient(ABC):
    """Minimal chat-style language model interface."""

    @abstractmethod
    def generate(
        self,
        messages: list[Message],
        config: Optional[GenerationConfig] = None,
    ) -> LLMResponse:
        """Generate a reply to ``messages``."""

    def complete(self, prompt: str, **kwargs: object) -> str:
        """Convenience wrapper: single user turn in, text out."""
        config = GenerationConfig(**kwargs) if kwargs else None  # type: ignore[arg-type]
        return self.generate([Message("user", prompt)], config).text

    @property
    def name(self) -> str:
        return type(self).__name__
