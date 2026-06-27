"""Result objects returned by the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

from avlex.tasks import Task
from avlex.types import Array


@dataclass
class PipelineResult:
    """The model's text output plus the intermediate artifacts that produced it.

    Keeping the ``prompt``, ``words``, and soft ``tokens`` around makes the
    pipeline debuggable — you can see exactly what the LLM was shown.
    """

    text: str
    task: Task
    prompt: str
    words: list[str] | None = None
    tokens: Array | None = None
    meta: dict = field(default_factory=dict)

    @property
    def answer(self) -> str:
        """Alias for :attr:`text`, reads better for question answering."""
        return self.text

    def __str__(self) -> str:
        return self.text
