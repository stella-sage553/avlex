"""Task definitions for what avlex should ask the LLM to do."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Task(str, Enum):
    """Supported downstream tasks."""

    CAPTION = "caption"
    VIDEO_QA = "video_qa"
    SUMMARIZE = "summarize"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value


@dataclass(frozen=True)
class TaskSpec:
    """How a task is phrased to the language model."""

    task: Task
    instruction: str
    needs_question: bool = False


_DEFAULT_SPECS: dict[Task, TaskSpec] = {
    Task.CAPTION: TaskSpec(
        Task.CAPTION,
        "Write one natural sentence describing what happens in the clip.",
    ),
    Task.VIDEO_QA: TaskSpec(
        Task.VIDEO_QA,
        "Answer the question using only what can be seen and heard in the clip.",
        needs_question=True,
    ),
    Task.SUMMARIZE: TaskSpec(
        Task.SUMMARIZE,
        "Summarize the clip in two or three short sentences.",
    ),
}


def get_task_spec(task: Task | str) -> TaskSpec:
    """Look up the :class:`TaskSpec` for ``task`` (enum or its string value)."""
    key = Task(task)
    return _DEFAULT_SPECS[key]
