"""Built-in prompt templates for the supported tasks."""

from __future__ import annotations

from avlex.prompts.template import PromptTemplate
from avlex.tasks import Task

_SYSTEM = "You are a careful assistant that describes short video clips using only the audio-visual cues provided."

CAPTION = PromptTemplate(
    system=_SYSTEM,
    template=("Perception:\n{perception}\n\nTask: {instruction}\nAnswer:"),
)

VIDEO_QA = PromptTemplate(
    system=_SYSTEM,
    template=(
        "Perception:\n{perception}\n\n"
        "Question: {question}\n"
        "Task: {instruction}\n"
        "Answer:"
    ),
)

SUMMARIZE = PromptTemplate(
    system=_SYSTEM,
    template=("Perception:\n{perception}\n\nTask: {instruction}\nAnswer:"),
)

_TEMPLATES: dict[Task, PromptTemplate] = {
    Task.CAPTION: CAPTION,
    Task.VIDEO_QA: VIDEO_QA,
    Task.SUMMARIZE: SUMMARIZE,
}


def get_template(task: Task | str) -> PromptTemplate:
    """Return the built-in template for ``task``."""
    return _TEMPLATES[Task(task)]
