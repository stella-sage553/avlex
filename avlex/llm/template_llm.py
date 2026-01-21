"""A deterministic, offline language model stand-in.

:class:`TemplateLLM` reads the perception block that the prompt assembler built
and turns it into a caption, summary, or answer with simple, fully deterministic
rules. It exists so the whole pipeline runs (and is testable) with no model
weights and no network. Swap in a real :class:`~avlex.llm.base.LLMClient` for
quality; the pipeline does not change.
"""

from __future__ import annotations

from avlex.llm.base import GenerationConfig, LLMClient, LLMResponse, Message
from avlex.prompts.render import parse_perception


def _line_after(text: str, prefix: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            return stripped[len(prefix) :].strip()
    return ""


def _caption(percept: dict[str, str]) -> str:
    parts = [percept[key] for key in ("visual", "audio") if key in percept]
    if not parts:
        return "The clip has no clearly described audio-visual content."
    sentence = "The clip shows " + " and ".join(parts)
    if "timeline" in percept:
        sentence += f", and {percept['timeline']}"
    return sentence + "."


def _summary(percept: dict[str, str]) -> str:
    head = _caption({k: percept[k] for k in ("visual", "audio") if k in percept})
    if "timeline" in percept:
        head += f" Overall, {percept['timeline']}."
    return head


def _answer(question: str, percept: dict[str, str]) -> str:
    ql = question.lower()
    visual_cues = ("move", "movement", "motion", "happening", "action", "see", "visual")
    audio_cues = ("sound", "hear", "audio", "loud", "music", "speech", "noise", "voice")
    if any(cue in ql for cue in visual_cues) and "visual" in percept:
        return f"Visually, there is {percept['visual']}."
    if any(cue in ql for cue in audio_cues) and "audio" in percept:
        return f"In the audio, there is {percept['audio']}."
    return _caption(percept)


class TemplateLLM(LLMClient):
    """Rule-based, deterministic LLM stand-in for offline pipelines and tests."""

    def generate(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
    ) -> LLMResponse:
        user = next(
            (m.content for m in reversed(messages) if m.role == "user"),
            "",
        )
        percept = parse_perception(user)
        question = _line_after(user, "Question:")
        instruction = _line_after(user, "Task:").lower()
        if question:
            text = _answer(question, percept)
        elif "summar" in instruction or "two or three" in instruction:
            text = _summary(percept)
        else:
            text = _caption(percept)
        return LLMResponse(
            text=text, meta={"backend": "template", "perception": percept}
        )
