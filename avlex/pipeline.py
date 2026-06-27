"""The high-level pipeline: clip in, language out.

A :class:`Pipeline` runs the five stages avlex is built around::

    encoders -> fusion -> bridge -> prompt -> llm

Each stage is an injectable component, so the same orchestration works whether
you bridge with a token bridge and a text LLM (the offline default) or with a
soft bridge and a multimodal model.
"""

from __future__ import annotations

from dataclasses import dataclass

from avlex.bridges.base import Bridge
from avlex.clip import ClipFeatures
from avlex.encoders.base import Encoder
from avlex.fusion.base import Fusion
from avlex.llm.base import LLMClient, Message
from avlex.prompts.library import get_template
from avlex.prompts.render import render_words
from avlex.result import PipelineResult
from avlex.tasks import Task, get_task_spec
from avlex.types import Array, Modality


@dataclass
class Pipeline:
    """Compose encoders, fusion, a bridge, and an LLM into one callable."""

    visual_encoder: Encoder | None
    audio_encoder: Encoder | None
    fusion: Fusion
    bridge: Bridge
    llm: LLMClient

    def _encode(self, clip: ClipFeatures) -> dict[Modality, Array]:
        feats: dict[Modality, Array] = {}
        if self.visual_encoder is not None and clip.has_visual:
            feats[Modality.VISUAL] = self.visual_encoder.encode(clip.visual)
        if self.audio_encoder is not None and clip.has_audio:
            feats[Modality.AUDIO] = self.audio_encoder.encode(clip.audio)
        if not feats:
            raise ValueError("no encodable modalities for this clip")
        return feats

    def _assemble(
        self,
        task: Task,
        words: list[str] | None,
        question: str | None,
    ) -> tuple[str, list[Message]]:
        spec = get_task_spec(task)
        template = get_template(task)
        values: dict[str, object] = {
            "perception": render_words(words or []),
            "instruction": spec.instruction,
        }
        if spec.needs_question:
            if not question:
                raise ValueError(f"task {task} requires a question")
            values["question"] = question
        user = template.render(**values)
        messages: list[Message] = []
        if template.system:
            messages.append(Message("system", template.system))
        messages.append(Message("user", user))
        return user, messages

    def run(
        self,
        clip: ClipFeatures,
        task: Task | str = Task.CAPTION,
        question: str | None = None,
    ) -> PipelineResult:
        """Run the full pipeline for ``task`` and return a :class:`PipelineResult`."""
        task = Task(task)
        features = self._encode(clip)
        fused = self.fusion.fuse(features)
        output = self.bridge.bridge(fused)
        prompt, messages = self._assemble(task, output.words, question)
        response = self.llm.generate(messages)
        return PipelineResult(
            text=response.text,
            task=task,
            prompt=prompt,
            words=output.words,
            tokens=output.tokens,
            meta={"bridge": self.bridge.name, "llm": self.llm.name, **response.meta},
        )

    def caption(self, clip: ClipFeatures) -> PipelineResult:
        """Caption ``clip`` in one sentence."""
        return self.run(clip, Task.CAPTION)
