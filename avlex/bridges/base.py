"""Bridge abstraction: the heart of avlex.

A *bridge* takes encoded (and fused) audio-visual features and turns them into
something a language model can consume. There are two flavours:

* **soft bridges** emit ``tokens`` — a ``(K, D)`` array of continuous vectors
  meant to be fed to a multimodal LLM as soft-prompt embeddings;
* **token bridges** emit ``words`` — short textual descriptors that drop straight
  into a prompt for any text-only LLM.

A bridge may produce either or both.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from avlex.types import Array, Modality


@dataclass
class BridgeInput:
    """Features handed to a bridge.

    ``sequence`` is the fused, common-dimension timeline used by soft bridges;
    ``modalities`` keeps the per-stream features that interpretable bridges read.
    """

    sequence: Array
    modalities: dict[Modality, Array] = field(default_factory=dict)
    spans: dict[Modality, tuple[int, int]] = field(default_factory=dict)


@dataclass
class BridgeOutput:
    """What a bridge produces for the prompt assembler."""

    tokens: Array | None = None
    words: list[str] | None = None
    meta: dict = field(default_factory=dict)

    @property
    def num_tokens(self) -> int:
        return 0 if self.tokens is None else int(self.tokens.shape[0])

    @property
    def has_words(self) -> bool:
        return bool(self.words)


class Bridge(ABC):
    """Map audio-visual features into LLM-ready tokens and/or words."""

    @abstractmethod
    def bridge(self, features: BridgeInput) -> BridgeOutput:
        """Reduce ``features`` to a :class:`BridgeOutput`."""

    def __call__(self, features: BridgeInput) -> BridgeOutput:
        return self.bridge(features)

    @property
    def name(self) -> str:
        return type(self).__name__
