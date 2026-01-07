"""Fusion abstraction: combine per-modality features into a bridge input.

A :class:`Fusion` projects each stream to a common width and merges them into the
single ``sequence`` that soft bridges consume, while keeping the original
per-modality features around for the interpretable token bridge.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from avlex.bridges.base import BridgeInput
from avlex.types import Array, Modality


class Fusion(ABC):
    """Merge ``{modality: (T, D)}`` features into a :class:`BridgeInput`."""

    @abstractmethod
    def fuse(self, features: dict[Modality, Array]) -> BridgeInput:
        """Combine ``features`` into one fused timeline."""

    def __call__(self, features: dict[Modality, Array]) -> BridgeInput:
        return self.fuse(features)

    @property
    def name(self) -> str:
        return type(self).__name__
