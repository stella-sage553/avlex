"""Encoder abstraction.

An :class:`Encoder` turns one modality's raw input into a temporal feature
sequence ``(T, D)``. The bundled encoders are pure NumPy; a torch-backed encoder
only needs to subclass this and return an array.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from avlex.types import Array, Modality


class Encoder(ABC):
    """Base class for audio and visual encoders."""

    #: Which modality this encoder consumes.
    modality: Modality

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Dimensionality ``D`` of each emitted feature vector."""

    @abstractmethod
    def encode(self, raw: Array) -> Array:
        """Encode ``raw`` into a ``(T, D)`` float array."""

    def __call__(self, raw: Array) -> Array:
        return self.encode(raw)

    @property
    def name(self) -> str:
        return type(self).__name__

    def __repr__(self) -> str:
        return f"{self.name}(modality={self.modality}, dim={self.output_dim})"
