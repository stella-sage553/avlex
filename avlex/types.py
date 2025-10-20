"""Shared type aliases and the modality enum.

Keeping these in one place avoids import cycles between encoders, bridges, and
fusion, all of which speak in terms of temporal feature sequences.
"""

from __future__ import annotations

from enum import Enum

import numpy as np

#: A temporal feature sequence with shape ``(T, D)`` — ``T`` frames/segments of a
#: ``D``-dimensional descriptor. This is the lingua franca between components.
FeatureSequence = np.ndarray

#: Generic numeric array alias used where the rank is documented in the docstring.
Array = np.ndarray


class Modality(str, Enum):
    """The two streams avlex bridges into a language model."""

    AUDIO = "audio"
    VISUAL = "visual"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value
