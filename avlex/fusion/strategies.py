"""Concrete fusion strategies.

* :class:`ConcatFusion` keeps each stream on its own stretch of the timeline.
* :class:`InterleaveFusion` aligns the streams and zips them frame by frame.
* :class:`GatedFusion` aligns and energy-weights them into one stream.

All three project to a shared ``d_model`` and preserve the raw per-modality
features so a downstream :class:`~avlex.bridges.token_bridge.TokenBridge` can still
read interpretable descriptors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from avlex.bridges.base import BridgeInput
from avlex.fusion.base import Fusion
from avlex.fusion.temporal import project_modality, resample_time
from avlex.types import Array, Modality
from avlex.utils.seeding import derive_seed

_ORDER = (Modality.VISUAL, Modality.AUDIO)


def _ordered(features: dict[Modality, Array]) -> list[tuple[Modality, Array]]:
    return [(m, features[m]) for m in _ORDER if m in features]


@dataclass
class ConcatFusion(Fusion):
    """Project each stream and concatenate them along time."""

    d_model: int = 128
    seed: int = 0

    def fuse(self, features: dict[Modality, Array]) -> BridgeInput:
        parts: list[np.ndarray] = []
        spans: dict[Modality, tuple[int, int]] = {}
        cursor = 0
        for modality, feat in _ordered(features):
            proj = project_modality(
                feat, self.d_model, derive_seed(self.seed, str(modality))
            )
            spans[modality] = (cursor, cursor + proj.shape[0])
            cursor += proj.shape[0]
            parts.append(proj)
        sequence = (
            np.concatenate(parts, axis=0)
            if parts
            else np.zeros((0, self.d_model), dtype=np.float64)
        )
        return BridgeInput(sequence=sequence, modalities=dict(features), spans=spans)


@dataclass
class InterleaveFusion(Fusion):
    """Align streams to a common length and interleave them frame by frame."""

    d_model: int = 128
    seed: int = 0

    def fuse(self, features: dict[Modality, Array]) -> BridgeInput:
        items = _ordered(features)
        length = max(feat.shape[0] for _, feat in items)
        projected = [
            resample_time(
                project_modality(feat, self.d_model, derive_seed(self.seed, str(m))),
                length,
            )
            for m, feat in items
        ]
        stacked = np.stack(projected, axis=1)  # (length, n_modalities, d_model)
        sequence = stacked.reshape(length * len(projected), self.d_model)
        return BridgeInput(sequence=sequence, modalities=dict(features))


@dataclass
class GatedFusion(Fusion):
    """Align streams and combine them with a per-step energy gate."""

    d_model: int = 128
    seed: int = 0

    def fuse(self, features: dict[Modality, Array]) -> BridgeInput:
        items = _ordered(features)
        length = max(feat.shape[0] for _, feat in items)
        sequence = np.zeros((length, self.d_model), dtype=np.float64)
        weight_sum = np.zeros((length, 1), dtype=np.float64)
        for modality, feat in items:
            proj = resample_time(
                project_modality(
                    feat, self.d_model, derive_seed(self.seed, str(modality))
                ),
                length,
            )
            gate = np.linalg.norm(proj, axis=1, keepdims=True)
            sequence += gate * proj
            weight_sum += gate
        sequence = sequence / np.maximum(weight_sum, 1e-8)
        return BridgeInput(sequence=sequence, modalities=dict(features))
