"""Token bridge: audio-visual features -> perceptual words.

Unlike the soft bridges, this one emits *text*. It reads interpretable
descriptors out of the per-modality features (how much is moving, how loud and
bright the audio is, whether the pace rises or falls) and turns them into short
phrases that slot straight into a prompt for an ordinary text LLM. That makes the
whole pipeline runnable with no model weights at all.

The thresholds below are calibrated for avlex's bundled NumPy encoders; override
them when bridging features from a different encoder.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from avlex.bridges.base import Bridge, BridgeInput, BridgeOutput
from avlex.types import Modality
from avlex.utils.arrays import as_float

_INF = float("inf")

#: (upper-threshold, phrase) buckets, scanned low to high.
MOTION_WORDS = [
    (0.08, "barely any movement"),
    (0.20, "a little movement"),
    (0.40, "steady movement"),
    (_INF, "lots of movement"),
]
LOUDNESS_WORDS = [
    (0.6, "quiet"),
    (1.6, "moderate sound"),
    (_INF, "loud sound"),
]
BRIGHTNESS_WORDS = [
    (0.33, "low, rumbling tones"),
    (0.6, "mid-range tones"),
    (_INF, "bright, high tones"),
]


def _bucket(value: float, table: list[tuple[float, str]]) -> str:
    for upper, word in table:
        if value <= upper:
            return word
    return table[-1][1]


def _centroid_salience(feat: np.ndarray) -> np.ndarray:
    """Per-frame normalized centroid over the feature index, in ``[0, 1]``."""
    f = np.clip(feat, 0.0, None)
    dim = f.shape[1]
    idx = np.arange(dim)
    rowsum = f.sum(axis=1) + 1e-8
    return (f * idx).sum(axis=1) / rowsum / max(dim - 1, 1)


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def _trend(activity: np.ndarray) -> str:
    if activity.shape[0] < 2:
        return "the clip is too short to tell a story"
    x = np.linspace(0.0, 1.0, activity.shape[0])
    slope = float(np.polyfit(x, _minmax(activity), 1)[0])
    if slope > 0.15:
        return "the action builds toward the end"
    if slope < -0.15:
        return "it settles down after an active start"
    return "the pace stays fairly even"


@dataclass
class TokenBridge(Bridge):
    """Describe a clip with short perceptual phrases for text-only LLMs."""

    include_timeline: bool = True

    def bridge(self, features: BridgeInput) -> BridgeOutput:
        words: list[str] = []
        meta: dict = {}
        activity: dict[Modality, np.ndarray] = {}

        for modality, raw in features.modalities.items():
            feat = as_float(raw)
            if feat.ndim != 2 or feat.shape[0] == 0:
                continue
            activity[modality] = np.linalg.norm(feat, axis=1)
            if modality == Modality.VISUAL:
                salience = float(_centroid_salience(feat).mean())
                words.append(f"visual: {_bucket(salience, MOTION_WORDS)}")
                meta["visual_salience"] = salience
            else:
                loudness = float(feat.mean())
                brightness = float(_centroid_salience(feat).mean())
                words.append(
                    f"audio: {_bucket(loudness, LOUDNESS_WORDS)} with "
                    f"{_bucket(brightness, BRIGHTNESS_WORDS)}"
                )
                meta["audio_loudness"] = loudness
                meta["audio_brightness"] = brightness

        if self.include_timeline and activity:
            ref = activity.get(Modality.VISUAL)
            if ref is None:
                ref = next(iter(activity.values()))
            words.append(f"timeline: {_trend(ref)}")

        return BridgeOutput(words=words, meta=meta)
