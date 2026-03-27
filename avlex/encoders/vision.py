"""Lightweight NumPy visual encoders.

These are intentionally simple, training-free descriptors. They are good enough
to drive the offline pipeline and to stand in for a real ViT-style encoder during
tests; swap in a torch-backed :class:`~avlex.encoders.base.Encoder` for quality.
"""

from __future__ import annotations

import numpy as np

from avlex.encoders.base import Encoder
from avlex.types import Array, Modality


def _to_gray(raw: Array) -> np.ndarray:
    """Coerce frames to ``(T, H, W)`` float in ``[0, 1]``."""
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim == 4:  # (T, H, W, C)
        arr = arr.mean(axis=-1)
    elif arr.ndim != 3:
        raise ValueError(f"expected (T,H,W[,C]) frames, got shape {arr.shape}")
    peak = float(arr.max()) if arr.size else 0.0
    if peak > 1.0:
        arr = arr / peak
    return arr


class MotionHistogramEncoder(Encoder):
    """Per-frame histogram of inter-frame change magnitude.

    Captures *how much* is moving in each frame: a near-static frame piles mass in
    the low bins, a busy frame spreads it out. The first frame has no predecessor
    so its histogram is all zeros.
    """

    modality = Modality.VISUAL

    def __init__(self, n_bins: int = 32) -> None:
        if n_bins < 2:
            raise ValueError("n_bins must be >= 2")
        self.n_bins = n_bins

    @property
    def output_dim(self) -> int:
        return self.n_bins

    def encode(self, raw: Array) -> np.ndarray:
        frames = _to_gray(raw)
        n = frames.shape[0]
        edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        feats = np.zeros((n, self.n_bins), dtype=np.float64)
        for t in range(1, n):
            diff = np.abs(frames[t] - frames[t - 1]).ravel()
            hist, _ = np.histogram(diff, bins=edges)
            total = hist.sum()
            if total > 0:
                feats[t] = hist / total
        return feats


class ColorStatsEncoder(Encoder):
    """Per-frame intensity histogram — a brightness/contrast descriptor.

    Complements :class:`MotionHistogramEncoder`: motion says *how much* changes,
    intensity says *how bright* the scene is and how spread its tones are.
    """

    modality = Modality.VISUAL

    def __init__(self, n_bins: int = 16) -> None:
        if n_bins < 2:
            raise ValueError("n_bins must be >= 2")
        self.n_bins = n_bins

    @property
    def output_dim(self) -> int:
        return self.n_bins

    def encode(self, raw: Array) -> np.ndarray:
        frames = _to_gray(raw)
        n = frames.shape[0]
        edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        feats = np.zeros((n, self.n_bins), dtype=np.float64)
        for t in range(n):
            hist, _ = np.histogram(frames[t].ravel(), bins=edges)
            total = hist.sum()
            if total > 0:
                feats[t] = hist / total
        return feats
