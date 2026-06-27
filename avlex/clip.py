"""The :class:`ClipFeatures` container that flows into a pipeline.

A clip carries whatever the encoders expect as *raw* input — video frames and an
audio waveform for the bundled NumPy encoders, or pre-extracted features if you
bring your own encoder. avlex deliberately does not decode media itself; that
keeps the core free of ffmpeg/opencv and makes results reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from avlex.types import Array


@dataclass
class ClipFeatures:
    """Raw audio and/or visual data for a single clip.

    Parameters
    ----------
    visual:
        Frames as ``(T, H, W)`` grayscale or ``(T, H, W, C)``, or a precomputed
        feature sequence ``(T, D)``.
    audio:
        A waveform ``(N,)`` or a precomputed feature sequence ``(T, D)``.
    fps:
        Frames per second of ``visual`` (used for temporal alignment).
    sample_rate:
        Sample rate of ``audio`` in Hz.
    meta:
        Free-form metadata (clip id, source path, ...).
    """

    visual: Array | None = None
    audio: Array | None = None
    fps: float = 1.0
    sample_rate: int = 16000
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.visual is None and self.audio is None:
            raise ValueError("ClipFeatures needs at least one of visual/audio")
        if self.visual is not None:
            self.visual = np.asarray(self.visual)
        if self.audio is not None:
            self.audio = np.asarray(self.audio)
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")

    @property
    def has_visual(self) -> bool:
        return self.visual is not None

    @property
    def has_audio(self) -> bool:
        return self.audio is not None

    def __repr__(self) -> str:
        v = None if self.visual is None else tuple(self.visual.shape)
        a = None if self.audio is None else tuple(self.audio.shape)
        return f"ClipFeatures(visual={v}, audio={a}, fps={self.fps})"
