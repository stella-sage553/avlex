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


def synthetic_clip(
    seed: int = 0,
    n_frames: int = 24,
    size: int = 16,
    sample_rate: int = 16000,
    duration: float = 1.5,
) -> ClipFeatures:
    """Build a deterministic toy clip: a drifting, brightening blob over rising audio.

    Handy for demos, docs, and tests — it exercises the whole pipeline with no
    media files. Activity grows over time so the timeline reads as "building".
    """
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float64)
    sigma = size / 6.0
    frames = np.zeros((n_frames, size, size), dtype=np.float64)
    for t in range(n_frames):
        frac = t / max(n_frames - 1, 1)
        cx = size * (0.2 + 0.6 * frac)
        cy = size * (0.5 + 0.25 * np.sin(2.0 * np.pi * frac * 2.0))
        amp = 0.3 + 0.7 * frac
        blob = amp * np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * sigma**2))
        frames[t] = np.clip(blob + 0.02 * rng.standard_normal((size, size)), 0.0, 1.0)

    n_samples = int(sample_rate * duration)
    time = np.arange(n_samples) / sample_rate
    envelope = np.linspace(0.1, 1.0, n_samples)
    tone = np.sin(2.0 * np.pi * 330.0 * time) + 0.5 * np.sin(2.0 * np.pi * 660.0 * time)
    audio = envelope * tone + 0.05 * rng.standard_normal(n_samples)

    return ClipFeatures(
        visual=frames,
        audio=audio,
        fps=n_frames / duration,
        sample_rate=sample_rate,
        meta={"synthetic": True, "seed": seed},
    )
