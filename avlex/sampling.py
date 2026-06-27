"""Frame and segment sampling strategies.

Long clips have more frames than a bridge needs. These helpers pick a subset,
either evenly across time or by favouring frames where a lot is moving.
"""

from __future__ import annotations

import numpy as np

from avlex.types import Array


def uniform_indices(n: int, k: int) -> np.ndarray:
    """Return up to ``k`` evenly spaced indices into ``range(n)``."""
    if n <= 0:
        raise ValueError("n must be positive")
    if k <= 0:
        raise ValueError("k must be positive")
    if k >= n:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, k).round().astype(int))


def motion_keyframe_indices(frames: Array, k: int) -> np.ndarray:
    """Pick the ``k`` frames with the most temporal change, in time order."""
    arr = np.asarray(frames, dtype=np.float64)
    if arr.ndim < 2:
        raise ValueError("frames must have a leading time axis")
    n = arr.shape[0]
    if k <= 0:
        raise ValueError("k must be positive")
    if k >= n:
        return np.arange(n)
    flat = arr.reshape(n, -1)
    motion = np.zeros(n)
    motion[1:] = np.abs(flat[1:] - flat[:-1]).mean(axis=1)
    chosen = np.argsort(motion)[-k:]
    return np.sort(chosen)


def sample_frames(frames: Array, k: int, strategy: str = "uniform") -> np.ndarray:
    """Return ``k`` sampled frames using ``strategy`` (``uniform``|``motion``)."""
    arr = np.asarray(frames)
    if strategy == "uniform":
        idx = uniform_indices(arr.shape[0], k)
    elif strategy == "motion":
        idx = motion_keyframe_indices(arr, k)
    else:
        raise ValueError(f"unknown sampling strategy: {strategy!r}")
    return arr[idx]
