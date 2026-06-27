"""Small array helpers used across encoders, bridges, and fusion.

Everything here operates on plain :class:`numpy.ndarray` values and is written to
be deterministic and dependency-free so the core stays importable without torch.
"""

from __future__ import annotations

import numpy as np

from avlex.types import Array


def as_float(x: Array) -> np.ndarray:
    """Return ``x`` as a contiguous float64 array (cheap no-op if already so)."""
    return np.ascontiguousarray(x, dtype=np.float64)


def l2_normalize(x: Array, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    """L2-normalize ``x`` along ``axis``; rows with ~zero norm are left at zero."""
    arr = as_float(x)
    norm = np.sqrt(np.sum(arr * arr, axis=axis, keepdims=True))
    return arr / np.maximum(norm, eps)


def softmax(x: Array, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along ``axis``."""
    arr = as_float(x)
    arr = arr - np.max(arr, axis=axis, keepdims=True)
    exp = np.exp(arr)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def mean_pool(x: Array, axis: int = 0) -> np.ndarray:
    """Mean over ``axis`` (time, by convention) keeping the feature dimension."""
    return as_float(x).mean(axis=axis)


def _pad_to_multiple(seq: np.ndarray, factor: int) -> np.ndarray:
    pad = (-seq.shape[0]) % factor
    if pad:
        seq = np.concatenate([seq, np.repeat(seq[-1:], pad, axis=0)], axis=0)
    return seq


def avgpool_time(seq: Array, factor: int) -> np.ndarray:
    """Average non-overlapping windows of ``factor`` steps along time."""
    arr = as_float(seq)
    if factor <= 1:
        return arr
    arr = _pad_to_multiple(arr, factor)
    return arr.reshape(-1, factor, arr.shape[1]).mean(axis=1)


def stack_frames(seq: Array, factor: int) -> np.ndarray:
    """Concatenate ``factor`` consecutive steps into one wider vector."""
    arr = as_float(seq)
    if factor <= 1:
        return arr
    arr = _pad_to_multiple(arr, factor)
    return arr.reshape(-1, factor * arr.shape[1])
