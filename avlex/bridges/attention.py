"""NumPy attention primitives used by the resampler and Q-Former bridges.

These are plain, un-trained building blocks: the bridges draw their projection
weights from a seed (see :mod:`avlex.utils.seeding`), so the maths here only has
to be correct, not learned.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from avlex.types import Array
from avlex.utils.arrays import as_float, softmax


def scaled_dot_product_attention(
    query: Array,
    key: Array,
    value: Array,
    mask: Optional[Array] = None,
) -> np.ndarray:
    """Attention over ``(Tq, d) x (Tk, d) x (Tk, dv) -> (Tq, dv)``.

    ``mask`` is a boolean array broadcastable to the score matrix; ``False``
    entries are not attended to.
    """
    q, k, v = as_float(query), as_float(key), as_float(value)
    scores = (q @ k.T) / np.sqrt(q.shape[-1])
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    weights = softmax(scores, axis=-1)
    return weights @ v


def layer_norm(x: Array, eps: float = 1e-5) -> np.ndarray:
    """Standardize the last axis of ``x`` (no learnable affine)."""
    arr = as_float(x)
    mean = arr.mean(axis=-1, keepdims=True)
    var = arr.var(axis=-1, keepdims=True)
    return (arr - mean) / np.sqrt(var + eps)
