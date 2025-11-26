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


def multi_head_attention(
    query: Array,
    key: Array,
    value: Array,
    n_heads: int = 1,
    mask: Optional[Array] = None,
) -> np.ndarray:
    """Head-split attention: slice the feature dim into ``n_heads`` and concat.

    The model dimension must be divisible by ``n_heads``. With ``n_heads == 1``
    this is exactly :func:`scaled_dot_product_attention`.
    """
    q, k, v = as_float(query), as_float(key), as_float(value)
    dim = q.shape[-1]
    if dim % n_heads != 0:
        raise ValueError(f"feature dim {dim} not divisible by n_heads {n_heads}")
    head_dim = dim // n_heads
    heads = []
    for h in range(n_heads):
        sl = slice(h * head_dim, (h + 1) * head_dim)
        heads.append(scaled_dot_product_attention(q[:, sl], k[:, sl], v[:, sl], mask))
    return np.concatenate(heads, axis=-1)


def feed_forward(x: Array, seed: int, hidden_mult: int = 4) -> np.ndarray:
    """A fixed two-layer ReLU MLP with seeded weights (residual block helper)."""
    from avlex.utils.seeding import derive_seed, seeded_matrix

    arr = as_float(x)
    dim = arr.shape[-1]
    w1 = seeded_matrix(derive_seed(seed, "ffn", "w1"), (dim, dim * hidden_mult))
    w2 = seeded_matrix(derive_seed(seed, "ffn", "w2"), (dim * hidden_mult, dim))
    hidden = np.maximum(arr @ w1, 0.0)
    return hidden @ w2
