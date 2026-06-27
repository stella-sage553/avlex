"""Deterministic vector quantization and temporal segmentation.

Used by the token bridge to carve a clip into a handful of contiguous "events"
and to map feature vectors onto a fixed codebook.
"""

from __future__ import annotations

import numpy as np

from avlex.types import Array
from avlex.utils.arrays import as_float
from avlex.utils.seeding import rng_from_seed


def assign_codebook(x: Array, codebook: Array) -> np.ndarray:
    """Nearest-centroid assignment; ties resolve to the lowest index."""
    pts = as_float(x)
    book = as_float(codebook)
    dist = ((pts[:, None, :] - book[None, :, :]) ** 2).sum(axis=-1)
    return np.argmin(dist, axis=1)


def kmeans(
    x: Array,
    k: int,
    n_iter: int = 25,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """A small deterministic Lloyd's k-means. Returns ``(centroids, labels)``."""
    pts = as_float(x)
    n = pts.shape[0]
    if k <= 0:
        raise ValueError("k must be positive")
    k = min(k, n)
    init = rng_from_seed(seed).choice(n, size=k, replace=False)
    centroids = pts[np.sort(init)].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(n_iter):
        labels = assign_codebook(pts, centroids)
        moved = False
        for j in range(k):
            members = pts[labels == j]
            if members.size:
                new = members.mean(axis=0)
                if not np.allclose(new, centroids[j]):
                    centroids[j] = new
                    moved = True
        if not moved:
            break
    return centroids, labels


def temporal_segments(n: int, k: int) -> np.ndarray:
    """Split ``range(n)`` into ``k`` near-equal contiguous segment ids."""
    if n <= 0:
        raise ValueError("n must be positive")
    k = max(1, min(k, n))
    return np.minimum(np.arange(n) * k // n, k - 1)
