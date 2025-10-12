"""Deterministic randomness helpers.

avlex never trains weights; bridges that need a fixed projection draw it from a
seed so the same configuration always produces the same tokens. We hash names
into seeds with SHA-256 instead of the builtin :func:`hash`, which is salted per
process and would break reproducibility across runs.
"""

from __future__ import annotations

import hashlib

import numpy as np


def rng_from_seed(seed: int) -> np.random.Generator:
    """Return a fresh NumPy generator for ``seed``."""
    return np.random.default_rng(seed)


def derive_seed(base_seed: int, *names: str) -> int:
    """Combine ``base_seed`` with string ``names`` into a stable 64-bit seed."""
    digest = hashlib.sha256(str(base_seed).encode("utf-8"))
    for name in names:
        digest.update(b"\x00")
        digest.update(name.encode("utf-8"))
    return int.from_bytes(digest.digest()[:8], "big")


def seeded_matrix(
    seed: int,
    shape: tuple[int, int],
    scale: float | None = None,
) -> np.ndarray:
    """Draw a deterministic Gaussian matrix of ``shape`` for ``seed``.

    The default scale follows the usual ``1/sqrt(fan_in)`` initialization so the
    projected features keep a sensible magnitude.
    """
    fan_in = shape[0]
    if scale is None:
        scale = 1.0 / max(fan_in, 1) ** 0.5
    return rng_from_seed(seed).standard_normal(shape) * scale
