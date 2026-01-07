"""Temporal helpers shared by the fusion strategies."""

from __future__ import annotations

import numpy as np

from avlex.types import Array
from avlex.utils.arrays import as_float
from avlex.utils.seeding import derive_seed, seeded_matrix


def resample_time(seq: Array, target_len: int) -> np.ndarray:
    """Linearly resample a ``(T, D)`` sequence to ``(target_len, D)``."""
    arr = as_float(seq)
    n = arr.shape[0]
    if target_len <= 0:
        raise ValueError("target_len must be positive")
    if n == target_len:
        return arr
    if n == 1:
        return np.repeat(arr, target_len, axis=0)
    src = np.linspace(0.0, 1.0, n)
    dst = np.linspace(0.0, 1.0, target_len)
    return np.stack(
        [np.interp(dst, src, arr[:, d]) for d in range(arr.shape[1])], axis=1
    )


def project_modality(seq: Array, out_dim: int, seed: int) -> np.ndarray:
    """Project ``(T, D)`` features to ``(T, out_dim)`` with a seeded linear map."""
    arr = as_float(seq)
    in_dim = arr.shape[1]
    weight = seeded_matrix(
        derive_seed(seed, "fuse_proj", str(in_dim), str(out_dim)), (in_dim, out_dim)
    )
    return arr @ weight
