"""Linear projector bridge.

The simplest soft bridge: optionally compress the fused timeline, then project
each step into the LLM embedding dimension with a fixed (seeded) linear map. This
mirrors the MLP projectors used by Llama-AVSR and friends, minus the training.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from avlex.bridges.base import Bridge, BridgeInput, BridgeOutput
from avlex.utils.arrays import as_float
from avlex.utils.seeding import derive_seed, seeded_matrix


def _avgpool_time(seq: np.ndarray, factor: int) -> np.ndarray:
    """Average non-overlapping windows of ``factor`` steps along time."""
    if factor <= 1:
        return seq
    n = seq.shape[0]
    pad = (-n) % factor
    if pad:
        seq = np.concatenate([seq, np.repeat(seq[-1:], pad, axis=0)], axis=0)
    return seq.reshape(-1, factor, seq.shape[1]).mean(axis=1)


def _stack_frames(seq: np.ndarray, factor: int) -> np.ndarray:
    """Concatenate ``factor`` consecutive steps into one wider vector."""
    if factor <= 1:
        return seq
    n = seq.shape[0]
    pad = (-n) % factor
    if pad:
        seq = np.concatenate([seq, np.repeat(seq[-1:], pad, axis=0)], axis=0)
    return seq.reshape(-1, factor * seq.shape[1])


@dataclass
class LinearProjector(Bridge):
    """Compress then linearly project the fused features to ``out_dim`` tokens."""

    out_dim: int = 512
    compression: str = "avgpool"  # "avgpool" | "stack" | "none"
    factor: int = 2
    seed: int = 0

    def _compress(self, seq: np.ndarray) -> np.ndarray:
        if self.compression == "avgpool":
            return _avgpool_time(seq, self.factor)
        if self.compression == "stack":
            return _stack_frames(seq, self.factor)
        if self.compression == "none":
            return seq
        raise ValueError(f"unknown compression: {self.compression!r}")

    def bridge(self, features: BridgeInput) -> BridgeOutput:
        seq = as_float(features.sequence)
        if seq.ndim != 2 or seq.shape[0] == 0:
            raise ValueError("LinearProjector needs a non-empty (T, D) sequence")
        compressed = self._compress(seq)
        in_dim = compressed.shape[1]
        weight = seeded_matrix(
            derive_seed(self.seed, "projector", str(in_dim), str(self.out_dim)),
            (in_dim, self.out_dim),
        )
        tokens = compressed @ weight
        return BridgeOutput(
            tokens=tokens,
            meta={"compression": self.compression, "factor": self.factor},
        )
