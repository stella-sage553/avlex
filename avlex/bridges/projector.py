"""Linear projector bridge.

The simplest soft bridge: optionally compress the fused timeline, then project
each step into the LLM embedding dimension with a fixed (seeded) linear map. This
mirrors the MLP projectors used by Llama-AVSR and friends, minus the training.
"""

from __future__ import annotations

from dataclasses import dataclass

from avlex.bridges.base import Bridge, BridgeInput, BridgeOutput
from avlex.utils.arrays import as_float, avgpool_time, stack_frames
from avlex.utils.seeding import derive_seed, seeded_matrix


@dataclass
class LinearProjector(Bridge):
    """Compress then linearly project the fused features to ``out_dim`` tokens."""

    out_dim: int = 512
    compression: str = "avgpool"  # "avgpool" | "stack" | "none"
    factor: int = 2
    seed: int = 0

    def _compress(self, seq):
        if self.compression == "avgpool":
            return avgpool_time(seq, self.factor)
        if self.compression == "stack":
            return stack_frames(seq, self.factor)
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
