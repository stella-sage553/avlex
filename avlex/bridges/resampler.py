"""Perceiver-style resampler bridge.

A fixed set of learnable-in-spirit query tokens cross-attend the (variable
length) fused timeline and collapse it to ``num_queries`` tokens, regardless of
how long the clip is. This is the resampling trick behind Flamingo/BLIP-2 style
connectors, here with seeded weights instead of trained ones.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from avlex.bridges.attention import feed_forward, layer_norm, multi_head_attention
from avlex.bridges.base import Bridge, BridgeInput, BridgeOutput
from avlex.utils.arrays import as_float
from avlex.utils.seeding import derive_seed, seeded_matrix


@dataclass
class PerceiverResampler(Bridge):
    """Reduce a timeline to a fixed number of query tokens via cross-attention."""

    num_queries: int = 16
    out_dim: int = 512
    n_heads: int = 4
    depth: int = 2
    seed: int = 0

    def _heads_for(self, dim: int) -> int:
        return self.n_heads if dim % self.n_heads == 0 else 1

    def bridge(self, features: BridgeInput) -> BridgeOutput:
        seq = as_float(features.sequence)
        if seq.ndim != 2 or seq.shape[0] == 0:
            raise ValueError("PerceiverResampler needs a non-empty (T, D) sequence")
        dim = seq.shape[1]
        heads = self._heads_for(dim)
        queries = seeded_matrix(
            derive_seed(self.seed, "perceiver", "queries", str(dim)),
            (self.num_queries, dim),
            scale=1.0,
        )
        x = queries
        for layer in range(self.depth):
            attended = multi_head_attention(x, seq, seq, n_heads=heads)
            x = layer_norm(x + attended)
            x = layer_norm(x + feed_forward(x, derive_seed(self.seed, "perceiver", str(layer))))
        weight = seeded_matrix(
            derive_seed(self.seed, "perceiver", "out", str(dim), str(self.out_dim)),
            (dim, self.out_dim),
        )
        tokens = x @ weight
        return BridgeOutput(tokens=tokens, meta={"num_queries": self.num_queries})
