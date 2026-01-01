"""Q-Former style bridge.

Like the resampler, but each block first lets the query tokens talk to *each
other* (self-attention) before attending to the audio-visual features
(cross-attention). This is the BERT-based Q-Former pattern used by BLIP-2,
Video-LLaMA, and Bay-CAT, reduced to its structural essentials.
"""

from __future__ import annotations

from dataclasses import dataclass

from avlex.bridges.attention import feed_forward, layer_norm, multi_head_attention
from avlex.bridges.base import Bridge, BridgeInput, BridgeOutput
from avlex.utils.arrays import as_float
from avlex.utils.seeding import derive_seed, seeded_matrix


@dataclass
class QFormerBridge(Bridge):
    """Query tokens with interleaved self- and cross-attention blocks."""

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
            raise ValueError("QFormerBridge needs a non-empty (T, D) sequence")
        dim = seq.shape[1]
        heads = self._heads_for(dim)
        x = seeded_matrix(
            derive_seed(self.seed, "qformer", "queries", str(dim)),
            (self.num_queries, dim),
            scale=1.0,
        )
        for layer in range(self.depth):
            x = layer_norm(x + multi_head_attention(x, x, x, n_heads=heads))
            x = layer_norm(x + multi_head_attention(x, seq, seq, n_heads=heads))
            x = layer_norm(
                x + feed_forward(x, derive_seed(self.seed, "qformer", str(layer)))
            )
        weight = seeded_matrix(
            derive_seed(self.seed, "qformer", "out", str(dim), str(self.out_dim)),
            (dim, self.out_dim),
        )
        tokens = x @ weight
        return BridgeOutput(tokens=tokens, meta={"num_queries": self.num_queries})
