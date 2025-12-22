import numpy as np

from avlex.bridges import BridgeInput, QFormerBridge


def _seq(t: int = 15, d: int = 16) -> BridgeInput:
    return BridgeInput(sequence=np.random.default_rng(1).normal(size=(t, d)))


def test_qformer_output_shape():
    bridge = QFormerBridge(num_queries=6, out_dim=24)
    out = bridge.bridge(_seq())
    assert out.tokens.shape == (6, 24)


def test_qformer_is_deterministic():
    bridge = QFormerBridge(num_queries=5, out_dim=16)
    inp = _seq()
    np.testing.assert_array_equal(bridge.bridge(inp).tokens, bridge.bridge(inp).tokens)


def test_qformer_falls_back_to_one_head_when_indivisible():
    # d=10 is not divisible by the default 4 heads; should still run
    bridge = QFormerBridge(num_queries=3, out_dim=8, n_heads=4)
    out = bridge.bridge(BridgeInput(sequence=np.ones((12, 10))))
    assert out.tokens.shape == (3, 8)
