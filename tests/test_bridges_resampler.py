import numpy as np

from avlex.bridges import BridgeInput, PerceiverResampler


def _seq(t: int, d: int = 16) -> BridgeInput:
    return BridgeInput(sequence=np.random.default_rng(t).normal(size=(t, d)))


def test_resampler_emits_fixed_query_count():
    bridge = PerceiverResampler(num_queries=8, out_dim=32)
    out = bridge.bridge(_seq(20))
    assert out.tokens.shape == (8, 32)


def test_token_count_independent_of_input_length():
    bridge = PerceiverResampler(num_queries=8, out_dim=32)
    short = bridge.bridge(_seq(10))
    long = bridge.bridge(_seq(60))
    assert short.tokens.shape == long.tokens.shape == (8, 32)


def test_resampler_is_deterministic():
    bridge = PerceiverResampler(num_queries=4, out_dim=16)
    inp = _seq(15)
    np.testing.assert_array_equal(bridge.bridge(inp).tokens, bridge.bridge(inp).tokens)


def test_resampler_output_is_finite():
    bridge = PerceiverResampler(num_queries=6, out_dim=16, depth=3)
    out = bridge.bridge(_seq(25))
    assert np.all(np.isfinite(out.tokens))
