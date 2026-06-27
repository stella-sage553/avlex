import numpy as np

from avlex.bridges import BridgeInput, LinearProjector


def _seq(t: int = 10, d: int = 16) -> BridgeInput:
    rng = np.random.default_rng(0)
    return BridgeInput(sequence=rng.normal(size=(t, d)))


def test_avgpool_reduces_token_count():
    bridge = LinearProjector(out_dim=32, compression="avgpool", factor=2)
    out = bridge.bridge(_seq(t=10, d=16))
    assert out.tokens is not None
    assert out.tokens.shape == (5, 32)


def test_stack_widens_then_projects():
    bridge = LinearProjector(out_dim=32, compression="stack", factor=2)
    out = bridge.bridge(_seq(t=10, d=16))
    assert out.num_tokens == 5
    assert out.tokens.shape[1] == 32


def test_no_compression_keeps_all_steps():
    bridge = LinearProjector(out_dim=8, compression="none")
    out = bridge.bridge(_seq(t=7, d=16))
    assert out.num_tokens == 7


def test_projector_is_deterministic():
    bridge = LinearProjector(out_dim=12)
    inp = _seq()
    np.testing.assert_array_equal(bridge.bridge(inp).tokens, bridge.bridge(inp).tokens)


def test_empty_sequence_raises():
    bridge = LinearProjector()
    try:
        bridge.bridge(BridgeInput(sequence=np.zeros((0, 16))))
    except ValueError:
        return
    raise AssertionError("expected ValueError on empty sequence")
