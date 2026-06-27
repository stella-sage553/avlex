import numpy as np

from avlex.bridges import BridgeInput, TokenBridge
from avlex.types import Modality


def _rising_visual(t: int = 12, d: int = 32) -> np.ndarray:
    # row magnitude grows over time -> activity that builds
    base = np.random.default_rng(0).random((t, d))
    scale = np.linspace(0.2, 2.0, t)[:, None]
    return base * scale


def _audio(t: int = 20, d: int = 40) -> np.ndarray:
    return np.abs(np.random.default_rng(1).normal(size=(t, d)))


def _input() -> BridgeInput:
    mods = {Modality.VISUAL: _rising_visual(), Modality.AUDIO: _audio()}
    return BridgeInput(sequence=np.zeros((1, 4)), modalities=mods)


def test_token_bridge_emits_words():
    out = TokenBridge().bridge(_input())
    assert out.words
    tags = [w.split(":")[0] for w in out.words]
    assert "visual" in tags
    assert "audio" in tags
    assert "timeline" in tags


def test_token_bridge_is_deterministic():
    inp = _input()
    assert TokenBridge().bridge(inp).words == TokenBridge().bridge(inp).words


def test_rising_activity_reads_as_building():
    out = TokenBridge().bridge(_input())
    timeline = next(w for w in out.words if w.startswith("timeline:"))
    assert "builds" in timeline


def test_timeline_can_be_disabled():
    out = TokenBridge(include_timeline=False).bridge(_input())
    assert all(not w.startswith("timeline:") for w in out.words)
