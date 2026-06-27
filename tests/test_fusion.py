import numpy as np

from avlex.fusion import ConcatFusion, GatedFusion, InterleaveFusion, get_fusion
from avlex.fusion.temporal import resample_time
from avlex.types import Modality


def _features():
    rng = np.random.default_rng(0)
    return {
        Modality.VISUAL: rng.random((5, 32)),
        Modality.AUDIO: rng.random((8, 40)),
    }


def test_resample_time_changes_length_only():
    seq = np.random.default_rng(1).random((3, 4))
    out = resample_time(seq, 9)
    assert out.shape == (9, 4)
    # endpoints are preserved by linear interpolation
    np.testing.assert_allclose(out[0], seq[0])
    np.testing.assert_allclose(out[-1], seq[-1])


def test_resample_constant_sequence_is_constant():
    seq = np.tile([1.0, 2.0, 3.0], (4, 1))
    np.testing.assert_allclose(resample_time(seq, 7), np.tile([1.0, 2.0, 3.0], (7, 1)))


def test_concat_fusion_lengths_and_spans():
    fused = ConcatFusion(d_model=64).fuse(_features())
    assert fused.sequence.shape == (13, 64)
    assert fused.spans[Modality.VISUAL] == (0, 5)
    assert fused.spans[Modality.AUDIO] == (5, 13)
    # original features survive for the token bridge
    assert set(fused.modalities) == {Modality.VISUAL, Modality.AUDIO}


def test_interleave_doubles_aligned_length():
    fused = InterleaveFusion(d_model=64).fuse(_features())
    assert fused.sequence.shape == (16, 64)


def test_gated_fusion_aligns_to_longest():
    fused = GatedFusion(d_model=64).fuse(_features())
    assert fused.sequence.shape == (8, 64)
    assert np.all(np.isfinite(fused.sequence))


def test_single_modality_is_allowed():
    fused = ConcatFusion(d_model=32).fuse({Modality.AUDIO: np.ones((6, 40))})
    assert fused.sequence.shape == (6, 32)


def test_fusion_registry():
    assert isinstance(get_fusion("gated", d_model=16), GatedFusion)
