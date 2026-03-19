import numpy as np
import pytest

from avlex.sampling import motion_keyframe_indices, sample_frames, uniform_indices


def test_uniform_spans_endpoints():
    idx = uniform_indices(100, 5)
    assert len(idx) == 5
    assert idx[0] == 0
    assert idx[-1] == 99


def test_uniform_returns_all_when_k_exceeds_n():
    np.testing.assert_array_equal(uniform_indices(4, 10), np.arange(4))


def test_uniform_rejects_bad_args():
    with pytest.raises(ValueError):
        uniform_indices(0, 3)
    with pytest.raises(ValueError):
        uniform_indices(5, 0)


def test_motion_keyframes_prefer_change():
    frames = np.zeros((6, 4, 4))
    frames[3] = 1.0  # a big change appears at frame 3 and reverts at frame 4
    idx = motion_keyframe_indices(frames, 2)
    assert 3 in idx
    assert list(idx) == sorted(idx)


def test_sample_frames_motion_count():
    frames = np.random.default_rng(0).random((10, 4, 4))
    assert sample_frames(frames, 3, strategy="motion").shape[0] == 3


def test_sample_frames_unknown_strategy():
    with pytest.raises(ValueError):
        sample_frames(np.zeros((5, 2, 2)), 2, strategy="bogus")
