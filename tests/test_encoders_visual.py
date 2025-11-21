import numpy as np

from avlex.encoders import ColorStatsEncoder, MotionHistogramEncoder


def _moving_clip(seed: int = 0, n: int = 6, size: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((n, size, size))


def test_motion_encoder_shapes():
    enc = MotionHistogramEncoder(n_bins=24)
    feats = enc.encode(_moving_clip())
    assert feats.shape == (6, 24)
    assert enc.output_dim == 24
    # each non-first frame is a probability histogram
    np.testing.assert_allclose(feats[1:].sum(axis=1), np.ones(5))


def test_motion_first_frame_is_zero():
    enc = MotionHistogramEncoder()
    feats = enc.encode(_moving_clip())
    np.testing.assert_allclose(feats[0], 0.0)


def test_motion_encoder_is_deterministic():
    enc = MotionHistogramEncoder()
    clip = _moving_clip(seed=3)
    np.testing.assert_array_equal(enc.encode(clip), enc.encode(clip))


def test_static_clip_has_more_motion_than_none():
    enc = MotionHistogramEncoder(n_bins=8)
    static = np.ones((5, 8, 8))
    moving = _moving_clip(seed=1)
    # low bin holds the "no change" mass; static frames sit entirely there
    assert enc.encode(static)[1:, 0].mean() > enc.encode(moving)[1:, 0].mean()


def test_color_stats_handles_rgb():
    enc = ColorStatsEncoder(n_bins=12)
    frames = np.random.default_rng(0).random((4, 8, 8, 3))
    feats = enc.encode(frames)
    assert feats.shape == (4, 12)
    np.testing.assert_allclose(feats.sum(axis=1), np.ones(4))
