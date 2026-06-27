import numpy as np
import pytest

from avlex import ClipFeatures, synthetic_clip


def test_requires_at_least_one_modality():
    with pytest.raises(ValueError):
        ClipFeatures()


def test_modality_flags():
    clip = ClipFeatures(visual=np.zeros((3, 4, 4)))
    assert clip.has_visual
    assert not clip.has_audio


def test_rejects_non_positive_fps():
    with pytest.raises(ValueError):
        ClipFeatures(audio=np.zeros(10), fps=0.0)


def test_synthetic_clip_is_deterministic():
    a = synthetic_clip(seed=1)
    b = synthetic_clip(seed=1)
    np.testing.assert_array_equal(a.visual, b.visual)
    np.testing.assert_array_equal(a.audio, b.audio)


def test_synthetic_clip_shapes():
    clip = synthetic_clip(n_frames=10, size=8, duration=1.0, sample_rate=8000)
    assert clip.visual.shape == (10, 8, 8)
    assert clip.audio.shape == (8000,)
    assert clip.meta["synthetic"] is True
