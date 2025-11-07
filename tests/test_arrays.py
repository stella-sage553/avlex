import numpy as np

from avlex.utils.arrays import l2_normalize, mean_pool, softmax


def test_l2_normalize_unit_rows():
    x = np.array([[3.0, 4.0], [0.0, 0.0]])
    out = l2_normalize(x)
    np.testing.assert_allclose(np.linalg.norm(out[0]), 1.0)
    # a zero row must not blow up to nan/inf
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out[1], [0.0, 0.0])


def test_softmax_rows_sum_to_one():
    x = np.random.default_rng(0).normal(size=(4, 5))
    s = softmax(x, axis=-1)
    np.testing.assert_allclose(s.sum(axis=-1), np.ones(4))
    assert np.all(s >= 0)


def test_softmax_is_shift_invariant():
    x = np.array([[1.0, 2.0, 3.0]])
    np.testing.assert_allclose(softmax(x), softmax(x + 100.0))


def test_mean_pool_matches_numpy():
    x = np.arange(12, dtype=float).reshape(3, 4)
    np.testing.assert_allclose(mean_pool(x, axis=0), x.mean(axis=0))
