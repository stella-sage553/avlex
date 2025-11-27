import numpy as np

from avlex.bridges.attention import (
    feed_forward,
    layer_norm,
    multi_head_attention,
    scaled_dot_product_attention,
)


def test_attention_output_shape():
    rng = np.random.default_rng(0)
    q = rng.normal(size=(3, 8))
    k = rng.normal(size=(5, 8))
    v = rng.normal(size=(5, 4))
    out = scaled_dot_product_attention(q, k, v)
    assert out.shape == (3, 4)


def test_uniform_keys_average_values():
    # identical keys -> uniform weights -> output is the mean of the values
    q = np.ones((1, 4))
    k = np.zeros((6, 4))
    v = np.random.default_rng(1).normal(size=(6, 3))
    out = scaled_dot_product_attention(q, k, v)
    np.testing.assert_allclose(out[0], v.mean(axis=0))


def test_mask_blocks_positions():
    q = np.ones((1, 2))
    k = np.ones((3, 2))
    v = np.array([[1.0], [2.0], [9.0]])
    mask = np.array([[True, True, False]])
    out = scaled_dot_product_attention(q, k, v, mask=mask)
    np.testing.assert_allclose(out[0], [1.5])


def test_layer_norm_zero_mean_unit_var():
    x = np.random.default_rng(2).normal(size=(4, 16))
    out = layer_norm(x)
    np.testing.assert_allclose(out.mean(axis=-1), np.zeros(4), atol=1e-9)
    np.testing.assert_allclose(out.std(axis=-1), np.ones(4), atol=1e-3)


def test_multi_head_matches_single_head_when_one():
    rng = np.random.default_rng(3)
    q, k, v = rng.normal(size=(2, 8)), rng.normal(size=(4, 8)), rng.normal(size=(4, 8))
    single = scaled_dot_product_attention(q, k, v)
    multi = multi_head_attention(q, k, v, n_heads=1)
    np.testing.assert_allclose(single, multi)


def test_multi_head_requires_divisible_dim():
    q = np.ones((2, 6))
    try:
        multi_head_attention(q, q, q, n_heads=4)
    except ValueError:
        return
    raise AssertionError("expected ValueError for indivisible dim")


def test_feed_forward_is_deterministic():
    x = np.random.default_rng(4).normal(size=(3, 8))
    np.testing.assert_array_equal(feed_forward(x, seed=7), feed_forward(x, seed=7))
