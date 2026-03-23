import numpy as np

from avlex.bridges.quantize import assign_codebook, kmeans, temporal_segments


def test_assign_codebook_picks_nearest():
    codebook = np.array([[0.0, 0.0], [10.0, 10.0]])
    points = np.array([[1.0, 1.0], [9.0, 9.0], [0.5, 0.0]])
    np.testing.assert_array_equal(assign_codebook(points, codebook), [0, 1, 0])


def test_kmeans_separates_two_blobs():
    rng = np.random.default_rng(0)
    blob_a = rng.normal([0.0, 0.0], 0.1, size=(20, 2))
    blob_b = rng.normal([5.0, 5.0], 0.1, size=(20, 2))
    centroids, labels = kmeans(np.vstack([blob_a, blob_b]), k=2, seed=0)
    assert len(set(labels.tolist())) == 2
    assert np.linalg.norm(centroids[0] - centroids[1]) > 3.0


def test_kmeans_is_deterministic():
    x = np.random.default_rng(1).normal(size=(30, 3))
    c1, l1 = kmeans(x, 3, seed=4)
    c2, l2 = kmeans(x, 3, seed=4)
    np.testing.assert_array_equal(l1, l2)
    np.testing.assert_allclose(c1, c2)


def test_temporal_segments_are_contiguous():
    seg = temporal_segments(10, 3)
    assert seg.min() == 0
    assert seg.max() == 2
    assert len(seg) == 10
    # ids never decrease
    assert np.all(np.diff(seg) >= 0)
