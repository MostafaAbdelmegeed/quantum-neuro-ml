import numpy as np
from quantum_neuro_ml.permutation import permuted_labels


def test_permutation_preserves_each_site_class_count_and_is_reproducible():
    labels = np.tile([0, 0, 0, 1, 1, 1], 4)
    groups = np.repeat(list("ABCD"), 6)
    shuffled = permuted_labels(labels, groups, 42, 0)
    np.testing.assert_array_equal(shuffled, permuted_labels(labels, groups, 42, 0))
    assert not np.array_equal(shuffled, labels)
    for group in "ABCD":
        np.testing.assert_array_equal(np.bincount(shuffled[groups == group]), np.bincount(labels[groups == group]))
