# tests/test_postprocess.py
import numpy as np
import pytest

from rice_ml.supervised_learning import postprocess as pp


# ---------------------------
# majority_vote
# ---------------------------

def test_majority_vote_unweighted_numeric_simple():
    labels = np.array([0, 1, 1, 2, 1, 0])
    out = pp.majority_vote(labels)
    assert out == 1  # most frequent


def test_majority_vote_unweighted_strings():
    labels = np.array(["cat", "dog", "cat", "cat", "dog"], dtype=object)
    out = pp.majority_vote(labels)
    assert out == "cat"


def test_majority_vote_weighted_sum_correct():
    # label 0 has two neighbors but lower total weight than label 1
    labels = np.array([0, 0, 1, 1, 1])
    weights = np.array([0.1, 0.2, 0.05, 0.4, 0.3])  # sums: label0=0.3, label1=0.75
    out = pp.majority_vote(labels, weights=weights)
    assert out == 1


def test_majority_vote_tie_break_numeric_smallest_label():
    labels = np.array([1, 2])
    weights = np.array([1.0, 1.0])  # tie on total weight
    out = pp.majority_vote(labels, weights=weights)
    # deterministic tie-break chooses smallest label after np.unique sorting
    assert out == 1


def test_majority_vote_tie_break_strings_alphabetical():
    labels = np.array(["dog", "cat"], dtype=object)
    weights = np.array([1.0, 1.0])
    out = pp.majority_vote(labels, weights=weights)
    # np.unique sorts -> ["cat","dog"], so tie -> "cat"
    assert out == "cat"


def test_majority_vote_order_invariant_permutation():
    labels = np.array([2, 2, 1, 1])
    weights = np.array([0.4, 0.1, 0.2, 0.3])  # totals: lab2=0.5, lab1=0.5 (tie -> pick 1)
    out1 = pp.majority_vote(labels, weights)
    # permute order; result must be same due to aggregation + deterministic tie-break
    perm = np.array([3, 2, 1, 0])
    out2 = pp.majority_vote(labels[perm], weights[perm])
    assert out1 == 1 and out2 == 1


def test_majority_vote_all_same_label():
    labels = np.array([7, 7, 7, 7])
    out = pp.majority_vote(labels)
    assert out == 7


def test_majority_vote_all_zero_weights_returns_smallest_label():
    labels = np.array([5, 2, 9])
    weights = np.array([0.0, 0.0, 0.0])
    out = pp.majority_vote(labels, weights=weights)
    # agg is all zeros -> np.argmax -> index 0 in uniq -> smallest label (2)
    assert out == 2


def test_majority_vote_weights_length_mismatch_raises():
    labels = np.array([0, 1, 1])
    weights = np.array([0.5, 0.5])  # wrong shape
    with pytest.raises(ValueError):
        pp.majority_vote(labels, weights=weights)


def test_majority_vote_input_immutability():
    labels = np.array([0, 1, 1, 2])
    weights = np.array([0.2, 0.3, 0.4, 0.1])
    labels_copy = labels.copy()
    weights_copy = weights.copy()
    _ = pp.majority_vote(labels, weights)
    np.testing.assert_array_equal(labels, labels_copy)
    np.testing.assert_array_equal(weights, weights_copy)


# ---------------------------
# average_label
# ---------------------------

def test_average_label_unweighted_mean():
    vals = np.array([1.0, 2.0, 3.0, 4.0])
    out = pp.average_label(vals)
    assert out == pytest.approx(np.mean(vals))


def test_average_label_weighted_mean_basic():
    vals = np.array([10.0, 20.0, 30.0])
    w = np.array([1.0, 1.0, 2.0])
    expected = float(np.dot(vals, w) / w.sum())
    out = pp.average_label(vals, weights=w)
    assert out == pytest.approx(expected)


def test_average_label_weighted_single_dominant_neighbor():
    vals = np.array([1.0, 100.0, -50.0])
    w = np.array([0.0, 1.0, 0.0])  # should pick 100.0
    out = pp.average_label(vals, weights=w)
    assert out == 100.0


def test_average_label_zero_weight_sum_falls_back_to_mean():
    vals = np.array([2.0, 4.0, 6.0])
    w = np.array([0.0, 0.0, 0.0])
    out = pp.average_label(vals, weights=w)
    assert out == pytest.approx(vals.mean())


def test_average_label_returns_float_dtype():
    vals = np.array([1, 2, 3], dtype=int)
    out = pp.average_label(vals)
    assert isinstance(out, float)


def test_average_label_weights_length_mismatch_raises():
    vals = np.array([1.0, 2.0, 3.0])
    w = np.array([1.0, 2.0])  # wrong length
    with pytest.raises(ValueError):
        pp.average_label(vals, weights=w)


def test_average_label_input_immutability():
    vals = np.array([3.0, 5.0, 7.0])
    w = np.array([1.0, 2.0, 3.0])
    vals_copy = vals.copy()
    w_copy = w.copy()
    _ = pp.average_label(vals, weights=w)
    np.testing.assert_array_equal(vals, vals_copy)
    np.testing.assert_array_equal(w, w_copy)


# ---------------------------
# Mixed robustness
# ---------------------------

def test_handles_non_contiguous_inputs():
    vals = np.arange(10, dtype=float)
    vals_view = vals[::2]  # non-contiguous
    w = np.linspace(1, 5, num=vals_view.size)
    # functions should work and shapes align
    v = pp.average_label(vals_view, weights=w)
    assert isinstance(v, float)

    labels = np.array([0, 1, 0, 2, 2, 1])[::2]  # [0,0,2]
    lw = np.array([0.3, 0.4, 0.3])
    mv = pp.majority_vote(labels, weights=lw)
    assert mv in (0, 2)
