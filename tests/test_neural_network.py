import sys
from pathlib import Path

# Start at the current directory
root = Path().resolve()

while not (root / "src" / "rice_ml").exists() and root != root.parent:
    root = root.parent

sys.path.append(str(root / "src"))


import numpy as np
import pytest

from rice_ml.supervised_learning.neural_network import (
    NeuralNetwork,
    _relu,
    _relu_grad,
    _softmax,
)


# ----------------------------------------------------------------------
# Helper fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def simple_multiclass_data():
    """
    Very simple 3-class classification problem in 2D.

    Base points:
        [0, 0] -> class 0
        [0, 1] -> class 1
        [1, 0] -> class 2

    We repeat them several times to make training easier.
    """
    X_base = np.array([[0.0, 0.0],
                       [0.0, 1.0],
                       [1.0, 0.0]])
    y_base = np.array([0, 1, 2])

    n_repeats = 40
    X = np.vstack([X_base] * n_repeats)
    y = np.tile(y_base, n_repeats)
    return X, y


@pytest.fixture
def simple_regression_data():
    """
    Simple 1D regression: y = 2x + 0.5
    """
    X = np.linspace(-1, 1, 50).reshape(-1, 1)
    y = 2 * X.squeeze() + 0.5
    return X, y


# ----------------------------------------------------------------------
# Helper function tests: ReLU, ReLU grad, softmax
# ----------------------------------------------------------------------


def test_relu_and_grad_basic():
    z = np.array([-2.0, 0.0, 3.5])
    r = _relu(z)
    g = _relu_grad(z)

    np.testing.assert_allclose(r, np.array([0.0, 0.0, 3.5]))
    np.testing.assert_allclose(g, np.array([0.0, 0.0, 1.0]))


def test_softmax_rowwise_properties():
    z = np.array([[1.0, 2.0, 3.0],
                  [1000.0, 1001.0, 1002.0]])  # check numerical stability

    s = _softmax(z)

    # shape preserved
    assert s.shape == z.shape

    # each row sums to 1
    row_sums = np.sum(s, axis=1)
    np.testing.assert_allclose(row_sums, np.ones_like(row_sums))

    # all entries are between 0 and 1
    assert np.all(s >= 0.0)
    assert np.all(s <= 1.0)

    # shift invariance: softmax(z) == softmax(z + c)
    c = np.array([[10.0, 10.0, 10.0],
                  [-5.0, -5.0, -5.0]])
    s_shifted = _softmax(z + c)
    np.testing.assert_allclose(s, s_shifted, atol=1e-7)


# ----------------------------------------------------------------------
# Initialization / constructor tests
# ----------------------------------------------------------------------


def test_init_classification_requires_n_classes():
    with pytest.raises(ValueError, match="n_classes must be provided"):
        _ = NeuralNetwork(
            n_inputs=4,
            hidden_size=8,
            task="multiclass_classification",
            n_classes=None,
        )


def test_init_classification_requires_at_least_two_classes():
    with pytest.raises(ValueError, match=">= 2"):
        _ = NeuralNetwork(
            n_inputs=4,
            hidden_size=8,
            task="multiclass_classification",
            n_classes=1,
        )


def test_init_regression_ignores_n_classes():
    nn = NeuralNetwork(
        n_inputs=3,
        hidden_size=None,
        task="regression",
        n_classes=10,  # should be ignored
    )
    assert nn.n_outputs == 1
    assert nn.n_classes == 10  # stored but irrelevant for outputs
    assert nn.W2_.shape == (3, 1)
    assert nn.b2_.shape == (1,)


def test_init_with_hidden_layer_shapes():
    n_inputs = 5
    hidden_size = 7
    n_classes = 4
    nn = NeuralNetwork(
        n_inputs=n_inputs,
        hidden_size=hidden_size,
        task="multiclass_classification",
        n_classes=n_classes,
        random_state=0,
    )

    assert nn.W1_.shape == (n_inputs, hidden_size)
    assert nn.b1_.shape == (hidden_size,)
    assert nn.W2_.shape == (hidden_size, n_classes)
    assert nn.b2_.shape == (n_classes,)


def test_init_without_hidden_layer_shapes():
    n_inputs = 5
    n_classes = 3
    nn = NeuralNetwork(
        n_inputs=n_inputs,
        hidden_size=None,
        task="multiclass_classification",
        n_classes=n_classes,
        random_state=0,
    )

    assert nn.W1_ is None
    assert nn.b1_ is None
    assert nn.W2_.shape == (n_inputs, n_classes)
    assert nn.b2_.shape == (n_classes,)


# ----------------------------------------------------------------------
# Forward pass tests
# ----------------------------------------------------------------------


def test_forward_shapes_classification_with_hidden(simple_multiclass_data):
    X, _ = simple_multiclass_data
    nn = NeuralNetwork(
        n_inputs=2,
        hidden_size=5,
        task="multiclass_classification",
        n_classes=3,
        random_state=0,
    )

    batch_X = X[:10]
    h_pre, h_act, logits, probs = nn._forward(batch_X)

    assert h_pre.shape == (10, 5)
    assert h_act.shape == (10, 5)
    assert logits.shape == (10, 3)
    assert probs.shape == (10, 3)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(10))


def test_forward_shapes_regression_no_hidden(simple_regression_data):
    X, _ = simple_regression_data
    nn = NeuralNetwork(
        n_inputs=1,
        hidden_size=None,
        task="regression",
        random_state=0,
    )

    batch_X = X[:10]
    h_pre, h_act, logits, preds = nn._forward(batch_X)

    assert h_pre is None
    assert h_act is None
    assert logits.shape == (10, 1)
    assert preds.shape == (10, 1)
    # For regression, logits and preds are the same
    np.testing.assert_allclose(logits, preds)


# ----------------------------------------------------------------------
# Loss function tests
# ----------------------------------------------------------------------


def test_cross_entropy_loss_manual():
    nn = NeuralNetwork(
        n_inputs=2,
        hidden_size=None,
        task="multiclass_classification",
        n_classes=3,
        random_state=0,
    )

    probs = np.array([[0.1, 0.6, 0.3],
                      [0.7, 0.2, 0.1]])
    y = np.array([1, 0])

    loss = nn._cross_entropy_loss(probs, y)

    # Manual: -mean(log(p(correct)))
    correct = np.array([0.6, 0.7])
    expected = -np.mean(np.log(correct))
    np.testing.assert_allclose(loss, expected, rtol=1e-7, atol=1e-7)


def test_mse_loss_manual():
    nn = NeuralNetwork(
        n_inputs=1,
        hidden_size=None,
        task="regression",
        random_state=0,
    )

    y_hat = np.array([[1.0], [3.0]])
    y = np.array([0.0, 2.0])

    # diff = [[1], [1]] -> squared = [[1], [1]]
    # mean = 1, Loss = 0.5 * 1 = 0.5
    loss = nn._mse_loss(y_hat, y)
    np.testing.assert_allclose(loss, 0.5)


# ----------------------------------------------------------------------
# Training behavior tests
# ----------------------------------------------------------------------


def test_train_requires_numpy_array_for_X(simple_multiclass_data):
    X, y = simple_multiclass_data
    X_list = X.tolist()  # not a numpy array
    nn = NeuralNetwork(
        n_inputs=2,
        hidden_size=4,
        task="multiclass_classification",
        n_classes=3,
    )

    with pytest.raises(TypeError, match="X must be a NumPy array"):
        nn.train(X_list, y)


def test_train_decreases_loss_classification(simple_multiclass_data):
    X, y = simple_multiclass_data

    nn = NeuralNetwork(
        n_inputs=2,
        hidden_size=8,
        task="multiclass_classification",
        n_classes=3,
        learning_rate=0.1,
        batch_size=32,
        max_epochs=30,
        random_state=0,
    )

    nn.train(X, y)

    assert len(nn.loss_history_) == nn.max_epochs
    assert nn.loss_history_[0] >= nn.loss_history_[-1]


def test_train_decreases_loss_regression(simple_regression_data):
    X, y = simple_regression_data

    nn = NeuralNetwork(
        n_inputs=1,
        hidden_size=None,
        task="regression",
        learning_rate=0.05,
        batch_size=16,
        max_epochs=50,
        random_state=0,
    )

    nn.train(X, y)

    assert len(nn.loss_history_) == nn.max_epochs
    assert nn.loss_history_[0] >= nn.loss_history_[-1]


def test_train_with_large_batch_size(simple_multiclass_data):
    X, y = simple_multiclass_data

    nn = NeuralNetwork(
        n_inputs=2,
        hidden_size=4,
        task="multiclass_classification",
        n_classes=3,
        learning_rate=0.1,
        batch_size=10_000,  # larger than dataset -> full-batch
        max_epochs=5,
        random_state=0,
    )

    nn.train(X, y)
    assert len(nn.loss_history_) == 5  # one loss per epoch


def test_train_deterministic_with_same_random_state(simple_multiclass_data):
    X, y = simple_multiclass_data

    kwargs = dict(
        n_inputs=2,
        hidden_size=6,
        task="multiclass_classification",
        n_classes=3,
        learning_rate=0.1,
        batch_size=32,
        max_epochs=15,
        random_state=42,
    )

    nn1 = NeuralNetwork(**kwargs)
    nn2 = NeuralNetwork(**kwargs)

    nn1.train(X, y)
    nn2.train(X, y)

    np.testing.assert_allclose(nn1.W2_, nn2.W2_)
    np.testing.assert_allclose(nn1.b2_, nn2.b2_)
    if nn1.hidden_size is not None:
        np.testing.assert_allclose(nn1.W1_, nn2.W1_)
        np.testing.assert_allclose(nn1.b1_, nn2.b1_)
    np.testing.assert_allclose(nn1.loss_history_, nn2.loss_history_)


# ----------------------------------------------------------------------
# Predict / predict_proba / score – classification
# ----------------------------------------------------------------------


def test_predict_classification_labels_and_accuracy(simple_multiclass_data):
    X, y = simple_multiclass_data

    nn = NeuralNetwork(
        n_inputs=2,
        hidden_size=8,
        task="multiclass_classification",
        n_classes=3,
        learning_rate=0.1,
        batch_size=32,
        max_epochs=40,
        random_state=0,
    )

    nn.train(X, y)
    y_pred = nn.predict(X)

    # labels shape
    assert y_pred.shape == y.shape
    # labels are class indices
    assert y_pred.dtype == np.int64 or np.issubdtype(y_pred.dtype, np.integer)

    # accuracy should be better than random (1 / n_classes ~= 0.33)
    acc = nn.score(X, y)
    assert 0.0 <= acc <= 1.0
    assert acc > 1.0 / 3.0


def test_predict_proba_classification_sums_to_one(simple_multiclass_data):
    X, y = simple_multiclass_data
    nn = NeuralNetwork(
        n_inputs=2,
        hidden_size=None,
        task="multiclass_classification",
        n_classes=3,
        learning_rate=0.1,
        batch_size=16,
        max_epochs=30,
        random_state=0,
    )

    nn.train(X, y)
    probs = nn.predict_proba(X[:20])

    assert probs.shape == (20, 3)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(20), atol=1e-6)


def test_predict_proba_returns_copy_not_view(simple_multiclass_data):
    X, y = simple_multiclass_data
    nn = NeuralNetwork(
        n_inputs=2,
        hidden_size=4,
        task="multiclass_classification",
        n_classes=3,
        learning_rate=0.1,
        batch_size=16,
        max_epochs=10,
        random_state=0,
    )

    nn.train(X, y)

    p1 = nn.predict_proba(X[:5])
    p1[:] = 0.0
    p2 = nn.predict_proba(X[:5])

    # p2 should not be affected by modifying p1
    assert not np.allclose(p1, p2)


# ----------------------------------------------------------------------
# Predict / predict_proba / score – regression
# ----------------------------------------------------------------------


def test_predict_and_predict_proba_regression_shapes(simple_regression_data):
    X, y = simple_regression_data
    nn = NeuralNetwork(
        n_inputs=1,
        hidden_size=4,
        task="regression",
        learning_rate=0.05,
        batch_size=16,
        max_epochs=30,
        random_state=0,
    )

    nn.train(X, y)

    y_pred = nn.predict(X)
    y_proba = nn.predict_proba(X)

    # predict returns flat (n_samples,)
    assert y_pred.shape == (X.shape[0],)

    # predict_proba returns (n_samples, 1) for regression
    assert y_proba.shape == (X.shape[0], 1)


def test_predict_accepts_list_for_regression(simple_regression_data):
    X, y = simple_regression_data
    nn = NeuralNetwork(
        n_inputs=1,
        hidden_size=None,
        task="regression",
        learning_rate=0.05,
        batch_size=16,
        max_epochs=40,
        random_state=0,
    )

    nn.train(X, y)

    X_list = X.tolist()
    y_pred_list = nn.predict(X_list)
    y_pred_array = nn.predict(X)

    np.testing.assert_allclose(y_pred_list, y_pred_array, atol=1e-6)


def test_score_regression_negative_mse_behavior():
    """
    Manually set weights so that predictions match y in one case,
    and are far from y in another, and verify score ordering.
    """
    nn = NeuralNetwork(
        n_inputs=1,
        hidden_size=None,
        task="regression",
        random_state=0,
    )

    # Override randomly initialized params
    nn.W2_ = np.array([[1.0]])
    nn.b2_ = np.array([0.0])

    X = np.array([[0.0], [1.0], [2.0]])
    y_good = np.array([0.0, 1.0, 2.0])  # perfect fit
    y_bad = np.array([0.0, 10.0, 20.0])

    score_good = nn.score(X, y_good)  # -MSE ~ 0
    score_bad = nn.score(X, y_bad)    # large negative

    assert score_good > score_bad
    np.testing.assert_allclose(score_good, 0.0, atol=1e-8)


def test_train_regression_reduces_mse(simple_regression_data):
    """
    Verify that after training, the regression model actually fits the data
    with a small MSE.
    """
    X, y = simple_regression_data
    nn = NeuralNetwork(
        n_inputs=1,
        hidden_size=None,
        task="regression",
        learning_rate=0.05,
        batch_size=16,
        max_epochs=80,
        random_state=0,
    )

    nn.train(X, y)
    y_pred = nn.predict(X)
    mse = np.mean((y_pred - y) ** 2)

    assert mse < 1e-2
