import sys
from pathlib import Path

# Start at the current directory
root = Path().resolve()

while not (root / "src" / "rice_ml").exists() and root != root.parent:
    root = root.parent

sys.path.append(str(root / "src"))

import numpy as np
import pandas as pd
import pytest

from rice_ml.supervised_learning.perceptron import Perceptron


# -------------------------
# Helpers / Fixtures
# -------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)

@pytest.fixture
def separable_df(rng):
    """
    Linearly separable 2D dataset formatted exactly for your class:
    DataFrame with columns:
      - x: numpy array per row (shape (2,))
      - y: string label (two classes)
    """
    n = 100
    cov = np.array([[0.15, 0.0], [0.0, 0.15]])

    # Negative class around (-2, -2), label "A"
    X_neg = rng.multivariate_normal([-2.0, -2.0], cov, size=n)
    y_neg = ["A"] * n

    # Positive class around (+2, +2), label "B"
    X_pos = rng.multivariate_normal([+2.0, +2.0], cov, size=n)
    y_pos = ["B"] * n

    xs = [row for row in X_neg] + [row for row in X_pos]
    ys = y_neg + y_pos
    df = pd.DataFrame({"x": xs, "y": ys})
    # Shuffle rows for realism; order will be fixed inside train()
    return df.sample(frac=1.0, random_state=7).reset_index(drop=True)

@pytest.fixture
def three_class_df():
    xs = [np.array([0.0, 0.0]), np.array([1.0, 1.0]), np.array([2.0, 2.0])]
    ys = ["A", "B", "C"]
    return pd.DataFrame({"x": xs, "y": ys})


# ---------- Tests ----------

def test_prepare_data_binary_and_label_maps(separable_df):
    p = Perceptron()
    X, y = p.prepare_data(separable_df)

    # Shapes and types
    assert X.shape == (len(separable_df), 2)
    assert set(np.unique(y)) == {-1, 1}

    # Inverse mapping should recover labels
    recovered = np.vectorize(p._inv_label_map_.get)(y)
    assert set(recovered) == set(separable_df["y"].unique())

def test_prepare_data_raises_on_non_binary(three_class_df):
    p = Perceptron()
    with pytest.raises(ValueError):
        p.prepare_data(three_class_df)

def test_pre_activation_shape_after_training(separable_df):
    # Set a seed so the tiny random init in train() is reproducible
    np.random.seed(0)
    p = Perceptron(iterations=20)
    p.train(separable_df)

    X = np.vstack(separable_df["x"].to_numpy())
    margins = p.pre_activation(X)
    assert margins.shape == (len(separable_df),)

def test_train_achieves_high_accuracy_on_separable_data(separable_df):
    np.random.seed(0)
    p = Perceptron(learning_rate=1.0, iterations=50)
    p.train(separable_df)

    acc = p.score(separable_df)
    assert acc >= 0.95

def test_predict_returns_original_string_labels(separable_df):
    np.random.seed(0)
    p = Perceptron(iterations=30)
    p.train(separable_df)

    preds = p.predict(separable_df)
    assert isinstance(preds[0], str)
    assert set(np.unique(preds)) == set(separable_df["y"].unique())

def test_score_matches_manual_accuracy(separable_df):
    np.random.seed(0)
    p = Perceptron(iterations=30)
    p.train(separable_df)

    y_true = separable_df["y"].to_numpy()
    y_pred = p.predict(separable_df)
    manual = (y_true == y_pred).mean()
    assert abs(p.score(separable_df) - manual) < 1e-12

def test_loss_counts_number_of_misclassifications(separable_df):
    """
    Your loss() returns 0.25 * sum((y_hat - y_true_pm1)^2),
    which equals the integer count of misclassified samples.
    """
    np.random.seed(0)
    p = Perceptron(iterations=40)
    p.train(separable_df)

    base_loss = p.loss(separable_df)

    # Force one certain mistake by flipping the first row's label
    df_bad = separable_df.copy()
    all_labels = list(separable_df["y"].unique())
    first_label = df_bad.loc[0, "y"]
    alt_label = all_labels[1] if first_label == all_labels[0] else all_labels[0]
    df_bad.loc[0, "y"] = alt_label

    new_loss = p.loss(df_bad)
    assert new_loss >= base_loss
    assert (new_loss - base_loss) >= 1.0  # at least one extra misclassification

def test_learning_rate_scale_invariance_given_fixed_order_and_tiny_random_init(separable_df):
    """
    With fixed update order and extremely small random init (1e-6),
    scaling the learning rate usually scales (w,b) but preserves the sign
    of w·x + b for most points → identical predictions.
    """
    np.random.seed(0)
    p1 = Perceptron(learning_rate=1.0, iterations=30)
    p1.train(separable_df)
    pred1 = p1.predict(separable_df)

    np.random.seed(0)  # same init again
    p2 = Perceptron(learning_rate=0.1, iterations=30)
    p2.train(separable_df)
    pred2 = p2.predict(separable_df)

    assert np.array_equal(pred1, pred2)
