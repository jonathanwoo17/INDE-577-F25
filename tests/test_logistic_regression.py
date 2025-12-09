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

from rice_ml.supervised_learning.logistic_regression import LogisticRegression


# Fixtures
@pytest.fixture
def linearly_separable_numeric_df():
    """
    Simple 1D linearly separable dataset with numeric labels {0, 1}.

    x: -2, -1 -> class 0
       +1, +2 -> class 1
    """
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([0, 0, 1, 1])
    df = pd.DataFrame({"x": list(X), "y": y})
    return df


@pytest.fixture
def linearly_separable_string_df():
    """
    Same geometry as above but with string labels {"neg", "pos"}.
    """
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array(["neg", "neg", "pos", "pos"])
    df = pd.DataFrame({"x": list(X), "y": y})
    return df


# prepare_data tests
def test_prepare_data_numeric_labels(linearly_separable_numeric_df):
    df = linearly_separable_numeric_df
    model = LogisticRegression()

    X, y = model.prepare_data(df)

    # Shapes
    assert X.shape == (4, 1)
    assert y.shape == (4,)

    # Label mapping for numeric labels
    assert model._label_map_ == {0: 0, 1: 1}
    assert model._inv_label_map_ == {0: 0, 1: 1}

    # y is in {0, 1}
    assert set(np.unique(y)) == {0.0, 1.0}


def test_prepare_data_string_labels(linearly_separable_string_df):
    df = linearly_separable_string_df
    model = LogisticRegression()

    X, y = model.prepare_data(df)

    assert X.shape == (4, 1)
    assert y.shape == (4,)

    # Categories are sorted by np.unique
    # For ["neg", "pos"], categories = ["neg", "pos"]
    assert model._label_map_ == {"neg": 0, "pos": 1}
    assert model._inv_label_map_ == {0: "neg", 1: "pos"}

    assert set(np.unique(y)) == {0.0, 1.0}


def test_prepare_data_raises_non_binary_labels():
    df = pd.DataFrame(
        {
            "x": [np.array([0.0]), np.array([1.0]), np.array([2.0])],
            "y": [0, 1, 2],
        }
    )
    model = LogisticRegression()

    with pytest.raises(ValueError, match="only supports binary"):
        _ = model.prepare_data(df)


def test_prepare_data_raises_single_class():
    df = pd.DataFrame(
        {"x": [np.array([0.0]), np.array([1.0]), np.array([2.0])], "y": [1, 1, 1]}
    )
    model = LogisticRegression()

    with pytest.raises(ValueError, match="only supports binary"):
        _ = model.prepare_data(df)


# Sigmoid tests
def test_sigmoid_basic_properties():
    # Scalar
    assert np.isclose(LogisticRegression._sigmoid(0.0), 0.5)

    # Vector input
    z = np.array([-1.0, 0.0, 1.0])
    s = LogisticRegression._sigmoid(z)

    assert s.shape == z.shape
    assert np.all((s > 0.0) & (s < 1.0))

    # Monotonicity: s(-1) < s(0) < s(1)
    assert s[0] < s[1] < s[2]


# fit / training behavior
def test_fit_initializes_params_and_loss_decreases(linearly_separable_numeric_df):
    df = linearly_separable_numeric_df
    model = LogisticRegression(
        learning_rate=0.1,
        n_epochs=50,
        random_state=0,
    )

    model.fit(df)

    # Parameters initialized
    assert model.w_ is not None
    assert isinstance(model.b_, float)
    assert model.w_.shape == (1,)

    # Loss is tracked over epochs
    assert len(model.losses_) == model.n_epochs

    # Final loss is lower than initial loss
    assert model.losses_[0] >= model.losses_[-1]


def test_fit_is_deterministic_with_same_random_state(linearly_separable_numeric_df):
    df = linearly_separable_numeric_df

    kwargs = dict(learning_rate=0.1, n_epochs=100, random_state=42)

    model1 = LogisticRegression(**kwargs).fit(df)
    model2 = LogisticRegression(**kwargs).fit(df)

    np.testing.assert_allclose(model1.w_, model2.w_)
    np.testing.assert_allclose(model1.b_, model2.b_)
    np.testing.assert_allclose(model1.losses_, model2.losses_)


def test_fit_achieves_high_accuracy_on_simple_data(linearly_separable_numeric_df):
    df = linearly_separable_numeric_df

    model = LogisticRegression(
        learning_rate=0.1,
        n_epochs=2000,
        random_state=0,
    )
    model.fit(df)

    acc = model.score(df)
    assert acc == pytest.approx(1.0, abs=1e-6)


# predict_proba tests
def test_predict_proba_raises_if_not_fitted(linearly_separable_numeric_df):
    df = linearly_separable_numeric_df
    model = LogisticRegression()

    with pytest.raises(RuntimeError, match="trained"):
        _ = model.predict_proba(df)


def test_predict_proba_shape_and_bounds(linearly_separable_string_df):
    df = linearly_separable_string_df
    model = LogisticRegression(
        learning_rate=0.1,
        n_epochs=2000,
        random_state=0,
    )
    model.fit(df)

    probs = model.predict_proba(df)

    assert probs.shape == (len(df),)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)


# predict tests
def test_predict_returns_original_label_types(linearly_separable_string_df):
    df = linearly_separable_string_df
    model = LogisticRegression(
        learning_rate=0.1,
        n_epochs=2000,
        random_state=0,
    )
    model.fit(df)

    preds = model.predict(df)

    # Should be strings "neg"/"pos", not numeric
    assert set(np.unique(preds)) == {"neg", "pos"}

    # On this easy dataset, accuracy should be perfect
    assert model.score(df) == pytest.approx(1.0, abs=1e-6)


def test_predict_respects_threshold():
    """
    Manually set parameters so that p = 0.5 for all examples.
    Then vary threshold and check predicted labels flip.
    """
    model = LogisticRegression()
    # Manually create a "trained" model
    model.w_ = np.array([0.0])
    model.b_ = 0.0
    model._inv_label_map_ = {0: "neg", 1: "pos"}

    df = pd.DataFrame({"x": [np.array([0.0]), np.array([0.0])]})

    preds_default = model.predict(df, threshold=0.5)
    preds_high = model.predict(df, threshold=0.9)

    # p = 0.5 -> >= 0.5 treated as positive
    assert np.all(preds_default == "pos")
    # p = 0.5 < 0.9 -> treated as negative
    assert np.all(preds_high == "neg")


# score tests
def test_score_computes_accuracy(linearly_separable_numeric_df):
    df = linearly_separable_numeric_df
    model = LogisticRegression(
        learning_rate=0.1,
        n_epochs=2000,
        random_state=0,
    )
    model.fit(df)

    acc = model.score(df)
    assert 0.0 <= acc <= 1.0
    assert acc == pytest.approx(1.0, abs=1e-6)


# loss tests
def test_loss_matches_manual_computation(linearly_separable_string_df):
    df = linearly_separable_string_df
    model = LogisticRegression(
        learning_rate=0.1,
        n_epochs=500,
        random_state=0,
    )
    model.fit(df)

    # Model's loss method
    model_loss = model.loss(df)

    # Manual computation from predicted probabilities and numeric labels
    X, y_numeric = model.prepare_data(df)  # re-uses same label mapping
    probs = model.predict_proba(df)

    eps = 1e-12
    manual_loss = -np.mean(
        y_numeric * np.log(probs + eps) + (1.0 - y_numeric) * np.log(1.0 - probs + eps)
    )

    np.testing.assert_allclose(model_loss, manual_loss, rtol=1e-7, atol=1e-7)


def test_loss_raises_if_not_fitted(linearly_separable_numeric_df):
    df = linearly_separable_numeric_df
    model = LogisticRegression()

    with pytest.raises(RuntimeError, match="trained"):
        _ = model.loss(df)


# pre_activation tests
def test_pre_activation_raises_if_not_trained():
    model = LogisticRegression()
    X = np.array([[1.0], [2.0]])

    with pytest.raises(RuntimeError):
        _ = model.pre_activation(X)


def test_pre_activation_computes_linear_term():
    model = LogisticRegression()
    # Pretend we trained:
    model.w_ = np.array([2.0, -1.0])
    model.b_ = 0.5

    X = np.array(
        [
            [1.0, 0.0],  # z = 2*1 + -1*0 + 0.5 = 2.5
            [0.0, 1.0],  # z = 2*0 + -1*1 + 0.5 = -0.5
        ]
    )

    z = model.pre_activation(X)
    expected = X @ model.w_ + model.b_

    np.testing.assert_allclose(z, expected)


# Column name customization tests
def test_custom_vector_and_label_columns():
    df = pd.DataFrame(
        {
            "features": [np.array([-1.0]), np.array([1.0])],
            "target": ["neg", "pos"],
        }
    )

    model = LogisticRegression(
        learning_rate=0.1,
        n_epochs=500,
        vector_col="features",
        label_col="target",
        random_state=0,
    )

    model.fit(df)

    # Basic sanity checks
    probs = model.predict_proba(df)
    preds = model.predict(df)

    assert probs.shape == (2,)
    assert set(preds) == {"neg", "pos"}
    assert model.score(df) == pytest.approx(1.0, abs=1e-6)
