import numpy as np
import pandas as pd
import pytest

from rice_ml.supervised_learning.linear_regression import SingleNeuron 


# ---------- Fixtures ----------
@pytest.fixture
def toy_df():
    """
    Build a tiny linear dataset with a single feature where
    y = 2*x + 1 (no noise) so the neuron can learn it easily.
    The feature column is stored as vectors to match prepare_data.
    """
    x = np.arange(0, 5, dtype=float)  # [0,1,2,3,4]
    y = 2 * x + 1
    df = pd.DataFrame(
        {
            "features": [np.array([xi]) for xi in x],
            "target": y,
        }
    )
    return df


@pytest.fixture
def multifeat_df():
    """
    Multi-feature linear dataset: y = 3*x1 - 2*x2 + 0.5
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 2))
    y = 3 * X[:, 0] - 2 * X[:, 1] + 0.5
    df = pd.DataFrame(
        {
            "features": [row for row in X],
            "target": y,
        }
    )
    return df


# ---------- Tests ----------
def test_init_and_attributes():
    model = SingleNeuron()
    assert hasattr(model, "activation_function")
    assert model.vector_col is None
    assert model.target_col is None


def test_prepare_data_shapes(toy_df):
    model = SingleNeuron()
    model.vector_col = "features"
    model.target_col = "target"
    X, y = model.prepare_data(toy_df)

    assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray)
    assert X.shape == (len(toy_df), 1)
    assert y.shape == (len(toy_df),)


def test_prepare_data_rejects_non_numeric_target(toy_df):
    df = toy_df.copy()
    df["target"] = df["target"].astype(str)  # make it non-numeric

    model = SingleNeuron()
    model.vector_col = "features"
    model.target_col = "target"

    with pytest.raises(ValueError, match="requires numeric target"):
        _ = model.prepare_data(df)


def test_training_reduces_error(toy_df):
    model = SingleNeuron()
    model.vector_col = "features"
    model.target_col = "target"
    X, y = model.prepare_data(toy_df)

    model.train(X, y, alpha=0.05, epochs=50, random_state=42)

    # errors_ should exist, match epochs, and generally decrease
    assert hasattr(model, "errors_")
    assert len(model.errors_) == 50
    assert model.errors_[0] > model.errors_[-1]


def test_predict_single_and_batch(toy_df):
    model = SingleNeuron()
    model.vector_col = "features"
    model.target_col = "target"
    X, y = model.prepare_data(toy_df)

    model.train(X, y, alpha=0.05, epochs=100, random_state=0)

    # single sample (1D)
    pred_single = model.predict(X[0])
    assert np.isscalar(pred_single) or np.shape(pred_single) == ()

    # batch (2D)
    pred_batch = model.predict(X)
    assert isinstance(pred_batch, np.ndarray)
    assert pred_batch.shape == (X.shape[0],)


def test_learns_simple_linear_relation(toy_df):
    model = SingleNeuron()
    model.vector_col = "features"
    model.target_col = "target"
    X, y = model.prepare_data(toy_df)

    model.train(X, y, alpha=0.1, epochs=300, random_state=123)

    preds = model.predict(X)
    mse = np.mean((preds - y) ** 2)
    assert mse < 1e-3  # should be very small on noise-free line


def test_reproducibility_with_random_state(multifeat_df):
    model1 = SingleNeuron()
    model1.vector_col = "features"
    model1.target_col = "target"
    X, y = model1.prepare_data(multifeat_df)

    model2 = SingleNeuron()
    model2.vector_col = "features"
    model2.target_col = "target"

    # Same seed -> identical training trajectory & weights
    model1.train(X, y, alpha=0.05, epochs=80, random_state=7)
    model2.train(X, y, alpha=0.05, epochs=80, random_state=7)

    np.testing.assert_allclose(model1.w_, model2.w_, rtol=0, atol=1e-10)
    np.testing.assert_allclose(model1.errors_, model2.errors_, rtol=0, atol=1e-12)


def test_custom_activation_is_used(multifeat_df):
    # Use a clipped (piecewise) activation to ensure it's actually applied
    def clip_activation(z):
        return np.minimum(np.maximum(z, -1.0), 1.0)

    model = SingleNeuron(activation_function=clip_activation)
    model.vector_col = "features"
    model.target_col = "target"
    X, y = model.prepare_data(multifeat_df)

    model.train(X, y, alpha=0.05, epochs=10, random_state=0)

    # Predictions must fall within the clip range
    preds = model.predict(X)
    assert np.all(preds <= 1.0) and np.all(preds >= -1.0)


def test_weights_change_after_training(multifeat_df):
    model = SingleNeuron()
    model.vector_col = "features"
    model.target_col = "target"
    X, y = model.prepare_data(multifeat_df)

    # Initialize by calling train for 0 epochs to materialize w_ (simulate manual init)
    # Since the class initializes in train, we capture initial weights right after a single step.
    model.train(X, y, alpha=0.0, epochs=1, random_state=99)
    w_init = model.w_.copy()

    # Train with nonzero learning rate and more epochs
    model.train(X, y, alpha=0.05, epochs=5, random_state=99)
    assert not np.allclose(model.w_, w_init)
