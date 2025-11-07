import numpy as np
from typing import Optional, Tuple, List, Literal


def _relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)


def _relu_grad(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(z.dtype)


def _softmax(z: np.ndarray) -> np.ndarray:
    """
    Numerically stable softmax.
    z: (batch, n_classes)
    returns probs: (batch, n_classes)
    """
    z_shift = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shift)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


class NeuralNetwork:
    """
    Minimal neural network trained with mini-batch SGD.

    Supports:
    - multiclass classification with softmax + cross-entropy
    - regression with linear output + mean squared error

    Architecture:
    - optional single hidden layer with ReLU
    - output layer:
        * softmax over n_classes  (classification)
        * linear 1-D output       (regression)

    This class assumes you've already done any preprocessing like:
    - flattening images (28x28 -> length 784),
    - scaling / normalization,
    - extracting labels as ints or floats.
    That should live in preprocess.py.

    Parameters
    ----------
    n_inputs : int
        Number of input features per sample.
        e.g. 784 for flattened 28x28 Fashion-MNIST images.

    hidden_size : Optional[int], default=None
        If not None (e.g. 128), the model is:
            input -> ReLU(hidden_size) -> output
        If None, the model is:
            input -> output
        where "output" is either softmax layer or linear layer.

    task : {'multiclass_classification', 'regression'}
        Training objective / output behavior.
        - 'multiclass_classification': softmax outputs with cross-entropy loss.
          Assumes y are integer class labels 0..(n_classes-1).
        - 'regression': linear output (no activation) with MSE loss.
          Assumes y are continuous floats.

    n_classes : Optional[int], default=None
        Required if task == 'multiclass_classification'.
        Number of classes.
        Ignored if task == 'regression'.

    learning_rate : float, default=0.01
        SGD step size.

    batch_size : int, default=32
        Mini-batch size. If >= n_samples, behaves like full-batch.

    max_epochs : int, default=10
        Number of passes over the dataset.

    shuffle : bool, default=True
        Shuffle data at each epoch start.

    random_state : Optional[int], default=None
        Seed for reproducibility.

    Attributes
    ----------
    W1_ : np.ndarray or None
        Weights for input -> hidden (shape (n_inputs, hidden_size)) if hidden layer exists,
        else None.

    b1_ : np.ndarray or None
        Bias for hidden layer (shape (hidden_size,)) if hidden layer exists,
        else None.

    W2_ : np.ndarray
        Weights for final layer.
        Shape:
            (hidden_size, n_outputs) if hidden_size is not None
            (n_inputs,    n_outputs) if hidden_size is None
        where n_outputs is:
            n_classes (classification)
            1         (regression)

    b2_ : np.ndarray
        Bias for final layer, shape (n_outputs,).

    loss_history_ : list[float]
        Mean loss per epoch across batches.
    """

    def __init__(
        self,
        n_inputs: int,
        hidden_size: Optional[int],
        task: Literal["multiclass_classification", "regression"],
        n_classes: Optional[int] = None,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        max_epochs: int = 10,
        shuffle: bool = True,
        random_state: Optional[int] = None,
    ):
        self.n_inputs = n_inputs
        self.hidden_size = hidden_size
        self.task = task

        # for classification we need to know how many classes
        if task == "multiclass_classification":
            if n_classes is None:
                raise ValueError("n_classes must be provided for multiclass_classification.")
            if n_classes < 2:
                raise ValueError("n_classes must be >= 2 for multiclass_classification.")
            self.n_outputs = n_classes
        else:
            # regression -> 1 continuous output
            self.n_outputs = 1

        self.n_classes = n_classes  # may be None if regression

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.shuffle = shuffle

        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

        # parameters (set in _init_params)
        self.W1_: Optional[np.ndarray] = None
        self.b1_: Optional[np.ndarray] = None
        self.W2_: np.ndarray = None  # type: ignore
        self.b2_: np.ndarray = None  # type: ignore

        self.loss_history_: List[float] = []

        self._init_params()

    # ------------------------------------------------------------------
    # initialization
    # ------------------------------------------------------------------

    def _init_params(self) -> None:
        """
        Initialize model parameters with small random weights.
        He-style init for hidden ReLU layer.
        """
        if self.hidden_size is not None:
            # hidden layer
            self.W1_ = self._rng.normal(
                loc=0.0,
                scale=np.sqrt(2.0 / self.n_inputs),  # He init for ReLU
                size=(self.n_inputs, self.hidden_size),
            )
            self.b1_ = np.zeros(self.hidden_size)

            # output layer
            self.W2_ = self._rng.normal(
                loc=0.0,
                scale=np.sqrt(1.0 / self.hidden_size),
                size=(self.hidden_size, self.n_outputs),
            )
            self.b2_ = np.zeros(self.n_outputs)

        else:
            # no hidden layer, direct input -> output
            self.W2_ = self._rng.normal(
                loc=0.0,
                scale=np.sqrt(1.0 / self.n_inputs),
                size=(self.n_inputs, self.n_outputs),
            )
            self.b2_ = np.zeros(self.n_outputs)

    # ------------------------------------------------------------------
    # forward pass
    # ------------------------------------------------------------------

    def _forward(
        self, X: np.ndarray
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray, np.ndarray]:
        """
        Forward pass.

        Parameters
        ----------
        X : np.ndarray, shape (batch, n_inputs)

        Returns
        -------
        h_pre : np.ndarray or None
            Pre-activation of hidden layer (batch, hidden_size) or None.
        h_act : np.ndarray or None
            ReLU activation of hidden layer (batch, hidden_size) or None.
        logits_or_output : np.ndarray
            For classification:
                raw unnormalized scores (logits), shape (batch, n_classes)
            For regression:
                raw linear output, shape (batch, 1)
        probs_or_pred : np.ndarray
            For classification:
                softmax probabilities, shape (batch, n_classes)
            For regression:
                same as logits_or_output (just linear output).
        """
        if self.hidden_size is not None:
            h_pre = X @ self.W1_ + self.b1_         # (batch, hidden_size)
            h_act = _relu(h_pre)                    # (batch, hidden_size)
            logits = h_act @ self.W2_ + self.b2_    # (batch, n_outputs)
        else:
            h_pre = None
            h_act = None
            logits = X @ self.W2_ + self.b2_        # (batch, n_outputs)

        if self.task == "multiclass_classification":
            probs = _softmax(logits)                # (batch, n_classes)
            return h_pre, h_act, logits, probs
        else:
            # regression -> linear output is already the prediction
            return h_pre, h_act, logits, logits

    # ------------------------------------------------------------------
    # losses
    # ------------------------------------------------------------------

    def _cross_entropy_loss(self, probs: np.ndarray, y: np.ndarray) -> float:
        """
        Multiclass cross-entropy (mean over batch).
        probs: (batch, n_classes)
        y:     (batch,), int labels in [0, n_classes-1]
        """
        batch_size = y.shape[0]
        correct = probs[np.arange(batch_size), y]
        correct = np.clip(correct, 1e-12, 1.0)
        return float(-np.mean(np.log(correct)))

    def _mse_loss(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        """
        Mean squared error, with 0.5 scaling to match gradient style.
        y_hat: (batch, 1)
        y:     (batch,) or (batch, 1)
        """
        y = y.reshape(-1, 1)
        diff = y_hat - y
        return float(0.5 * np.mean(diff ** 2))

    # ------------------------------------------------------------------
    # backward pass (gradients)
    # ------------------------------------------------------------------

    def _backward_classification(
        self,
        X: np.ndarray,
        y: np.ndarray,
        h_pre: Optional[np.ndarray],
        h_act: Optional[np.ndarray],
        probs: np.ndarray,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray, np.ndarray]:
        """
        Backprop for softmax + cross-entropy.
        """
        batch_size = X.shape[0]

        # dL/dlogits = (probs - y_onehot) / batch_size
        y_onehot = np.zeros_like(probs)
        y_onehot[np.arange(batch_size), y] = 1.0
        dlogits = (probs - y_onehot) / batch_size  # (batch, n_classes)

        if self.hidden_size is not None:
            # gradients for output layer
            dW2 = h_act.T @ dlogits                        # (hidden_size, n_classes)
            db2 = np.sum(dlogits, axis=0)                  # (n_classes,)

            # backprop to hidden layer
            dh = dlogits @ self.W2_.T                      # (batch, hidden_size)
            dh_pre = dh * _relu_grad(h_pre)                # (batch, hidden_size)

            dW1 = X.T @ dh_pre                             # (n_inputs, hidden_size)
            db1 = np.sum(dh_pre, axis=0)                   # (hidden_size,)
        else:
            # no hidden layer
            dW2 = X.T @ dlogits                            # (n_inputs, n_classes)
            db2 = np.sum(dlogits, axis=0)                  # (n_classes,)
            dW1 = None
            db1 = None

        return dW1, db1, dW2, db2

    def _backward_regression(
        self,
        X: np.ndarray,
        y: np.ndarray,
        h_pre: Optional[np.ndarray],
        h_act: Optional[np.ndarray],
        y_hat: np.ndarray,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray, np.ndarray]:
        """
        Backprop for linear output + MSE.
        Loss = 0.5 * mean((y_hat - y)^2)
        dL/dy_hat = (y_hat - y) / batch_size
        """
        batch_size = X.shape[0]
        y = y.reshape(-1, 1)
        dyhat = (y_hat - y) / batch_size   # (batch, 1)

        if self.hidden_size is not None:
            # output layer
            dW2 = h_act.T @ dyhat                # (hidden_size, 1)
            db2 = np.sum(dyhat, axis=0)          # (1,)

            # backprop to hidden
            dh = dyhat @ self.W2_.T              # (batch, hidden_size)
            dh_pre = dh * _relu_grad(h_pre)      # (batch, hidden_size)

            dW1 = X.T @ dh_pre                   # (n_inputs, hidden_size)
            db1 = np.sum(dh_pre, axis=0)         # (hidden_size,)
        else:
            dW2 = X.T @ dyhat                    # (n_inputs, 1)
            db2 = np.sum(dyhat, axis=0)          # (1,)
            dW1 = None
            db1 = None

        return dW1, db1, dW2, db2

    # ------------------------------------------------------------------
    # parameter update
    # ------------------------------------------------------------------

    def _apply_gradients(
        self,
        dW1: Optional[np.ndarray],
        db1: Optional[np.ndarray],
        dW2: np.ndarray,
        db2: np.ndarray,
    ) -> None:
        """
        Single SGD update step.
        """
        if self.hidden_size is not None:
            self.W1_ -= self.learning_rate * dW1
            self.b1_ -= self.learning_rate * db1

        self.W2_ -= self.learning_rate * dW2
        self.b2_ -= self.learning_rate * db2

    # ------------------------------------------------------------------
    # public API (mirrors your package style)
    # ------------------------------------------------------------------

    def train(self, X: np.ndarray, y: np.ndarray) -> "NeuralNetwork":
        """
        Train the network using mini-batch SGD.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_inputs)
            Preprocessed input features. For images, this should be flattened.

        y : np.ndarray
            - If task='multiclass_classification':
                shape (n_samples,), integer class labels 0..n_classes-1
            - If task='regression':
                shape (n_samples,), continuous targets

        Returns
        -------
        self : NeuralNetwork
        """
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a NumPy array.")
        if not isinstance(y, np.ndarray):
            y = np.asarray(y)

        n_samples = X.shape[0]
        batch_size = min(self.batch_size, n_samples)

        self.loss_history_ = []

        for _epoch in range(self.max_epochs):
            # shuffle indices
            if self.shuffle:
                idx = self._rng.permutation(n_samples)
            else:
                idx = np.arange(n_samples)

            epoch_losses: List[float] = []

            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                batch_idx = idx[start:end]

                Xb = X[batch_idx]
                yb = y[batch_idx]

                # forward
                h_pre, h_act, logits_or_output, probs_or_pred = self._forward(Xb)

                # loss
                if self.task == "multiclass_classification":
                    loss = self._cross_entropy_loss(probs_or_pred, yb)
                    dW1, db1, dW2, db2 = self._backward_classification(
                        Xb, yb, h_pre, h_act, probs_or_pred
                    )
                else:
                    loss = self._mse_loss(probs_or_pred, yb)
                    dW1, db1, dW2, db2 = self._backward_regression(
                        Xb, yb, h_pre, h_act, probs_or_pred
                    )

                epoch_losses.append(loss)

                # update
                self._apply_gradients(dW1, db1, dW2, db2)

            # mean loss for epoch
            self.loss_history_.append(float(np.mean(epoch_losses)))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict outputs for new samples.

        Returns
        -------
        y_hat : np.ndarray
            - If task='multiclass_classification':
                hard class labels, shape (n_samples,)
            - If task='regression':
                continuous predictions, shape (n_samples,)
        """
        if not isinstance(X, np.ndarray):
            X = np.asarray(X, dtype=float)

        _, _, _, probs_or_pred = self._forward(X)

        if self.task == "multiclass_classification":
            return np.argmax(probs_or_pred, axis=1)

        # regression
        return probs_or_pred.reshape(-1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        For classification:
            returns class probabilities, shape (n_samples, n_classes).
        For regression:
            returns continuous predictions with shape (n_samples, 1)
            (mainly for debugging / consistency).
        """
        if not isinstance(X, np.ndarray):
            X = np.asarray(X, dtype=float)

        _, _, _, probs_or_pred = self._forward(X)
        return probs_or_pred.copy()

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluation metric.

        Returns
        -------
        score : float
            - If task='multiclass_classification': accuracy in [0,1]
            - If task='regression': negative MSE (higher is better)
        """
        if not isinstance(X, np.ndarray):
            X = np.asarray(X, dtype=float)
        if not isinstance(y, np.ndarray):
            y = np.asarray(y)

        y_pred = self.predict(X)

        if self.task == "multiclass_classification":
            return float((y_pred == y).mean())

        # regression
        y = y.reshape(-1)
        y_pred = y_pred.reshape(-1)
        mse = np.mean((y_pred - y) ** 2)
        return float(-mse)
