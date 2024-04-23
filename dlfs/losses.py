import numpy as np


class SquaredLoss:
    def forward(self, y_true: np.ndarray, y_pred: np.ndarray):
        assert y_true.shape == y_pred.shape
        loss = 0.5 * np.power((y_true - y_pred), 2)
        return np.sum(loss) / y_pred.shape[0]

    def backward(self, y_true, y_pred):
        assert y_true.shape == y_pred.shape
        return (y_pred - y_true) / y_pred.shape[0]


class SigmoidBinaryCrossEntropyLoss:
    # https://www.youtube.com/watch?v=rf4WF-5y8uY&t=1s
    def predict(self, Z):
        y_pred = 1.0 / (1.0 + np.exp(-Z))
        y_pred = np.clip(y_pred, 1.0e-9, 1.0 - 1.0e-9)
        return y_pred

    def forward(self, y_true, Z):
        assert np.isin(y_true, [0, 1]).all() == True
        y_pred = 1.0 / (1.0 + np.exp(-Z))
        y_pred = np.clip(y_pred, 1.0e-9, 1.0 - 1.0e-9)
        loss = -y_true * np.log(y_pred) - (1.0 - y_true) * np.log(1.0 - y_pred)
        return loss

    def backward(self, y_true, Z):
        assert np.isin(y_true, [0, 1]).all() == True
        y_pred = 1.0 / (1.0 + np.exp(-Z))
        y_pred = np.clip(y_pred, 1.0e-9, 1.0 - 1.0e-9)
        return y_pred - y_true


class SoftmaxCrossEntropyLoss:
    # https://www.youtube.com/watch?v=rf4WF-5y8uY&t=1s
    def predict(self, Z):
        assert Z.ndim == 2
        Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
        Z_exp = np.exp(Z_shifted)
        y_pred = Z_exp / np.sum(Z_exp, axis=1, keepdims=True)
        return y_pred

    def forward(self, y_true, Z):   # y_true is one hot encoding
        assert Z.ndim == 2
        assert y_true.shape == Z.shape
        assert np.sum(y_true, axis=1).all() == 1  # one-hot encoding
        Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
        Z_exp = np.exp(Z_shifted)
        y_pred = Z_exp / np.sum(Z_exp, axis=1, keepdims=True)
        y_pred = np.clip(y_pred, 1.0e-9, 1.0 - 1.0e-9)
        loss = - y_true * np.log(y_pred) 
        return loss

    def backward(self, y_true, Z):
        assert Z.ndim == 2
        assert y_true.shape == Z.shape
        assert np.sum(y_true, axis=1).all() == 1  # one-hot encoding
        Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
        Z_exp = np.exp(Z_shifted)
        y_pred = Z_exp / np.sum(Z_exp, axis=1, keepdims=True)
        y_pred = np.clip(y_pred, 1.0e-9, 1.0 - 1.0e-9)
        return y_pred - y_true


if __name__ == "__main__":

    y = np.random.randn(3, 6)
    for row in y:
        idx = np.argmax(row)
        row[:] = 0
        row[idx] = 1
    # y = np.where(y > 0, 1, 0)
    y_hat = np.random.randn(3, 6)
    print(y)
    print(y_hat)

    loss_fn = SoftmaxCrossEntropyLoss()
    predct = loss_fn.predict(y_hat)
    loss = loss_fn.forward(y, y_hat)
    grad = loss_fn.backward(y, y_hat)
    print("p", predct)
    print("l", loss)
    print("grad", grad)
    print(predct - grad)
