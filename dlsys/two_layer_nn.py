import numpy as np
from softmax_regression import softmax_loss


def sigmoid(X: np.ndarray):
    return 1.0 / (1.0 + np.exp(-X))


def sigmoid_grad(X: np.ndarray):
    S = sigmoid(X)
    return S * (1.0 - S)


def softmax(Z: np.ndarray):
    S = Z - Z.max()
    S = np.exp(S)
    S /= S.sum(axis=1, keepdims=True)
    return S


"""
Solves a classification problem
"""


class TwoLayerNN:
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        num_hidden_units: int,
        num_classes: int,
        lr: float = 0.1,
        batch_size: int = 100,
    ):
        assert X.ndim == 2
        assert y.ndim == 1
        assert X.shape[0] == y.shape[0]
        num_samples = X.shape[0]
        num_features = X.shape[1]

        W1 = np.random.randn(num_features, num_hidden_units)
        W2 = np.random.randn(num_hidden_units, num_classes)

        for epoch in range(5):
            for i in range(0, num_samples, batch_size):
                batch_x = X[i : min(i + batch_size, num_samples)]
                batch_y = y[i : min(i + batch_size, num_samples)]

                # forward
                O1 = batch_x @ W1
                O2 = sigmoid(O1)
                O3 = O2 @ W2  # hypothesis

                S = softmax(O3)
                y_one_hot = np.zeros((batch_y.shape[0], num_classes))
                y_one_hot[np.arange(batch_y.shape[0]), batch_y] = 1

                dW2 = O2.T @ (S - y_one_hot)
                dW1 = batch_x.T @ (sigmoid_grad(O1) * ((S - y_one_hot) @ W2.T))
                dW1 /= batch_size
                dW2 /= batch_size
                W1 -= lr * dW1
                W2 -= lr * dW2

                loss = softmax_loss(O3, batch_y)
                print("Loss:", loss)


if __name__ == "__main__":

    from mnist_data_parser import parse_mnist

    X_train, Y_train = parse_mnist(
        "data_mnist/train-images-idx3-ubyte/train-images.idx3-ubyte",
        "data_mnist/train-labels-idx1-ubyte/train-labels.idx1-ubyte",
    )
    print(X_train.shape)
    print(Y_train.shape)

    # X_train = np.random.randn(10, 4)
    # Y_train = np.random.randint(0, 9, size=(10))
    nn = TwoLayerNN(X_train, Y_train, num_hidden_units=400, num_classes=10)
