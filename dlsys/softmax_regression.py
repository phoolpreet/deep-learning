import numpy as np
from matplotlib import pyplot as plt


def softmax_loss(Z: np.ndarray, y: np.ndarray):
    """Return softmax loss.
    Args:
        Z (np.ndarray[np.float32]): 2D numpy array of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (np.ndarray[np.uint8]): 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    # https://colab.research.google.com/github/dlsyscourse/hw0/blob/master/hw0.ipynb
    # https://www.youtube.com/watch?v=MlivXhZFbNA
    """

    batch_size = Z.shape[0]
    assert Z.ndim == 2
    assert y.ndim == 1
    assert y.shape[0] == batch_size
    z_y = Z[np.arange(batch_size), y]
    loss = np.log(np.sum(np.exp(Z), axis=1)) - z_y
    return loss.mean()


'''
Solves a classification problem
'''
class SoftmaxRegression:
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        num_classes: int,
        lr: float = 0.1,
        batch_size: int = 100,
    ):
        assert X.ndim == 2
        assert y.ndim == 1
        assert X.shape[0] == y.shape[0]
        num_samples = X.shape[0]
        num_feature = X.shape[1]
        W = np.random.randn(num_feature, num_classes)
        W /= W.max()

        for epoch in range(15):
            loss = softmax_loss(X @ W, y)
            print("loss", loss)
            for i in range(0, num_samples, batch_size):
                batch_x = X[i : min(i + batch_size, num_samples)]
                batch_y = y[i : min(i + batch_size, num_samples)]

                h_out = batch_x @ W                                 # linear hypothesis
                h_exp = np.exp(h_out)
                Z = h_exp / np.sum(h_exp, axis=1, keepdims=True)    # softmax
                y_one_hot = np.zeros((batch_y.shape[0], num_classes))
                y_one_hot[np.arange(batch_y.shape[0]), batch_y] = 1
                dZ = Z - y_one_hot

                dW = batch_x.T @ dZ
                dW /= batch_size
                W -= lr * dW


if __name__ == "__main__":

    # Z = np.random.randn(32, 10)
    # y = np.random.randint(low=0, high=9, size=(32))
    # loss = softmax_loss(Z, y)
    # print(loss)

    from mnist_data_parser import parse_mnist

    X_train, Y_train = parse_mnist(
        "data_mnist/train-images-idx3-ubyte/train-images.idx3-ubyte",
        "data_mnist/train-labels-idx1-ubyte/train-labels.idx1-ubyte",
    )
    print(X_train.shape)
    print(Y_train.shape)

    softmax_reg = SoftmaxRegression(X_train, Y_train, 10)
