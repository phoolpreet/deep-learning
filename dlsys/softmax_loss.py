import numpy as np


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
    """
    
    batch_size = Z.shape[0]
    assert Z.ndim == 2
    assert y.ndim == 1
    assert y.shape[0] == batch_size

    z_nrmlzd = Z - Z.max(axis=1, keepdims=True)
    z_exp = np.exp(z_nrmlzd)
    softmax = z_exp / z_exp.sum(axis=1, keepdims=True)
    predicted_probs = softmax[np.indices(y.shape), y]
    log_probs = np.log(predicted_probs)
    return - log_probs.sum() / batch_size



if __name__ == "__main__":

    Z = np.random.randn(32, 10)
    y = np.random.random_integers(low=0, high=9, size=(32))
    loss = softmax_loss(Z, y)
    print(loss)