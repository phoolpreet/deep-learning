import numpy as np
from gate import MultiplyGate
from gate import AddGate
from optimizers import Optimizer
from optimizers import Adam


class Conv2D:
    """
    2D convolutional Layer
    Parameters:
    ----------
    n_filters: int. number of filters to be convolved with input image
    filter_shape: tuple. (filter_height, fiter_width)
    padding: string. Either 'same' or 'valid'
    stride: int. stride length of filter during convolution
    """

    def __init__(
        self,
        n_filters: int,
        filter_shape: tuple,
        padding: str = "same",
        stride: int = 1,
        optimizer: Optimizer = Adam,
        is_trainable: bool = True,
    ):
        assert n_filters > 0
        assert len(filter_shape) == 2
        assert filter_shape[0] > 0 and filter_shape[1] > 0
        assert padding in ["same", "valid"]
        assert stride > 0
        self.first = True
        self.n_filters = n_filters
        self.filter_height = filter_shape[0]
        self.filter_width = filter_shape[1]
        self.padding = padding
        self.stride = stride
        self.is_trainable = is_trainable
        self.W = None  # param W
        self.B = None  # param B
        self.optimizer_W = optimizer()
        self.optimizer_B = optimizer()

    def init_params(self, input_shape: tuple):
        assert len(input_shape) == 4  # [batch size, n_channel, height, width]
        in_channel = input_shape[1]
        limit = 1.0 / np.sqrt(self.filter_height * self.filter_width)
        out_channel = self.n_filters
        self.W = np.random.uniform(
            -limit,
            limit,
            size=(out_channel, in_channel, self.filter_height, self.filter_width),
        )
        self.B = np.zeros((out_channel, 1))

    def image_to_column(
        self,
        X: np.ndarray,
        filter_height: int,
        filter_width: int,
        padding: str,
        stride: int,
    ):
        pass

    def forward(self, X: np.ndarray, is_training: bool):
        assert X.ndim == 4  # [batch size, n_channel, height, width]
        if self.first == True:
            self.init_params(X.shape)
            self.first = False
        batch_size, in_channel, x_height, x_width = X.shape
        self.input = X
        self.input_column = self.image_to_column(
            X, self.filter_height, self.filter_width, self.padding, self.stride
        )

    def backward(self, accm_grad: np.ndarray):
        raise NotImplementedError
