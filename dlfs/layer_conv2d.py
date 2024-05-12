import numpy as np
from gate import MultiplyGate
from gate import AddGate
from optimizers import Optimizer
from optimizers import Adam
import math


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

    def get_img2col_indices(
        self,
        img_channel: int,
        img_height: int,
        img_width: int,
        filter_height: int,
        filter_width: int,
        padding_h: tuple,
        padding_w: tuple,
        stride: int,
    ):
        out_height = int((img_height + np.sum(padding_h) - filter_height) / stride + 1)
        out_width = int((img_width + np.sum(padding_w) - filter_width) / stride + 1)
        i0 = np.repeat(np.arange(filter_height), filter_width)
        i0 = np.tile(i0, img_channel)
        i1 = stride * np.repeat(np.arange(out_height), out_width)
        j0 = np.tile(np.arange(filter_width), filter_height * img_channel)
        j1 = stride * np.tile(np.arange(out_width), out_height)
        i = i0.reshape(-1, 1) + i1.reshape(1, -1)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)
        k = np.repeat(np.arange(img_channel), filter_height * filter_width).reshape(
            -1, 1
        )
        return i, j, k

    def determine_padding(
        self,
        filter_height: int,
        filter_width: int,
        padding: str,
    ):
        assert padding in ["valid", "same"]
        if padding == "valid":
            pad_h = (0, 0)
            pad_w = (0, 0)
        elif padding == "same":
            pad_h1 = int(math.floor((filter_height - 1) / 2))
            pad_h2 = int(math.ceil((filter_height - 1) / 2))
            pad_w1 = int(math.floor((filter_width - 1) / 2))
            pad_w2 = int(math.ceil((filter_width - 1) / 2))
            pad_h = (pad_h1, pad_h2)
            pad_w = (pad_w1, pad_w2)
        return pad_h, pad_w

    def output_shape(
        self,
        n_filters: int,
        filter_height: int,
        filter_width: int,
        input_height: int,
        input_width: int,
        padding: str,
    ):
        pad_h, pad_w = self.determine_padding(filter_height, filter_width, padding)
        output_height = (input_height + np.sum(pad_h) - filter_height) / self.stride + 1
        output_width = (input_width + np.sum(pad_w) - filter_width) / self.stride + 1
        return n_filters, int(output_height), int(output_width)

    def image_to_column(
        self,
        X: np.ndarray,  # [batch, channel, height, width]
        filter_height: int,
        filter_width: int,
        padding: str,
        stride: int,
    ):
        pad_h, pad_w = self.determine_padding(filter_height, filter_width, padding)

        X_padded = np.pad(X, ((0, 0), (0, 0), pad_h, pad_w), mode="constant")
        batch, img_channel, img_height, img_width = X.shape
        i, j, k = self.get_img2col_indices(
            img_channel,
            img_height,
            img_width,
            filter_height,
            filter_width,
            pad_h,
            pad_w,
            stride,
        )
        cols = X_padded[:, k, i, j]
        cols = cols.transpose(1, 2, 0).reshape(
            filter_height * filter_width * img_channel, -1
        )
        return cols

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
        self.W_column = self.W.reshape((self.n_filters, -1))
        output = self.W_column.dot(self.input_column) + self.B
        output_shape = self.output_shape(
            self.n_filters,
            self.filter_height,
            self.filter_width,
            x_height,
            x_width,
            self.padding,
        )
        output = output.reshape(output_shape + (batch_size, ))
        return output.transpose(3, 0, 1, 2)

    def backward(self, accum_grad: np.ndarray):
        assert accum_grad.ndim == 4
        accum_grad = accum_grad.transpose(1, 2, 3, 0).reshape(self.n_filters, -1)
        if self.is_trainable:
            W_grad = accum_grad.dot(self.input_column.T).reshape(self.W.shape)
            B_grad = np.sum(accum_grad, axis=1, keepdims=True)
            self.W = self.optimizer_W.update(self.W, W_grad)
            self.B = self.optimizer_B.update(self.B, B_grad)


if __name__ == "__main__":
    conv2d = Conv2D(10, (5, 6), "same", 2, Adam, True)
    X = np.random.randn(32, 3, 255, 200)
    Z = conv2d.forward(X, is_training=True)
    Z_grad = np.random.randn(*Z.shape)
    conv2d.backward(Z_grad)
