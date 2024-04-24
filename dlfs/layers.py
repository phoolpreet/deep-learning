import numpy as np
from gate import MultiplyGate
from gate import AddGate
from optimizers import Optimizer


class Layer:
    def forward(self, X: np.ndarray, is_training: bool):
        raise NotImplementedError

    def backward(self, accm_grad: np.ndarray):
        raise NotImplementedError


class Dense(Layer):
    def __init__(
        self, hidden_unit: int, optimizer: Optimizer, is_trainable: bool = True
    ):
        assert hidden_unit > 0
        self.first = True
        self.hidden_unit = hidden_unit
        self.is_trainable = is_trainable  # is this layer trainable or fixed weights
        self.W = None  # param W
        self.B = None  # param B
        self.optimizer_W = optimizer()
        self.optimizer_B = optimizer()
        self.multiplygate = MultiplyGate()
        self.addgate = AddGate()

    def init_params(self, input_shape):
        assert len(input_shape) == 2
        limit = 1.0 / np.sqrt(input_shape[1])
        self.W = np.random.uniform(-limit, limit, (input_shape[1], self.hidden_unit))
        self.B = np.zeros((1, self.hidden_unit))

    # X : [smples x features]
    def forward(self, X: np.ndarray, is_training: bool = True):
        if self.first == True:
            self.init_params(X.shape)
            self.first = False
        Z1 = self.multiplygate.forward(X, self.W)
        Z2 = self.addgate.forward(Z1, self.B)
        return Z2

    def backward(self, accum_grad: np.ndarray):
        # accum_grad = dZ2
        dZ1, dB = self.addgate.backward(accum_grad)
        dW, dX = self.multiplygate.backward(dZ1)
        if self.is_trainable:
            self.W = self.optimizer_W.update(self.W, dW)
            self.B = self.optimizer_B.update(self.B, dB)
        return dX


if __name__ == "__main__":
    from optimizers import StochadticGradientDescent
    from optimizers import Adagrad
    from losses import SquaredLoss

    X = np.random.randn(12, 3)
    Y = np.random.randn(12, 1)

    dense1 = Dense(4, Adagrad)
    dense2 = Dense(1, Adagrad)
    loss_fn = SquaredLoss()

    for i in range(50):
        Z1 = dense1.forward(X, is_training=True)
        Z2 = dense2.forward(Z1)
        loss = loss_fn.forward(Y, Z2)
        print("loss", loss)
        dZ2 = loss_fn.backward(Y, Z2)
        dZ1 = dense2.backward(dZ2)
        dX = dense1.backward(dZ1)

    