import numpy as np


class Optimizer:
    def update(self, param: np.ndarray, param_grad: np.ndarray):
        raise NotImplementedError


class StochadticGradientDescent(Optimizer):
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0):
        assert 0 < learning_rate < 1.0
        assert 0 <= momentum < 1.0
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.param_updt = None

    def update(self, param: np.ndarray, param_grad: np.ndarray):
        assert param.shape == param_grad.shape
        if self.param_updt is not None:
            assert self.param_updt.shape == param.shape
        if self.param_updt is None:
            self.param_updt = np.zeros_like(param)
        self.param_updt = (
            self.momentum * self.param_updt + (1.0 - self.momentum) * param_grad
        )
        param_new = param - self.learning_rate * self.param_updt
        return param_new


class Adagrad(Optimizer):
    def __init__(self, learning_rate: float = 0.01):
        assert 0 < learning_rate < 1.0
        self.learning_rate = learning_rate
        self.SSG = None  # sum of squares of gradients
        self.eps = 1.0e-8

    def update(self, param: np.ndarray, param_grad: np.ndarray):
        assert param.shape == param_grad.shape
        if self.SSG is not None:
            assert self.SSG.shape == param.shape
        if self.SSG is None:
            self.SSG = np.zeros_like(param)
        self.SSG += np.power(param_grad, 2)
        param_new = param - self.learning_rate * param_grad / np.sqrt(
            self.SSG + self.eps
        )
        return param_new


class Adadelta(Optimizer):
    def __init__(self, rho: float = 0.95, eps: float = 1.0e-6):
        self.rho = rho
        self.eps = eps
        self.param_updt = None
        self.avg_param_updt = None
        self.avg_param_grad = None

    def update(self, param: np.ndarray, param_grad: np.ndarray):
        assert param.shape == param_grad.shape
        if self.param_updt is not None:
            assert self.param_updt.shape == param.shape
            assert self.avg_param_updt.shape == param.shape
            assert self.avg_param_grad.shape == param.shape
        if self.param_updt is None:
            self.param_updt = np.zeros_like(param)
            self.avg_param_updt = np.zeros_like(param)
            self.avg_param_grad = np.zeros_like(param)

        self.avg_param_grad = (self.rho * self.avg_param_grad) + (
            1.0 - self.rho
        ) * np.power(param_grad, 2)

        RMS_delta_param = np.sqrt(self.avg_param_updt + self.eps)
        RMS_grad = np.sqrt(self.avg_param_grad + self.eps)
        adaptive_lr = RMS_delta_param / RMS_grad

        self.param_updt = adaptive_lr * param_grad
        self.avg_param_updt = (self.rho * self.avg_param_updt) + (
            1.0 - self.rho
        ) * np.power(self.param_updt, 2)

        param_new = param - self.param_updt
        return param_new


class RMSprop(Optimizer):
    def __init__(
        self, learning_rate: float = 0.01, rho: float = 0.9, eps: float = 1.0e-8
    ):
        self.learning_rate = learning_rate
        self.rho = rho
        self.eps = eps
        self.avg_param_grad = None

    def update(self, param: np.ndarray, param_grad: np.ndarray):
        assert param.shape == param_grad.shape
        if self.avg_param_grad is not None:
            assert self.avg_param_grad.shape == param.shape
        if self.avg_param_grad is None:
            self.avg_param_grad = np.zeros_like(param)

        self.avg_param_grad = self.rho * self.avg_param_grad + (
            1.0 - self.rho
        ) * np.power(param_grad, 2)

        param_new = param - self.learning_rate * param_grad / np.sqrt(
            self.avg_param_grad + self.eps
        )
        return param_new


class Adam(Optimizer):
    def __init__(
        self, learning_rate: float = 0.01, b1: float = 0.9, b2: float = 0.999
    ):
        self.learning_rate = learning_rate
        self.eps = 1.0e-8
        self.m = None
        self.v = None
        self.b1 = b1
        self.b2 = b2

    def update(self, param: np.ndarray, param_grad: np.ndarray):
        assert param.shape == param_grad.shape
        if self.m is None:
            self.m = np.zeros_like(param)
            self.v = np.zeros_like(param)
        
        self.m = self.b1 * self.m + (1 - self.b1) * param_grad
        self.v = self.b2 * self.v + (1 - self.b1) * np.power(param_grad, 2)
        m_hat = self.m / (1 - self.b1)
        v_hat = self.v / (1 - self.b2)

        param_updt = self.learning_rate * m_hat / (np.sqrt(v_hat + self.eps))
        param_new = param - param_updt
        return param_new
