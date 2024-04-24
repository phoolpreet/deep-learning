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
        self.SSG = None # sum of squares of gradients
        self.eps = 1.0e-8
    
    def update(self, param: np.ndarray, param_grad: np.ndarray):
        assert param.shape == param_grad.shape
        if self.SSG is not None:
            assert self.SSG.shape == param.shape
        if self.SSG is None:
            self.SSG = np.zeros_like(param)
        self.SSG += np.power(param_grad, 2)
        param_new = param - self.learning_rate * param_grad / np.sqrt(self.SSG + self.eps)
        return param_new
        