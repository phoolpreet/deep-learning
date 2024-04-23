import numpy as np


class Sigmoid:
    def __init__(self):
        self.output = None

    def forward(self, X):
        sigmoid = 1.0 / (1.0 + np.exp(-X))
        self.output = sigmoid
        return sigmoid

    def backward(self):
        if self.output is None:
            raise Exception("Signoid: backward called befor forward")
        sigmoid = self.output
        dX = (1.0 - sigmoid) * sigmoid
        self.output = None
        return dX


class Softmax:
    def __init__(self):
        self.output = None

    def forward(self, X):
        X = X - np.max(X, axis=-1, keepdims=True)
        EX = np.exp(X)
        softmax = EX / np.sum(EX, axis=-1, keepdims=True)
        self.output = softmax
        return softmax

    def backward(self):
        if self.output is None:
            raise Exception("Softmax: backward called befor forward")
        softmax = self.output
        dX = (1.0 - softmax) * softmax
        self.output = None
        return dX


class Tanh:
    def __init__(self):
        self.output = None

    def forward(self, X):
        tanh = np.tanh(X)
        self.output = tanh
        return tanh

    def backward(self):
        if self.output is None:
            raise Exception("Tanh: backward called befor forward")
        tanh = self.output
        dX = 1.0 - np.square(tanh)
        self.output = None
        return dX


class ReLU:
    def __init__(self):
        self.input = None

    def forward(self, X):
        relu = np.where(X >= 0, X, 0)
        self.input = X
        return relu

    def backward(self):
        if self.input is None:
            raise Exception("ReLU: backward called befor forward")
        dX = np.where(self.input >= 0, 1, 0)
        self.input = None
        return dX


class LeakyReLU:
    def __init__(self, alpha: float = 0.2):
        self.input = None
        self.alpha = alpha

    def forward(self, X):
        leakyrelu = np.where(X >= 0, X, self.alpha * X)
        self.input = X
        return leakyrelu

    def backward(self):
        if self.input is None:
            raise Exception("LeakyReLU: backward called befor forward")
        dX = np.where(self.input >= 0, 1, self.alpha)
        self.input = None
        return dX


class ELU:
    def __init__(self, alpha: float = 0.1):
        self.input = None
        self.alpha = alpha

    def forward(self, X):
        elu = np.where(X >= 0, X, self.alpha * (np.exp(X) - 1.0))
        self.input = X
        return elu

    def backward(self):
        if self.input is None:
            raise Exception("ELU: backward called befor forward")
        dX = np.where(self.input >= 0, 1, self.alpha * np.exp(self.input))
        self.input = None
        return dX


class SELU:
    def __init__(self, alpha: float = 0.1):
        self.input = None
        # https://arxiv.org/pdf/1706.02515.pdf
        self.alpha = 1.6733
        self.scale = 1.0507

    def forward(self, X):
        selu = self.scale * np.where(X >= 0, X, self.alpha * (np.exp(X) - 1.0))
        self.input = X
        return selu

    def backward(self):
        if self.input is None:
            raise Exception("SELU: backward called befor forward")
        dX = self.scale * np.where(self.input >= 0, 1, self.alpha * np.exp(self.input))
        self.input = None
        return dX


class SoftPlus:
    def __init__(self, alpha: float = 0.1):
        self.input = None

    def forward(self, X):
        softplus = np.log(1.0 + np.exp(X))
        self.input = X
        return softplus

    def backward(self):
        if self.input is None:
            raise Exception("SoftPlus: backward called befor forward")
        dX = 1.0 / (1.0 + np.exp(-self.input))
        self.input = None
        return dX


if __name__ == "__main__":
    a = np.random.randn(3)
    s = SoftPlus()

    f = s.forward(a)
    b = s.backward()
    print("a", a)
    print("f", f)
    print("b", b)