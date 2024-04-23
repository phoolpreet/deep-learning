import numpy as np


class MultiplyGate:
    def __init__(self):
        self.input = None

    def forward(self, W, X):
        self.input = {}
        self.input["W"] = W
        self.input["X"] = X
        Z = np.dot(W, X)
        return Z

    def backward(self, dZ):
        if self.input is None:
            raise Exception("MultiplyGate: backward called with no input set")
        assert dZ.shape[0] == self.input["W"].shape[0] 
        assert dZ.shape[1] == self.input["X"].shape[1] 
        dW = np.dot(dZ, self.input["X"].T)
        dX = np.dot(self.input["W"].T, dZ)
        self.input = None
        return dW, dX


class AddGate:
    def __init__(self):
        self.input = None

    def forward(self, X1, X2):
        assert X1.shape == X2.shape
        self.input = {}
        self.input["X1"] = X1
        self.input["X2"] = X2
        Z = X1 + X2
        return Z

    def backward(self, dZ):
        if self.input is None:
            raise Exception("AddGate: backward called with no input set")
        assert dZ.shape == self.input["X1"].shape
        dX1 = np.copy(dZ)
        dX2 = np.copy(dZ)
        self.input = None
        return dX1, dX2


if __name__ == "__main__":

    W = np.random.randn(5, 7)
    X1 = np.random.randn(7, 10)
    X2 = np.random.randn(7, 10)

    addgate = AddGate()
    # V1 = addgate.forward(X1, X2)
    V2, V3 = addgate.backward(np.ones_like(X1))
    print(id(V2))
    print(id(V3))

