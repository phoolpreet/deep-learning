import numpy as np


class MultiplyGate:
    def __init__(self):
        self.input = None

    def forward(self, X, W):
        # X : [smples x features]
        assert X.ndim == 2
        assert W.ndim == 2
        assert X.shape[1] == W.shape[0]
        self.input = {}
        self.input["W"] = W
        self.input["X"] = X
        Z = np.dot(X, W)
        return Z

    def backward(self, dZ):
        if self.input is None:
            raise Exception("MultiplyGate: backward called with no input set")
        assert dZ.shape[0] == self.input["X"].shape[0] 
        assert dZ.shape[1] == self.input["W"].shape[1] 
        dW = np.dot(self.input["X"].T, dZ)
        dX = np.dot(dZ, self.input["W"].T)
        self.input = None
        return dW, dX


class AddGate:
    def __init__(self):
        self.input = None

    def forward(self, X1, X2):
        self.input = {}
        self.input["X1"] = X1
        self.input["X2"] = X2
        Z = X1 + X2
        return Z

    def backward(self, dZ):
        if self.input is None:
            raise Exception("AddGate: backward called with no input set")
        dX1 = np.copy(dZ)
        dX2 = np.copy(dZ)
        if dX1.shape != self.input["X1"].shape:
            dX1 = np.mean(dX1, axis=0, keepdims=True)   # rows are sample
        if dX2.shape != self.input["X2"].shape:
            dX2 = np.mean(dX2, axis=0, keepdims=True)   # rows are sample
        assert dX1.shape == self.input["X1"].shape
        assert dX2.shape == self.input["X2"].shape
        self.input = None
        return dX1, dX2


if __name__ == "__main__":

    W = np.random.randn(30, 15)
    X1 = np.random.randn(100, 30)

    mgate = MultiplyGate()
    Z = mgate.forward(X1, W)
    dW, dX = mgate.backward(np.ones_like(Z))
    print((dW))
    print((dX))

