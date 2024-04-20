import numpy as np

class Sigmoid:
    def __init__(self):
        self.output = None

    def forward(self, X):
        sigmoid = 1.0 / (1.0 + np.exp(-X))
        self.output = sigmoid
        return sigmoid
    
    def backward(self, dS):
        if self.output is None:
            raise Exception("Signoid: backward called befor forward")
        sigmoid = self.output
        dX = (1.0 - sigmoid) * sigmoid * dS
        self.output = None
        return dX

    

class Tanh:
    def __init__(self):
        self.output = None

    def forward(self, X):
        tanh = np.tanh(X)
        self.output = tanh
        return tanh
    
    def backward(self, dT):
        if self.output is None:
            raise Exception("Tanh: backward called befor forward")
        tanh = self.output
        dX = (1.0 - np.square(tanh)) * dT
        self.output = None
        return dX
    

if __name__ == "__main__":
    a = np.random.randn(3, 4)
    b = np.random.randn(3, 4)
    sig = Tanh()

    o1 = sig.forward(a)
    o2 = sig.backward(a, b)
    print(a)
    print(b)
    print(o1)
    print(o2)