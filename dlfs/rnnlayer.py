from .activation import Tanh
from .gate import MultiplyGate, AddGate



class RNNLayer:
    def __init__(self):
        self.multiply_gate_1 = MultiplyGate()
        self.multiply_gate_2 = MultiplyGate()
        self.multiply_gate_3 = MultiplyGate()
        self.add_gate = AddGate()
        self.activation = Tanh()
        
    def forward(self, X, H_prev, U, W, V):
        self.var_1 = self.multiply_gate_1.forward(U, X)
        self.var_2 = self.multiply_gate_2.forward(W, H_prev)
        self.var_3 = self.add_gate.forward(self.var_2, self.var_1)
        self.H = self.activation.forward(self.var_3)
        self.O = self.multiply_gate_3.forward(V, self.H)
        return self.H, self.Out

    def backward(self, dH, dOut):
        dV, dH_partial = self.multiply_gate_3.backward(dOut)
        dH = dH + dH_partial
        dVar_3 = self.activation.backward(dH)
        dVar_2, dVar_1 = self.add_gate.backward(dVar_3)
        dW, dH_prev = self.multiply_gate_2.backward(dVar_2)
        dU, dX = self.multiply_gate_1.backward(dVar_1)
        return dH_prev, dU, dW, dV