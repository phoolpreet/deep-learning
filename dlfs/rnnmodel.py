from datetime import datetime
import numpy as np
import sys
from rnnlayer import RNNLayer
from rnn.losses import Softmax


class RNNModel:
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        self.U = np.random.uniform(
            -np.sqrt(1.0 / word_dim), np.sqrt(1.0 / word_dim), (hidden_dim, word_dim)
        )
        self.W = np.random.uniform(
            -np.sqrt(1.0 / hidden_dim),
            np.sqrt(1.0 / hidden_dim),
            (hidden_dim, hidden_dim),
        )
        self.V = np.random.uniform(
            -np.sqrt(1.0 / hidden_dim),
            np.sqrt(1.0 / hidden_dim),
            (word_dim, hidden_dim),
        )

    """
        forward propagation (predicting word probabilities)
        x is one single data, and a batch of data
        for example x = [0, 179, 341, 416], then its y = [179, 341, 416, 1]
    """

    def forward_propagation(self, X):
        sequence_length = len(X)
        layers = []
        H = np.zeros(self.hidden_dim)
        for t in range(sequence_length):
            layer = RNNLayer()
            input = np.zeros(self.word_dim)
            input[X[t]] = 1
            H, O = layer.forward(input, H, self.U, self.W, self.V)
            layers.append(layer)
        return layers

    def predict(self, X):
        softmax = Softmax()
        layers = self.forward_propagation(X)
        return [np.argmax(softmax.predict(layer.Out)) for layer in layers]
    
    def calculate_loss(self, X, Y):
        assert len(X) == len(Y)
        softmax = Softmax()
        layers = self.forward_propagation(X)
        loss = 0.0
        for i, layer in enumerate(layers):
            loss += softmax.loss(layer.Out, Y[i])
        return loss / len(Y)
    
    def calculate_total_loss(self, X, Y):
        assert len(X) == len(Y)
        loss = 0.0
        for i in range(len(X)):
            loss += self.calculate_loss(X[i], Y[i])
        return loss / len(Y)

    def bptt(self, X, Y):
        assert len(X) == len(Y)
        softmax = Softmax()
        layers = self.forward_propagation(X)
        dU = np.zeros_like(self.U)
        dW = np.zeros_like(self.W)
        dV = np.zeros_like(self.V)

        sequence_length = len(layers)
        H = np.zeros(self.hidden_dim)
        dH = np.zeros_like(H)
        for t in reversed(range(sequence_length)):
            dOut = softmax.diff(layers[t].Out, Y[t])
            input_onehot = np.zeros(self.word_dim)
            input_onehot[X[t]] = 1
            dH_prev, dU_t, dW_t, dV_t = layers[t].backward(dH, dOut)
            dU += dU_t
            dW += dW_t
            dV += dV_t
            dH += dH_prev
            dH = dH_prev
        return dH, dU, dW, dV


