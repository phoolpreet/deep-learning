from optimizers import StochadticGradientDescent
from optimizers import Adagrad
from optimizers import Adadelta
from optimizers import RMSprop
from optimizers import Adam

from activation import Sigmoid
from activation import Softmax
from activation import Tanh
from activation import ReLU
from activation import LeakyReLU
from activation import ELU
from activation import SELU
from activation import SoftPlus
from losses import SquaredLoss

from layers import Dense

from matplotlib import pyplot as plt
import numpy as np


def test_optimizers():
    X = np.random.randn(12, 3)
    Y = np.random.randn(12, 1)
    loss_fn = SquaredLoss()

    # StochadticGradientDescent ---------------------------------
    dense1 = Dense(4, StochadticGradientDescent)
    activation = Sigmoid()
    dense2 = Dense(1, StochadticGradientDescent)
    lst1 = []
    for i in range(50000):
        Z1 = dense1.forward(X, is_training=True)
        Z2 = activation.forward(Z1)
        Z3 = dense2.forward(Z2)
        loss = loss_fn.forward(Y, Z3)
        lst1.append(loss)
        if i == 0:
            W1 = np.copy(dense1.W)
            B1 = np.copy(dense1.B)
            W2 = np.copy(dense2.W)
            B2 = np.copy(dense2.B)
        dZ3 = loss_fn.backward(Y, Z3)
        dZ2 = dense2.backward(dZ3)
        dZ1 = activation.backward(dZ2)
        dX = dense1.backward(dZ1)
    plt.plot(lst1, label="sgd")

    # Adagrad ---------------------------------
    dense1 = Dense(4, Adagrad)
    dense1.first = False
    dense1.W = np.copy(W1)
    dense1.B = np.copy(B1)
    activation = Sigmoid()
    dense2 = Dense(1, Adagrad)
    dense2.first = False
    dense2.W = np.copy(W2)
    dense2.B = np.copy(B2)
    lst2 = []
    for i in range(50000):
        Z1 = dense1.forward(X, is_training=True)
        Z2 = activation.forward(Z1)
        Z3 = dense2.forward(Z2)
        loss = loss_fn.forward(Y, Z3)
        lst2.append(loss)
        dZ3 = loss_fn.backward(Y, Z3)
        dZ2 = dense2.backward(dZ3)
        dZ1 = activation.backward(dZ2)
        dX = dense1.backward(dZ1)
    plt.plot(lst2, label="Adagrad")

    # Adadelta ---------------------------------
    dense1 = Dense(4, Adadelta)
    dense1.first = False
    dense1.W = np.copy(W1)
    dense1.B = np.copy(B1)
    activation = Sigmoid()
    dense2 = Dense(1, Adadelta)
    dense2.first = False
    dense2.W = np.copy(W2)
    dense2.B = np.copy(B2)
    lst2 = []
    for i in range(50000):
        Z1 = dense1.forward(X, is_training=True)
        Z2 = activation.forward(Z1)
        Z3 = dense2.forward(Z2)
        loss = loss_fn.forward(Y, Z3)
        lst2.append(loss)
        dZ3 = loss_fn.backward(Y, Z3)
        dZ2 = dense2.backward(dZ3)
        dZ1 = activation.backward(dZ2)
        dX = dense1.backward(dZ1)
    plt.plot(lst2, label="Adadelta")

    # RMSprop ---------------------------------
    dense1 = Dense(4, RMSprop)
    dense1.first = False
    dense1.W = np.copy(W1)
    dense1.B = np.copy(B1)
    activation = Sigmoid()
    dense2 = Dense(1, RMSprop)
    dense2.first = False
    dense2.W = np.copy(W2)
    dense2.B = np.copy(B2)
    lst2 = []
    for i in range(50000):
        Z1 = dense1.forward(X, is_training=True)
        Z2 = activation.forward(Z1)
        Z3 = dense2.forward(Z2)
        loss = loss_fn.forward(Y, Z3)
        lst2.append(loss)
        dZ3 = loss_fn.backward(Y, Z3)
        dZ2 = dense2.backward(dZ3)
        dZ1 = activation.backward(dZ2)
        dX = dense1.backward(dZ1)
    plt.plot(lst2, label="RMSprop")

    # Adam ---------------------------------
    dense1 = Dense(4, Adam)
    dense1.first = False
    dense1.W = np.copy(W1)
    dense1.B = np.copy(B1)
    activation = Sigmoid()
    dense2 = Dense(1, Adam)
    dense2.first = False
    dense2.W = np.copy(W2)
    dense2.B = np.copy(B2)
    lst2 = []
    for i in range(50000):
        Z1 = dense1.forward(X, is_training=True)
        Z2 = activation.forward(Z1)
        Z3 = dense2.forward(Z2)
        loss = loss_fn.forward(Y, Z3)
        lst2.append(loss)
        dZ3 = loss_fn.backward(Y, Z3)
        dZ2 = dense2.backward(dZ3)
        dZ1 = activation.backward(dZ2)
        dX = dense1.backward(dZ1)
    plt.plot(lst2, label="Adam")

    plt.legend()
    plt.show()


def test_activations():

    X = np.random.randn(12, 3)
    Y = np.random.randn(12, 1)
    loss_fn = SquaredLoss()

    # Sigmoid ---------------------------------
    dense1 = Dense(4, Adam)
    activation = Sigmoid()
    dense2 = Dense(1, Adam)
    lst1 = []
    for i in range(500):
        Z1 = dense1.forward(X, is_training=True)
        Z2 = activation.forward(Z1)
        Z3 = dense2.forward(Z2)
        loss = loss_fn.forward(Y, Z3)
        lst1.append(loss)
        if i == 0:
            W1 = np.copy(dense1.W)
            B1 = np.copy(dense1.B)
            W2 = np.copy(dense2.W)
            B2 = np.copy(dense2.B)
        dZ3 = loss_fn.backward(Y, Z3)
        dZ2 = dense2.backward(dZ3)
        dZ1 = activation.backward(dZ2)
        dX = dense1.backward(dZ1)
    plt.plot(lst1, label="sigmoid")

    # Softmax ---------------------------------
    dense1 = Dense(4, Adam)
    dense1.first = False
    dense1.W = np.copy(W1)
    dense1.B = np.copy(B1)
    activation = Softmax()
    dense2 = Dense(1, Adam)
    dense2.first = False
    dense2.W = np.copy(W2)
    dense2.B = np.copy(B2)
    lst2 = []
    for i in range(500):
        Z1 = dense1.forward(X, is_training=True)
        Z2 = activation.forward(Z1)
        Z3 = dense2.forward(Z2)
        loss = loss_fn.forward(Y, Z3)
        lst2.append(loss)
        dZ3 = loss_fn.backward(Y, Z3)
        dZ2 = dense2.backward(dZ3)
        dZ1 = activation.backward(dZ2)
        dX = dense1.backward(dZ1)
    plt.plot(lst2, label="softmax")

    # Tanh ---------------------------------
    dense1 = Dense(4, Adam)
    dense1.first = False
    dense1.W = np.copy(W1)
    dense1.B = np.copy(B1)
    activation = Tanh()
    dense2 = Dense(1, Adam)
    dense2.first = False
    dense2.W = np.copy(W2)
    dense2.B = np.copy(B2)
    lst2 = []
    for i in range(500):
        Z1 = dense1.forward(X, is_training=True)
        Z2 = activation.forward(Z1)
        Z3 = dense2.forward(Z2)
        loss = loss_fn.forward(Y, Z3)
        lst2.append(loss)
        dZ3 = loss_fn.backward(Y, Z3)
        dZ2 = dense2.backward(dZ3)
        dZ1 = activation.backward(dZ2)
        dX = dense1.backward(dZ1)
    plt.plot(lst2, label="tanh")

    # Relu ---------------------------------
    dense1 = Dense(4, Adam)
    dense1.first = False
    dense1.W = np.copy(W1)
    dense1.B = np.copy(B1)
    activation = ReLU()
    dense2 = Dense(1, Adam)
    dense2.first = False
    dense2.W = np.copy(W2)
    dense2.B = np.copy(B2)
    lst2 = []
    for i in range(500):
        Z1 = dense1.forward(X, is_training=True)
        Z2 = activation.forward(Z1)
        Z3 = dense2.forward(Z2)
        loss = loss_fn.forward(Y, Z3)
        lst2.append(loss)
        dZ3 = loss_fn.backward(Y, Z3)
        dZ2 = dense2.backward(dZ3)
        dZ1 = activation.backward(dZ2)
        dX = dense1.backward(dZ1)
    plt.plot(lst2, label="relu")

    # Leaky Relu ---------------------------------
    dense1 = Dense(4, Adam)
    dense1.first = False
    dense1.W = np.copy(W1)
    dense1.B = np.copy(B1)
    activation = LeakyReLU()
    dense2 = Dense(1, Adam)
    dense2.first = False
    dense2.W = np.copy(W2)
    dense2.B = np.copy(B2)
    lst2 = []
    for i in range(500):
        Z1 = dense1.forward(X, is_training=True)
        Z2 = activation.forward(Z1)
        Z3 = dense2.forward(Z2)
        loss = loss_fn.forward(Y, Z3)
        lst2.append(loss)
        dZ3 = loss_fn.backward(Y, Z3)
        dZ2 = dense2.backward(dZ3)
        dZ1 = activation.backward(dZ2)
        dX = dense1.backward(dZ1)
    plt.plot(lst2, label="leaky-relu")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    # test_activations()
    test_optimizers()
