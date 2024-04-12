import numpy as np
from typing import List, Dict, Tuple
from matplotlib import pyplot as plt
from collections import deque
from copy import deepcopy
from scipy.special import logsumexp


def sigmoid(x: np.ndarray):
    return 1.0 / (1.0 + np.exp(-x))


def dsigmoid(x: np.ndarray):
    return sigmoid(x) * (1.0 - sigmoid(x))


def tanh(x: np.ndarray):
    return np.tanh(x)


def dtanh(x: np.ndarray):
    return 1.0 - (np.tanh(x) * np.tanh(x))


def softmax(x: np.ndarray, axis=None):
    return np.exp(x - logsumexp(x, axis=axis, keepdims=True))


def batch_softmax(input_array: np.ndarray):
    out = []
    for row in input_array:
        out.append(softmax(row, axis=1))
    return np.stack(out)


class RNNOptimizer:
    def __init__(self, lr: float = 0.01, gradient_clipping: bool = True) -> None:
        self.lr = lr
        self.gradient_clipping = gradient_clipping
        self.first = True

    def step(self) -> None:
        for layer in self.model.layers:
            for key in layer.params.key():
                if self.gradient_clipping:
                    np.clip(
                        layer.params[key]["deriv"], -2, 2, layer.params[key["deriv"]]
                    )
                self._update_rule(
                    param=layer.param[key]["value"], grad=layer.param[key]["deriv"]
                )

    def _update_rule(self, **kwargs) -> None:
        raise NotImplementedError()


class SGD(RNNOptimizer):
    def __init__(self, lr: float = 0.01, gradient_clipping: bool = True) -> None:
        super().__init(lr, gradient_clipping)

    def _update_rule(self, **kwargs) -> None:
        update = self.lr * kwargs["grad"]
        kwargs["param"] -= update


class AdaGrad(RNNOptimizer):
    def _init__(self, lr: float = 0.01, gradient_clipping: bool = True) -> None:
        super().__init__(lr, gradient_clipping)
        self.lr = lr
        self.gradient_clipping = gradient_clipping
        self.eps = 1.0e-7
        self.first = True

    def step(self) -> None:
        if self.first:
            self.sum_squares = {}
            for i, layer in enumerate(self.model.layers):
                self.sum_squares[i] = {}
                for key in layer.params.keys():
                    self.sum_squares[i][key] = np.zeros_like(layer.params[key]["value"])
        self.first = False

        for i, layer in enumerate(self.model.layers):
            for key in layer.params.keys():
                if self.gradient_clipping:
                    np.clip(
                        layer.params[key]["deriv"], -2, 2, layer.params[key]["deriv"]
                    )
                self._update_rule(
                    param=layer.params[key]["value"],
                    grad=layer.params[key]["deriv"],
                    sum_square=self.sum_squares[i][key],
                )

    def _update_rule(self, **kwargs) -> None:
        kwargs["sum_square"] += self.eps + np.power(kwargs["grad"], 2)
        lr = np.divide(self.lr, np.sqrt(kwargs["sum_square"]))
        kwargs["param"] -= lr * kwargs["grad"]


class Loss:
    def __init__(self):
        pass

    def forward(self, prediction: np.ndarray, target: np.ndarray) -> float:
        assert prediction.shape == target.shape
        self.prediction = prediction
        self.target = target
        self.output = self._output()
        return self.output

    def backward(self) -> np.ndarray:
        self.input_grad = self._input_grad()
        assert self.prediction.shape == self.input_grad.shape
        return self.input_grad

    def _output(self) -> float:
        raise NotImplementedError

    def _input_grad(self) -> np.ndarray:
        raise NotImplementedError


class SoftmaxCrossEntropy(Loss):
    def __init__(self, eps: float = 1e-9) -> None:
        super().__init()
        self.eps = eps
        self.single_class = False

    def _output(self) -> float:
        out = []
        for row in self.prediction:
            out.append(softmax(row, axis=1))
        softmax_preds = np.stack(out)
        self.softmax_preds = np.clip(softmax_preds, self.eps, 1.0 - self.eps)
        softmax_cross_entropy_loss = -1.0 * self.target * np.log(self.softmax_preds) - (
            1.0 - self.target
        ) * np.log(1.0 - self.softmax_preds)
        return np.sum(softmax_cross_entropy_loss)

    def _input_grad(self) -> np.ndarray:
        return self.softmax_preds - self.target


class RNNNode:
    def __init__(self):
        pass

    def forward(
        self, x_in: np.ndarray, H_in: np.ndarray, params_dict: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray]:
        """
        param x_in: numpy array of shape (batch_size, vocab_size) one-hot?
        param H_in: numpy array of shape (batch_size, hidden_size)
        param params_dict: RNN params dictionary
        return self.X_out: numpy array of shape (batch_size, vocab_size)
        return self.H_out: numpy array of shape (batch_size, hidden_size)
        """
        self.X_in = x_in
        self.H_in = H_in
        self.Z = np.column_stack((x_in, H_in))
        self.H_int = (
            np.dot(self.Z, params_dict["W_f"]["value"]) + params_dict["B_f"]["value"]
        )
        self.H_out = tanh(self.H_int)
        self.X_out = (
            np.dot(self.H_out, params_dict["W_v"]["value"])
            + params_dict["B_v"]["value"]
        )
        return self.X_out, self.H_out

    def backward(
        self,
        X_out_grad: np.ndarray,
        H_out_grad: np.ndarray,
        params_dict: Dict[str, Dict[str, np.ndarray]],
    ) -> Tuple[np.ndarray]:
        """
        param X_out_grad: numpy array of size (batch_size, vocab_size)
        param H_out_grad: numpy array of size (batch_size, hidden_size)
        param params_dict: RNN params dictionary
        return X_in_grad: numpy array of size (batch_size, vocab_size)
        return H_in_grad: numpy array of size (batch_size, hidden_size)
        """
        assert X_out_grad.shape == self.X_out.shape
        assert H_out_grad.shape == self.H_out.shape

        params_dict["B_v"]["deriv"] += X_out_grad.sum(axis=0)
        params_dict["W_v"]["deriv"] += np.dot(self.H_out.T, X_out_grad)

        # deriv of h_out depends on x_out as well
        dh = np.dot(X_out_grad, params_dict["W_v"]["value"].T)
        dh += H_out_grad
        dH_int = dh * dtanh(self.H_int)

        params_dict["B_f"]["deriv"] += dH_int.sum(axis=0)
        params_dict["W_f"]["deriv"] += np.dot(self.Z.T, dH_int)

        dz = np.dot(dH_int, params_dict["W_f"]["value"].T)
        X_in_grad = dz[:, : self.X_in.shape[1]]
        H_in_grad = dz[:, self.X_in.shape[1] :]
        assert X_in_grad.shape == self.X_in
        assert H_in_grad.shape == self.H_in
        return X_in_grad, H_in_grad


class RNNLayer:
    def __init__(
        self, hidden_size: int, output_size: int, weight_scale: float | None = None
    ):
        """
        param hidden_size: int
        param output_size: int
        param weight_scale: float or None
        """
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weight_scale = weight_scale
        self.start_H = np.zeros((1, hidden_size))
        self.first = True

    def _init_params(self, input_: np.ndarray):
        self.vocab_size = input_.shape[2]
        if not self.weight_scale:
            self.weight_scale = 2.0 / (self.vocab_size + self.output_size)
        self.params = {}
        self.params["W_f"] = {}
        self.params["B_f"] = {}
        self.params["W_v"] = {}
        self.params["B_v"] = {}
        self.params["W_f"]["value"] = np.random.normal(
            loc=0.0,
            scale=self.weight_scale,
            size=(self.hidden_size + self.vocab_size, self.hidden_size),
        )
        self.params["B_f"]["value"] = np.random.normal(
            loc=0.0,
            scale=self.weight_scale,
            size=(1, self.hidden_size),
        )
        self.params["W_v"]["value"] = np.random.normal(
            loc=0.0,
            scale=self.weight_scale,
            size=(self.hidden_size, self.output_size),
        )
        self.params["B_v"]["value"] = np.random.normal(
            loc=0.0,
            scale=self.weight_scale,
            size=(1, self.output_size),
        )
        self.params["W_f"]["deriv"] = np.zeros_like(self.params["W_f"]["value"])
        self.params["B_f"]["deriv"] = np.zeros_like(self.params["B_f"]["value"])
        self.params["W_v"]["deriv"] = np.zeros_like(self.params["W_v"]["value"])
        self.params["B_v"]["deriv"] = np.zeros_like(self.params["B_v"]["value"])

        self.cells = [RNNNode() for x in range(input_.shape[1])]
    
    def _clear_gradients(self):
        for key in self.params.keys():
            self.params[key]['deriv'] = np.zeros_like(self.params[key]['deriv'])
    
    def forward(self, x_seq_in: np.ndarray) -> np.ndarray:
        '''
        param x_seq_in: numpy array of size (batch_size, sequence_length, vocab_size)
        return x_seq_out: numpy array of size (batch_size, sequence_length, output_size)
        '''
        assert x_seq_in.ndim == 3
        if self.first:
            self._init_params(x_seq_in)
            self.first = False
        batch_size = x_seq_in.shape[0]
        H_in = np.copy(self.start_H)
        H_in = np.repeat(H_in, batch_size, axis=0)
        sequence_length = x_seq_in.shape[1]
        x_seq_out = np.zeros((batch_size, sequence_length, self.output_size))
        for t in range(sequence_length):
            x_in = x_seq_in[:, t, :]
            x_out, H_out = self.cells[t].forward(x_in, H_in, self.params)
            x_seq_out[:, t, :] = x_out
            H_in = H_out
        self.start_H = H_in.mean(axis=0, keepdims=True)
        return x_seq_out
    
    def backward(self, x_seq_out_grad: np.ndarray):
        '''
        param loss_grad: numpy array of size (batch_size, sequence_length, vocab_size)
        return loss_grad_out: numpy array of size (batch_size, sequence_length, vocab_size)
        '''
        assert x_seq_out_grad.ndim == 3
        batch_size = x_seq_out_grad.shape[0]
        sequence_length = x_seq_out_grad.shape[1]
        h_in_grad = np.zeros((batch_size, self.hidden_size))
        x_seq_in_grad = np.zeros((batch_size, sequence_length, self.vocab_size))
        for t in reversed(range(sequence_length)):
            x_out_grad = x_seq_out_grad[:, t, :]
            grad_out, h_in_grad = self.cells[t].backward(x_out_grad, h_in_grad, self.params)
            x_seq_in_grad[:, t, :] = grad_out
        return x_seq_in_grad



if __name__ == "__main__":

    X_in = np.random.rand(32, 1000)
    H_in = np.random.rand(32, 150)
    Z = np.column_stack((X_in, H_in))
    print(Z.shape)
