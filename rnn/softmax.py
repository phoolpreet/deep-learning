import numpy as np


class Softmax:
    def predict(self, x):
        exp_scores = np.exp(x)
        return exp_scores / np.sum(exp_scores)

    def loss(self, x, y):
        probs = self.predict(x)
        return -np.log(probs[y])

    def diff(self, x, y):
        probs = self.predict(x)
        probs[y] -= 1.0
        return probs


if __name__ == "__main__":

    a = np.random.randn(4)
    sft = Softmax()
    p = sft.predict(a)
    l = sft.loss(a, 1)
    print(p)
    print(l)
