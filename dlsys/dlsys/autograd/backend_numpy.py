import numpy as np


class Device:
    """Baseclass"""


class CPUDevice(Device):
    """Represents data in cpu"""

    def __repr__(self):
        return "cpu"

    def __hash__(self):
        return self._repr__().__hash__()

    def __eq__(self, other):
        return isinstance(other, CPUDevice)

    def enabled(self):
        return True

    def zeros(self, *shape, dtype="float32"):
        return np.zeros(shape, dtype=dtype)

    def ones(self, *shape, dtype="float32"):
        return np.ones(shape, dtype=dtype)

    def randn(self, *shape):  # numpy doesn't support types
        return np.random.randn(*shape)

    def rand(self, *shape):  # numpy doesn't support types
        return np.random.rand(*shape)

    def one_hot(self, n, i, dtype="float32"):
        return np.eye(n, dtype=dtype)[i]

    def empty(self, shape, dtype="float32"):
        return np.empty(shape, dtype=dtype)

    def full(self, shape, fill_value, dtype="float32"):
        return np.full(shape, fill_value, dtype=dtype)


def cpu():
    return CPUDevice()


def default_device():
    return cpu()


def all_devices():
    return [cpu()]
