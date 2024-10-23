import numpy


class Op:
    def __call__(self, *args):
        raise NotImplementedError()

    def compute(self, *args: tuple[NDArray]):
        raise NotImplementedError()


class TensorOp(Op):
    pass


class TensorTupleOp(Op):
    pass


class Value:
    def __init__(self, data, device, dtype, requires_grad):
        pass


class TensorTuple(Value):
    pass


class Tensor(Value):
    def __init__(self):
        pass


def compute_gradient_of_variables(output_tensor, out_grad):
    pass


def find_topo_sort(node_list: list[Value]) -> list[Value]:
    pass


def topo_sort_dfs(node, visited, topo_order):
    pass


def sum_node_list(node_list):
    pass
